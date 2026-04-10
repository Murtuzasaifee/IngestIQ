"""
parsers/azure_di_parser.py
--------------------------
Azure Document Intelligence backend for document parsing.

Uses the prebuilt-layout model to extract paragraphs, tables, and figures
from a local PDF file, then maps them to the shared ParseResult contract.

The PDF is sent directly to the Azure API as bytes — no cloud storage required.

Processing strategy:
    Always processes one page at a time using PyMuPDF to extract single-page
    PDFs and submit each separately. This solves the 4MB inline limit, prevents
    complex-table pages from being silently dropped, and ensures page_number
    mapping is always correct.

Required env vars:
    AZURE_DI_ENDPOINT  — e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_DI_KEY       — your Azure DI API key

Install: uv add azure-ai-documentintelligence
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF — used for per-page PDF splitting
from PIL import Image

from parsers.base import (
    BaseDocumentParser,
    ParsedElement,
    PageResult,
    ParseResult,
    crop_base64,
    rasterize_pdf,
)

logger = logging.getLogger(__name__)

# Azure DI ParagraphRole.value → normalised ParsedElement label.
# Keys must match para.role.value (e.g. "sectionHeading"), NOT str(para.role)
# which returns the enum repr "ParagraphRole.SECTION_HEADING".
_ROLE_TO_LABEL: dict[str, str] = {
    "title":          "title",
    "sectionHeading": "section_title",
    "pageHeader":     "page_header",
    "pageFooter":     "page_footer",
    "pageNumber":     "page_number",
    "footnote":       "text",
    "formulaBlock":   "text",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _role_value(role) -> str:
    """Extract the string value from a ParagraphRole enum or plain string.

    Azure DI SDK returns para.role as a ParagraphRole enum whose str() repr is
    "ParagraphRole.SECTION_HEADING", not "sectionHeading". We must use .value
    to get the actual string that matches _ROLE_TO_LABEL keys.
    """
    if role is None:
        return ""
    return getattr(role, "value", None) or str(role)


def _polygon_to_bbox(
    polygon: List[float],
    page_width: float,
    page_height: float,
) -> Optional[dict]:
    """Convert Azure DI polygon (flat x,y list in page units) to normalised bbox dict."""
    if not polygon or page_width <= 0 or page_height <= 0:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return {
        "left":   round(min(xs) / page_width,               4),
        "top":    round(min(ys) / page_height,              4),
        "width":  round((max(xs) - min(xs)) / page_width,   4),
        "height": round((max(ys) - min(ys)) / page_height,  4),
    }


def _first_bbox_and_page(item, page_map: dict) -> Tuple[Optional[dict], int]:
    """Return (normalised_bbox, page_number) from the item's first bounding region."""
    regions = getattr(item, "bounding_regions", None) or []
    if not regions:
        return None, 1
    region   = regions[0]
    page_num = region.page_number
    page     = page_map.get(page_num)
    if page is None:
        return None, page_num
    return _polygon_to_bbox(region.polygon or [], page.width, page.height), page_num


def _table_to_markdown(table) -> str:
    """Render an Azure DI DocumentTable as a Markdown table string."""
    if not table.column_count or not table.row_count:
        return ""
    grid = [[""] * table.column_count for _ in range(table.row_count)]
    for cell in table.cells or []:
        cell_text = (cell.content or "").replace("\n", " ").strip()
        grid[cell.row_index][cell.column_index] = cell_text
    lines: List[str] = []
    for i, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "|".join(["---"] * table.column_count) + "|")
    return "\n".join(lines)


def _build_table_span_intervals(result) -> List[Tuple[int, int]]:
    """Collect character offset intervals covered by all tables in a result."""
    intervals: List[Tuple[int, int]] = []
    for table in result.tables or []:
        for span in table.spans or []:
            intervals.append((span.offset, span.offset + span.length))
    return intervals


def _overlaps_table(para, table_intervals: List[Tuple[int, int]]) -> bool:
    """True if any of the paragraph's character spans overlap a table span."""
    for p_span in para.spans or []:
        p_start = p_span.offset
        p_end   = p_span.offset + p_span.length
        for t_start, t_end in table_intervals:
            if p_start < t_end and p_end > t_start:
                return True
    return False


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class AzureDocumentIntelligenceParser(BaseDocumentParser):
    """Document parser backed by Azure Document Intelligence (prebuilt-layout).

    Sends the local PDF directly to the Azure API as bytes (no S3/blob storage
    required), then maps the structured result to ParsedElement / ParseResult.

    Always processes one page at a time: each page is extracted as a single-page
    PDF via PyMuPDF and submitted separately. This avoids the 4MB inline limit,
    prevents complex-table pages from being silently dropped, and ensures
    page_number mapping is always correct.
    """

    def __init__(self, endpoint: str, api_key: str) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key  = api_key

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str) -> ParseResult:
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:
            raise RuntimeError(
                "azure-ai-documentintelligence is not installed. "
                "Run: uv add azure-ai-documentintelligence"
            ) from exc

        logger.info("Starting Azure DI analysis: %s", pdf_path)

        # Rasterize first — gives us the true page count and images for crops.
        page_images = rasterize_pdf(pdf_path)
        n_pages     = len(page_images)

        client = DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )

        logger.info("PDF: %d pages — processing page by page", n_pages)

        pages_elements: dict[int, List[ParsedElement]] = {}

        with fitz.open(pdf_path) as src:
            for pg_idx in range(n_pages):
                pg_num     = pg_idx + 1
                single_doc = fitz.open()
                single_doc.insert_pdf(src, from_page=pg_idx, to_page=pg_idx)
                page_bytes = single_doc.tobytes()
                single_doc.close()

                logger.info("Page %d/%d — %d bytes", pg_num, n_pages, len(page_bytes))
                result = self._call_azure_di(client, page_bytes, "1")
                logger.info(
                    "  → %d paragraphs | %d tables | %d figures",
                    len(result.paragraphs or []),
                    len(result.tables or []),
                    len(result.figures or []),
                )
                self._process_result(result, page_images, pages_elements, page_num_override=pg_num)

        # --- Assemble PageResults sorted by reading order ---
        page_results: List[PageResult] = []
        for pg_num in sorted(pages_elements):
            elements = pages_elements[pg_num]
            elements.sort(key=lambda e: (
                e.bbox["top"]  if e.bbox else 0.0,
                e.bbox["left"] if e.bbox else 0.0,
            ))
            for i, el in enumerate(elements):
                el.reading_order = i

            page_results.append(PageResult(
                page_number=pg_num,
                elements=elements,
            ))
            label_counts: Counter = Counter(el.label for el in elements)
            logger.info("Page %d: %d elements %s", pg_num, len(elements), dict(label_counts))

        total = sum(len(p.elements) for p in page_results)
        logger.info("Azure DI parse complete. Total elements: %d", total)
        return ParseResult(
            source_file=str(Path(pdf_path).resolve()),
            pages=page_results,
            total_elements=total,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_azure_di(self, client, pdf_bytes: bytes, pages: str) -> Any:
        """Submit pdf_bytes to Azure DI and return the AnalyzeResult.

        MARKDOWN output format is required: it does not gate page inclusion on
        successful table cell-grid resolution, so complex-table pages are never
        silently dropped (unlike the default TEXT mode).
        """
        from azure.ai.documentintelligence.models import DocumentContentFormat
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=pdf_bytes,
            content_type="application/octet-stream",
            pages=pages,
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        return poller.result()

    def _process_result(
        self,
        result: Any,
        page_images: Dict[int, Image.Image],
        pages_elements: Dict[int, List[ParsedElement]],
        page_num_override: Optional[int] = None,
    ) -> None:
        """Map an Azure DI AnalyzeResult into pages_elements.

        page_num_override: when set, all elements from this result are assigned
        to this page number. Used when processing single-page PDFs where Azure
        DI always returns page_number=1 regardless of the actual document page.
        """
        page_map        = {p.page_number: p for p in result.pages or []}
        table_intervals = _build_table_span_intervals(result)

        role_counts: Counter = Counter(_role_value(p.role) or "body" for p in result.paragraphs or [])
        logger.debug("Paragraph roles: %s", dict(role_counts))

        # --- Paragraphs ---
        for para in result.paragraphs or []:
            if _overlaps_table(para, table_intervals):
                continue
            text = (para.content or "").strip()
            if not text:
                continue
            label        = _ROLE_TO_LABEL.get(_role_value(para.role), "text")
            bbox, pg_num = _first_bbox_and_page(para, page_map)
            actual_pg    = page_num_override or pg_num
            pages_elements.setdefault(actual_pg, []).append(ParsedElement(
                label=label, text=text, bbox=bbox,
                score=1.0, reading_order=0, page_number=actual_pg,
            ))

        # --- Tables ---
        for table in result.tables or []:
            bbox, pg_num = _first_bbox_and_page(table, page_map)
            actual_pg    = page_num_override or pg_num
            markdown     = _table_to_markdown(table)
            if table.caption and (table.caption.content or "").strip():
                markdown = f"**{table.caption.content.strip()}**\n\n{markdown}"
            logger.debug("Table page=%d rows=%d cols=%d", actual_pg, table.row_count, table.column_count)
            pages_elements.setdefault(actual_pg, []).append(ParsedElement(
                label="table", text=markdown, bbox=bbox,
                score=1.0, reading_order=0, page_number=actual_pg,
                image_base64=crop_base64(page_images.get(actual_pg), bbox),
            ))

        # --- Figures ---
        for figure in result.figures or []:
            bbox, pg_num = _first_bbox_and_page(figure, page_map)
            actual_pg    = page_num_override or pg_num
            caption_text = ""
            if figure.caption and (figure.caption.content or "").strip():
                caption_text = figure.caption.content.strip()
            pages_elements.setdefault(actual_pg, []).append(ParsedElement(
                label="figure", text=caption_text, bbox=bbox,
                score=1.0, reading_order=0, page_number=actual_pg,
                image_base64=crop_base64(page_images.get(actual_pg), bbox),
            ))

