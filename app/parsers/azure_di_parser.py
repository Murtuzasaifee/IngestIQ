"""
parsers/azure_di_parser.py
--------------------------
Azure Document Intelligence backend for document parsing.

Uses the prebuilt-layout model to extract paragraphs, tables, and figures
from a local PDF file, then maps them to the shared ParseResult contract.

The PDF is sent directly to the Azure API as bytes — no cloud storage required.

Required env vars:
    AZURE_DI_ENDPOINT  — e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_DI_KEY       — your Azure DI API key

Install: uv add azure-ai-documentintelligence
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from parsers.base import (
    BaseDocumentParser,
    ParsedElement,
    PageResult,
    ParseResult,
    assemble_markdown,
    crop_base64,
)

logger = logging.getLogger(__name__)

# Azure DI ParagraphRole string → normalised ParsedElement label
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

def _polygon_to_bbox(
    polygon: List[float],
    page_width: float,
    page_height: float,
) -> Optional[dict]:
    """Convert Azure DI polygon (flat x,y list in page units) to normalised bbox dict.

    Azure DI returns coordinates in inches for PDFs. Dividing by page dimensions
    normalises them to the 0–1 range expected by crop_base64 and Qdrant payloads.
    """
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
    """Render an Azure DI DocumentTable as a Markdown table string.

    Builds a 2-D grid from cells (handles merged cells by writing to the
    top-left slot only), then serialises with a separator after the header row.
    """
    grid = [[""] * table.column_count for _ in range(table.row_count)]
    for cell in table.cells or []:
        grid[cell.row_index][cell.column_index] = (
            cell.content.replace("\n", " ").strip()
        )
    lines: List[str] = []
    for i, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "|".join(["---"] * table.column_count) + "|")
    return "\n".join(lines)


def _build_table_span_intervals(result) -> List[Tuple[int, int]]:
    """Collect character offset (start, end) intervals covered by all tables.

    Azure DI includes table cell text inside result.paragraphs as well, which
    would create duplicate chunks. We use these intervals to skip those paragraphs.
    """
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

    Paragraph deduplication: Azure DI surfaces table cell content in
    result.paragraphs in addition to result.tables. We filter those out using
    character-span overlap detection before building elements.
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

        client = DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )

        with open(pdf_path, "rb") as f:
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f,
                content_type="application/octet-stream",
            )
        result = poller.result()
        logger.info("Azure DI complete. Pages: %d", len(result.pages or []))

        page_images     = self._rasterize(pdf_path)
        page_map        = {p.page_number: p for p in result.pages or []}
        table_intervals = _build_table_span_intervals(result)

        # Build per-page element lists
        pages_elements: dict[int, List[ParsedElement]] = {}

        # --- Paragraphs (skip those that are inside table spans) ---
        for para in result.paragraphs or []:
            if _overlaps_table(para, table_intervals):
                continue

            text = (para.content or "").strip()
            if not text:
                continue

            label        = _ROLE_TO_LABEL.get(str(para.role or ""), "text")
            bbox, pg_num = _first_bbox_and_page(para, page_map)

            pages_elements.setdefault(pg_num, []).append(ParsedElement(
                label=label,
                text=text,
                bbox=bbox,
                score=1.0,       # Azure DI exposes confidence at word level only
                reading_order=0,
                page_number=pg_num,
            ))

        # --- Tables ---
        for table in result.tables or []:
            bbox, pg_num = _first_bbox_and_page(table, page_map)
            markdown     = _table_to_markdown(table)

            # Table caption → emit as figure_title so chunker links it to the table chunk
            if table.caption and (table.caption.content or "").strip():
                cap_text        = table.caption.content.strip()
                cap_bbox, _     = _first_bbox_and_page(table.caption, page_map)
                pages_elements.setdefault(pg_num, []).append(ParsedElement(
                    label="figure_title",
                    text=cap_text,
                    bbox=cap_bbox or bbox,
                    score=1.0,
                    reading_order=0,
                    page_number=pg_num,
                ))

            pages_elements.setdefault(pg_num, []).append(ParsedElement(
                label="table",
                text=markdown,
                bbox=bbox,
                score=1.0,
                reading_order=0,
                page_number=pg_num,
                image_base64=crop_base64(page_images.get(pg_num), bbox),
            ))

        # --- Figures ---
        for figure in result.figures or []:
            bbox, pg_num = _first_bbox_and_page(figure, page_map)
            page_image   = page_images.get(pg_num)

            # Azure DI caption used as the initial figure text; enrichment
            # will replace/augment it with a GPT vision caption.
            caption_text = ""
            if figure.caption and (figure.caption.content or "").strip():
                caption_text = figure.caption.content.strip()

            pages_elements.setdefault(pg_num, []).append(ParsedElement(
                label="figure",
                text=caption_text,
                bbox=bbox,
                score=1.0,
                reading_order=0,
                page_number=pg_num,
                image_base64=crop_base64(page_image, bbox),
            ))

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
                markdown=assemble_markdown(elements),
            ))
            logger.info("Page %d: %d elements", pg_num, len(elements))

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

    def _rasterize(self, pdf_path: str, dpi: int = 250) -> Dict[int, Image.Image]:
        """Rasterize every page with PyMuPDF for image/table crops."""
        scale  = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        images: Dict[int, Image.Image] = {}
        with fitz.open(pdf_path) as pdf:
            for i in range(len(pdf)):
                pix = pdf[i].get_pixmap(matrix=matrix)
                images[i + 1] = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        logger.info("Rasterized %d pages at %d DPI", len(images), dpi)
        return images
