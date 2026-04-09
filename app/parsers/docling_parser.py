"""
parsers/docling_parser.py
-------------------------
Docling backend for document parsing (local, no cloud required).

To activate:
    1. Set DOCUMENT_PARSER=docling in .env
    2. Install: uv add docling

Docling processes PDFs locally and supports layout analysis, table extraction,
and figure detection — producing the same ParseResult contract as TextractParser.
"""

import logging
from pathlib import Path
from typing import Dict, List

from parsers.base import (
    BaseDocumentParser,
    ParsedElement,
    PageResult,
    ParseResult,
    assemble_markdown,
    crop_base64,
)

logger = logging.getLogger(__name__)

# Docling label → normalised ParsedElement label
_DOCLING_TO_LABEL: dict[str, str] = {
    "text":             "text",
    "paragraph":        "text",
    "section_header":   "section_title",
    "title":            "title",
    "list_item":        "list",
    "table":            "table",
    "figure":           "figure",
    "figure_caption":   "figure_title",
    "page_header":      "page_header",
    "page_footer":      "page_footer",
    "footnote":         "text",
    "formula":          "text",
}


class DoclingParser(BaseDocumentParser):
    """Document parser backed by Docling (local, CPU/GPU, no S3 required).

    Docling converts PDFs to a structured document model. This parser maps
    Docling's DocItem types to ParsedElement labels and produces a ParseResult
    identical in shape to TextractParser's output.
    """

    def parse(self, pdf_path: str) -> ParseResult:
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed. Run: uv add docling"
            ) from exc

        logger.info("Starting Docling analysis: %s", pdf_path)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        converter = DocumentConverter()
        result    = converter.convert(pdf_path)
        doc       = result.document

        # Group items by page
        pages_map: Dict[int, List[ParsedElement]] = {}

        for item, _ in doc.iterate_items():
            page_num = getattr(item.prov[0], "page_no", 1) if item.prov else 1
            label    = _DOCLING_TO_LABEL.get(item.label.lower(), "text")

            text = ""
            if hasattr(item, "text"):
                text = item.text or ""
            elif hasattr(item, "export_to_markdown"):
                text = item.export_to_markdown() or ""

            bbox = None
            if item.prov:
                b = item.prov[0].bbox
                if b:
                    # Docling bbox is (l, t, r, b) in page coordinates; normalise
                    bbox = {
                        "left":   round(b.l / b.coord_origin.width, 4)  if hasattr(b, "coord_origin") else 0.0,
                        "top":    round(b.t / b.coord_origin.height, 4) if hasattr(b, "coord_origin") else 0.0,
                        "width":  round((b.r - b.l) / b.coord_origin.width, 4)  if hasattr(b, "coord_origin") else 0.0,
                        "height": round((b.b - b.t) / b.coord_origin.height, 4) if hasattr(b, "coord_origin") else 0.0,
                    }

            el = ParsedElement(
                label=label,
                text=text.strip(),
                bbox=bbox,
                score=1.0,          # Docling does not expose per-element confidence
                reading_order=0,    # assigned after sort below
                page_number=page_num,
                image_base64=None,  # populated below for figure elements
            )
            pages_map.setdefault(page_num, []).append(el)

        page_results: List[PageResult] = []
        for page_num in sorted(pages_map):
            elements = pages_map[page_num]
            elements.sort(key=lambda e: (
                e.bbox["top"]  if e.bbox else 0.0,
                e.bbox["left"] if e.bbox else 0.0,
            ))
            for i, el in enumerate(elements):
                el.reading_order = i

            page_results.append(PageResult(
                page_number=page_num,
                elements=elements,
                markdown=assemble_markdown(elements),
            ))
            logger.info("Page %d: %d elements", page_num, len(elements))

        total = sum(len(p.elements) for p in page_results)
        logger.info("Docling parse complete. Total elements: %d", total)
        return ParseResult(
            source_file=str(Path(pdf_path).resolve()),
            pages=page_results,
            total_elements=total,
        )
