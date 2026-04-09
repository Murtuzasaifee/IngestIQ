"""
document_processor.py
---------------------
AWS Textract OCR document parser.

Calls Textract (LAYOUT + TABLES features) and returns a ParseResult containing:
  source_file    — S3 URI of the source document.
  pages          — Per-page ParsedElement lists and assembled Markdown.
  total_elements — Sum of all elements across all pages.

Page rasterization (for LAYOUT_FIGURE crops) is done with PyMuPDF (fitz)
so that the poppler system binary is NOT required.
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import boto3
import fitz  # PyMuPDF
from PIL import Image
from textractor import Textractor
from textractor.data.constants import TextractFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label map — Textract block type → normalised element label
# LAYOUT_TABLE is intentionally absent: TABLE objects supply richer markdown
# ---------------------------------------------------------------------------

_BLOCK_TO_LABEL: dict[str, str] = {
    "LAYOUT_TEXT":           "text",
    "LAYOUT_TITLE":          "title",
    "LAYOUT_SECTION_HEADER": "section_title",
    "LAYOUT_LIST":           "list",
    "LAYOUT_FIGURE":         "figure",
    "LAYOUT_HEADER":         "page_header",
    "LAYOUT_FOOTER":         "page_footer",
    "LAYOUT_PAGE_NUMBER":    "page_number",
    "LAYOUT_KEY_VALUE_SET":  "key_value",
}

_SKIP_LABELS = {"page_header", "page_footer", "page_number"}

_MARKDOWN_PREFIX: dict[str, str] = {
    "title":         "# ",
    "section_title": "## ",
    "figure_title":  "**",
}
_MARKDOWN_SUFFIX: dict[str, str] = {
    "figure_title": "**",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ParsedElement:
    label: str                    # normalised label (see _BLOCK_TO_LABEL above)
    text: str                     # extracted text; Markdown for tables
    bbox: Optional[dict]          # normalised {left, top, width, height}
    score: float                  # Textract confidence (0–1)
    reading_order: int            # 0-based position within page (assigned after bbox sort)
    page_number: int              # 1-based page index
    image_base64: Optional[str] = None   # JPEG crop as base64 (figure elements only)


@dataclass
class PageResult:
    page_number: int
    elements: List[ParsedElement]
    markdown: str                 # assembled Markdown from elements (used for chunk metadata)


@dataclass
class ParseResult:
    source_file: str              # S3 URI of the original document
    pages: List[PageResult]
    total_elements: int           # sum of all elements across all pages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_block_type(block) -> str:
    """Defensively resolve Textract block type across textractor versions."""
    return (
        getattr(block, "layout_type", None)
        or getattr(block, "block_type", None)
        or (block.raw_object.get("BlockType", "LAYOUT_TEXT")
            if hasattr(block, "raw_object") else "LAYOUT_TEXT")
    )


def _bbox_dict(bbox) -> Optional[dict]:
    if bbox is None:
        return None
    return {
        "left":   round(bbox.x, 4),
        "top":    round(bbox.y, 4),
        "width":  round(bbox.width, 4),
        "height": round(bbox.height, 4),
    }


def _crop_base64(page_image: Image.Image, bbox: Optional[dict]) -> Optional[str]:
    """Crop a bbox region from a page image and return it as a base64 JPEG."""
    if not page_image or not bbox:
        return None
    w, h = page_image.size
    left   = int(bbox["left"] * w)
    top    = int(bbox["top"] * h)
    right  = int((bbox["left"] + bbox["width"]) * w)
    bottom = int((bbox["top"] + bbox["height"]) * h)
    if right <= left or bottom <= top:
        return None
    buf = io.BytesIO()
    page_image.crop((left, top, right, bottom)).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _assemble_markdown(elements: List[ParsedElement]) -> str:
    """Convert a list of parsed elements into a Markdown string.

    Args:
        elements: Parsed elements, each with label, text, bbox, score, reading_order.

    Returns:
        Assembled Markdown string with elements joined by double newlines.
    """
    parts: List[str] = []
    for el in elements:
        if el.label in _SKIP_LABELS or not el.text.strip():
            continue
        prefix = _MARKDOWN_PREFIX.get(el.label, "")
        suffix = _MARKDOWN_SUFFIX.get(el.label, "")
        if el.label == "figure":
            parts.append(f"[Figure — page {el.page_number}]")
        else:
            parts.append(f"{prefix}{el.text.strip()}{suffix}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Rasterization — PyMuPDF (no poppler needed)
# ---------------------------------------------------------------------------

def _rasterize_pdf_from_s3(
    s3_bucket: str,
    s3_key: str,
    aws_region: str,
    dpi: int = 250,
) -> Dict[int, Image.Image]:
    """Download PDF from S3 and rasterize every page to a PIL Image.

    Returns a dict keyed by 1-based page number.
    Uses PyMuPDF (fitz) — no poppler system binary required.
    """
    s3 = boto3.client("s3", region_name=aws_region)
    pdf_bytes = s3.get_object(Bucket=s3_bucket, Key=s3_key)["Body"].read()
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    images: Dict[int, Image.Image] = {}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for i in range(len(pdf)):
            pix = pdf[i].get_pixmap(matrix=matrix)
            images[i + 1] = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    logger.info("Rasterized %d pages at %d DPI", len(images), dpi)
    return images


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_document_from_s3(
    s3_bucket: str,
    s3_key: str,
    aws_region: str,
) -> ParseResult:
    """Parse a PDF from S3 using Amazon Textract.

    Args:
        s3_bucket: S3 bucket containing the PDF.
        s3_key: S3 object key of the PDF.
        aws_region: AWS region for both Textract and S3.

    Returns:
        ParseResult with:
          source_file: Path to the original document (S3 URI).
          pages: Per-page parsing results (elements + assembled Markdown).
          total_elements: Sum of all elements across all pages.
    """
    s3_path = f"s3://{s3_bucket}/{s3_key}"
    logger.info("Starting Textract analysis: %s", s3_path)

    # Credentials are read from env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
    # populated by load_dotenv() in main.py before this is called.
    extractor = Textractor(region_name=aws_region)
    document = extractor.start_document_analysis(
        file_source=s3_path,
        features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES],
        save_image=False,  # PyMuPDF rasterizes pages — do NOT set True (needs poppler)
    )
    logger.info("Textract complete. Pages: %d", document.num_pages)

    page_images = _rasterize_pdf_from_s3(s3_bucket, s3_key, aws_region)
    page_results: List[PageResult] = []

    for page in document.pages:
        page_num = page.page_num          # 1-based
        page_image = page_images.get(page_num)
        raw_elements: List[ParsedElement] = []

        # --- Layout blocks (text, titles, figures; LAYOUT_TABLE skipped) ---
        for layout in page.layouts:
            bt = _get_block_type(layout)

            # Skip LAYOUT_TABLE — TABLE objects below provide structured markdown
            if bt == "LAYOUT_TABLE":
                continue

            label = _BLOCK_TO_LABEL.get(bt, "text")
            text = layout.get_text().strip()
            bbox = _bbox_dict(layout.bbox)
            image_b64 = None

            if label == "figure":
                image_b64 = _crop_base64(page_image, bbox)
                # Figures may have no text; still emit the element for the image crop
            elif not text:
                continue

            raw_elements.append(ParsedElement(
                label=label,
                text=text,
                bbox=bbox,
                score=round(getattr(layout, "confidence", 1.0) or 1.0, 4),
                reading_order=0,          # assigned after merge + sort below
                page_number=page_num,
                image_base64=image_b64,
            ))

        # --- Table blocks (structured markdown rendering) ---
        for table in page.tables:
            try:
                text = table.to_markdown()
            except Exception:
                text = table.get_text()

            raw_elements.append(ParsedElement(
                label="table",
                text=text,
                bbox=_bbox_dict(table.bbox),
                score=round(getattr(table, "confidence", 1.0) or 1.0, 4),
                reading_order=0,
                page_number=page_num,
            ))

        # Sort by vertical then horizontal position to restore reading order
        # (merges layout and table elements into one coherent sequence)
        raw_elements.sort(key=lambda e: (
            e.bbox["top"]  if e.bbox else 0.0,
            e.bbox["left"] if e.bbox else 0.0,
        ))
        for i, el in enumerate(raw_elements):
            el.reading_order = i

        page_results.append(PageResult(
            page_number=page_num,
            elements=raw_elements,
            markdown=_assemble_markdown(raw_elements),
        ))
        logger.info("Page %d: %d elements", page_num, len(raw_elements))

    total = sum(len(p.elements) for p in page_results)
    logger.info("Parse complete. Total elements: %d", total)
    return ParseResult(source_file=s3_path, pages=page_results, total_elements=total)
