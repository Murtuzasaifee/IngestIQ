"""
parsers/base.py
---------------
Shared data contract and abstract base class for all document parser backends.

ParsedElement → PageResult → ParseResult is the universal output shape that
chunker.py, enrichment.py, and the rest of the pipeline consume. Every parser
backend (Textract, Azure DI, …) must produce this structure — nothing else in
the pipeline needs to know which backend is active.
"""

import base64
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data contract
# ---------------------------------------------------------------------------

@dataclass
class ParsedElement:
    label: str                         # normalised label: text|table|figure|title|…
    text: str                          # extracted text; Markdown for tables
    bbox: Optional[dict]               # normalised {left, top, width, height}
    score: float                       # parser confidence (0–1)
    reading_order: int                 # 0-based position within page
    page_number: int                   # 1-based page index
    image_base64: Optional[str] = None # JPEG crop as base64 (figure/table elements)


@dataclass
class PageResult:
    page_number: int
    elements: List[ParsedElement]


@dataclass
class ParseResult:
    source_file: str                   # origin URI or local path
    pages: List[PageResult]
    total_elements: int


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

# Labels that carry no retrieval value — skipped by chunker and parsers alike.
# Defined here as the single source of truth; chunker.py imports this set.
_SKIP_LABELS = {"page_header", "page_footer", "page_number"}


def bbox_dict(bbox) -> Optional[dict]:
    """Normalise a Textract bounding box object to a plain dict."""
    if bbox is None:
        return None
    return {
        "left":   round(bbox.x, 4),
        "top":    round(bbox.y, 4),
        "width":  round(bbox.width, 4),
        "height": round(bbox.height, 4),
    }


def crop_base64(page_image: Image.Image, bbox: Optional[dict]) -> Optional[str]:
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


def rasterize_pdf(pdf_path: str, dpi: int = 250) -> Dict[int, Image.Image]:
    """Rasterize every page of a local PDF with PyMuPDF (no poppler required).

    Returns a 1-based dict mapping page_number → PIL Image. Used by all parser
    backends for bbox-based image/table crops.
    """
    scale  = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    images: Dict[int, Image.Image] = {}
    with fitz.open(pdf_path) as pdf:
        for i in range(len(pdf)):
            pix = pdf[i].get_pixmap(matrix=matrix)
            images[i + 1] = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    logger.info("Rasterized %d pages at %d DPI", len(images), dpi)
    return images


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDocumentParser(ABC):
    """Contract that every document parser backend must satisfy.

    Implementations receive a local PDF path and return a backend-agnostic
    ParseResult. All backend-specific concerns (S3 upload, API calls, local
    subprocesses) are encapsulated inside the subclass.
    """

    @abstractmethod
    def parse(self, pdf_path: str) -> ParseResult:
        """Parse a local PDF and return a structured ParseResult.

        Args:
            pdf_path: Absolute or relative path to a local PDF file.

        Returns:
            ParseResult containing per-page elements in reading order.
        """
