"""
parsers/base.py
---------------
Shared data contract and abstract base class for all document parser backends.

ParsedElement → PageResult → ParseResult is the universal output shape that
chunker.py, enrichment.py, and the rest of the pipeline consume. Every parser
backend (Textract, Docling, …) must produce this structure — nothing else in
the pipeline needs to know which backend is active.
"""

import base64
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    markdown: str                      # assembled Markdown from elements


@dataclass
class ParseResult:
    source_file: str                   # origin URI or local path
    pages: List[PageResult]
    total_elements: int


# ---------------------------------------------------------------------------
# Shared utilities (used by multiple backends)
# ---------------------------------------------------------------------------

def bbox_dict(bbox) -> Optional[dict]:
    """Normalise a bounding box object to a plain dict."""
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


_SKIP_LABELS = {"page_header", "page_footer", "page_number"}

_MARKDOWN_PREFIX: dict[str, str] = {
    "title":         "# ",
    "section_title": "## ",
    "figure_title":  "**",
}
_MARKDOWN_SUFFIX: dict[str, str] = {
    "figure_title": "**",
}


def assemble_markdown(elements: List[ParsedElement]) -> str:
    """Convert a list of parsed elements into a Markdown string."""
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
