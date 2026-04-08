"""
document_processor.py
---------------------
Extracts chunks from a multi-page PDF using Amazon Textract (LAYOUT + TABLES features).
Each chunk carries modality metadata (text / table / image) plus bounding-box geometry.

Multi-page PDFs must be uploaded to S3 first; Textract's async StartDocumentAnalysis
API is used transparently by Textractor when a file is passed via S3 path.

Page rasterization (for LAYOUT_FIGURE crops) is done with PyMuPDF (fitz) so that
poppler is not required as a system dependency.
"""

import io
import base64
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import boto3
import fitz  # PyMuPDF
from PIL import Image

from textractor import Textractor
from textractor.data.constants import TextractFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Represents a single extracted chunk from the document."""

    chunk_id: str                          # Unique identifier: "p<page>_<index>"
    modality: str                          # "text" | "table" | "image"
    chunk_text: str                        # Raw text content
    page_number: int                       # 1-based page index
    image_base64: Optional[str] = None    # Base64-encoded crop (image chunks only)
    image_caption: Optional[str] = None   # GPT-generated caption (image chunks only)
    bbox: Optional[dict] = None           # Normalised bounding box {left,top,width,height}
    layout_type: Optional[str] = None     # Textract layout block type (e.g. LAYOUT_FIGURE)
    extra_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_to_dict(bbox) -> Optional[dict]:
    """Convert a Textractor BoundingBox to a plain dict (normalised 0-1 coords)."""
    if bbox is None:
        return None
    return {
        "left":   round(bbox.x, 4),
        "top":    round(bbox.y, 4),
        "width":  round(bbox.width, 4),
        "height": round(bbox.height, 4),
    }


def _crop_page_region(page_image: Image.Image, bbox: dict) -> Optional[str]:
    """
    Crop a region from a PIL page image using normalised bbox coords and return
    the result as a base64-encoded JPEG string.
    """
    if page_image is None or bbox is None:
        return None

    w, h = page_image.size
    left   = int(bbox["left"]   * w)
    top    = int(bbox["top"]    * h)
    right  = int((bbox["left"] + bbox["width"])  * w)
    bottom = int((bbox["top"]  + bbox["height"]) * h)

    # Guard against degenerate boxes
    if right <= left or bottom <= top:
        return None

    cropped = page_image.crop((left, top, right, bottom))
    buffer  = io.BytesIO()
    cropped.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _rasterize_pdf_from_s3(
    s3_bucket: str,
    s3_key: str,
    aws_region: str,
    dpi: int = 250,
) -> Dict[int, Image.Image]:
    """
    Download a PDF from S3 and rasterize every page to a PIL Image using PyMuPDF.
    Returns a dict keyed by 1-based page number.

    Uses PyMuPDF (fitz) so that poppler is NOT required as a system dependency.
    """
    s3 = boto3.client("s3", region_name=aws_region)
    pdf_bytes = s3.get_object(Bucket=s3_bucket, Key=s3_key)["Body"].read()

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    page_images: Dict[int, Image.Image] = {}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_index in range(len(pdf)):
            pix = pdf[page_index].get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images[page_index + 1] = img  # 1-based

    logger.info("Rasterized %d pages at %d DPI using PyMuPDF", len(page_images), dpi)
    return page_images


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_chunks_from_s3(
    s3_bucket: str,
    s3_key: str,
    aws_region: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> List[DocumentChunk]:
    """
    Extract layout-aware chunks from a PDF already on S3 using Amazon Textract.

    Parameters
    ----------
    s3_bucket : str
        S3 bucket where the PDF resides.
    s3_key : str
        S3 object key of the PDF.
    aws_region : str
        AWS region for both S3 and Textract.
    aws_access_key_id / aws_secret_access_key : str, optional
        Unused — credentials are read from environment variables set by load_dotenv().

    Returns
    -------
    List[DocumentChunk]
    """
    # ------------------------------------------------------------------
    # Initialise Textractor — credentials are picked up automatically
    # from environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
    # which load_dotenv() has already populated before this is called.
    # Textractor only accepts profile_name / region_name, not explicit creds.
    # ------------------------------------------------------------------
    extractor = Textractor(region_name=aws_region)

    s3_path = f"s3://{s3_bucket}/{s3_key}"
    logger.info("Starting Textract async analysis on %s", s3_path)

    # save_image=False — we rasterize pages ourselves with PyMuPDF to avoid
    # a dependency on the poppler system binary (required by pdf2image).
    document = extractor.start_document_analysis(
        file_source=s3_path,
        features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES],
        save_image=False,
    )

    logger.info("Textract completed. Pages: %d", document.num_pages)

    # Rasterize pages with PyMuPDF for LAYOUT_FIGURE cropping
    page_images = _rasterize_pdf_from_s3(s3_bucket, s3_key, aws_region)

    chunks: List[DocumentChunk] = []

    for page in document.pages:
        page_num    = page.page_num                    # 1-based
        page_image  = page_images.get(page_num)        # PIL.Image from PyMuPDF
        chunk_index = 0

        # ---------------------------------------------------------------
        # 1. Layout-level blocks (paragraphs, titles, headers, figures …)
        # ---------------------------------------------------------------
        for layout in page.layouts:
            # Defensive access: attribute name varies slightly across textractor versions.
            # Priority: layout_type -> block_type -> raw BlockType from the API response.
            block_type = (
                getattr(layout, "layout_type", None)
                or getattr(layout, "block_type", None)
                or (layout.raw_object.get("BlockType", "LAYOUT_TEXT") if hasattr(layout, "raw_object") else "LAYOUT_TEXT")
            )
            text       = layout.get_text().strip()
            bbox_dict  = _bbox_to_dict(layout.bbox)

            # Determine modality from Textract layout block type
            if "FIGURE" in block_type:
                modality  = "image"
                image_b64 = _crop_page_region(page_image, bbox_dict)
            else:
                modality  = "text"
                image_b64 = None

            if not text and modality == "text":
                continue  # Skip empty text blocks

            chunk = DocumentChunk(
                chunk_id     = f"p{page_num}_{chunk_index}",
                modality     = modality,
                chunk_text   = text,
                page_number  = page_num,
                image_base64 = image_b64,
                bbox         = bbox_dict,
                layout_type  = block_type,
                extra_metadata={
                    "source":     s3_path,
                    "confidence": getattr(layout, "confidence", None),
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        # ---------------------------------------------------------------
        # 2. Tables — extracted separately from layout blocks
        # ---------------------------------------------------------------
        for table in page.tables:
            try:
                table_text = table.to_markdown()
            except Exception:
                table_text = table.get_text()

            bbox_dict = _bbox_to_dict(table.bbox)

            chunk = DocumentChunk(
                chunk_id    = f"p{page_num}_{chunk_index}",
                modality    = "table",
                chunk_text  = table_text,
                page_number = page_num,
                bbox        = bbox_dict,
                layout_type = "TABLE",
                extra_metadata={
                    "source":   s3_path,
                    "num_rows": table.row_count,
                    "num_cols": table.column_count,
                },
            )
            chunks.append(chunk)
            chunk_index += 1

    logger.info("Total chunks extracted: %d", len(chunks))
    return chunks
