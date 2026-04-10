"""
parsers/textract_parser.py
--------------------------
AWS Textract backend for document parsing.

Uploads the local PDF to S3 (using TEXTRACT_S3_PREFIX / filename as the key),
runs Textract async analysis with LAYOUT + TABLES features, rasterizes pages
locally with PyMuPDF, and returns a ParseResult.

Credentials are picked up from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env
vars, which load_dotenv() in main.py sets before this code runs.

"""

import logging
import os
from pathlib import Path
from typing import Dict, List

import boto3
import fitz  # PyMuPDF
import watchtower
from PIL import Image
from textractor import Textractor
from textractor.data.constants import TextractFeatures

from parsers.base import (
    BaseDocumentParser,
    ParsedElement,
    PageResult,
    ParseResult,
    assemble_markdown,
    bbox_dict,
    crop_base64,
)

logger = logging.getLogger(__name__)

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


def _get_block_type(block) -> str:
    """Defensively resolve Textract block type across textractor versions."""
    return (
        getattr(block, "layout_type", None)
        or getattr(block, "block_type", None)
        or (block.raw_object.get("BlockType", "LAYOUT_TEXT")
            if hasattr(block, "raw_object") else "LAYOUT_TEXT")
    )


class TextractParser(BaseDocumentParser):
    """Document parser backed by Amazon Textract.

    S3 upload is an internal implementation detail — callers just pass a
    local PDF path to parse(). The S3 key is derived as:
        {s3_prefix}/{filename}   (or just {filename} when prefix is empty)
    """

    def __init__(self, s3_bucket: str, aws_region: str, s3_prefix: str = "") -> None:
        self._s3_bucket  = s3_bucket
        self._aws_region = aws_region
        self._s3_prefix  = s3_prefix.rstrip("/")
        self._setup_cloudwatch_logging()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _setup_cloudwatch_logging(self) -> None:
        """Attach a CloudWatch Logs handler to this module's logger.

        Reads CW_LOG_GROUP and CW_LOG_STREAM from the environment (both
        optional — defaults shown below). Guards against duplicate handlers
        if the parser is instantiated more than once in the same process.
        """
        log_group  = os.getenv("CW_LOG_GROUP",  "/aws-textract-rag/textract")
        log_stream = os.getenv("CW_LOG_STREAM", "textract-parser")

        # Skip if a CloudWatch handler is already registered
        if any(isinstance(h, watchtower.CloudWatchLogHandler) for h in logger.handlers):
            return

        logs_client = boto3.client("logs", region_name=self._aws_region)
        handler = watchtower.CloudWatchLogHandler(
            log_group_name=log_group,
            log_stream_name=log_stream,
            boto3_client=logs_client,
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.info("CloudWatch logging enabled → %s / %s", log_group, log_stream)

    def parse(self, pdf_path: str) -> ParseResult:
        filename = Path(pdf_path).name
        s3_key = f"{self._s3_prefix}/{filename}" if self._s3_prefix else filename
        s3_uri  = f"s3://{self._s3_bucket}/{s3_key}"

        self._upload(pdf_path, s3_key)

        logger.info("Starting Textract analysis: %s", s3_uri)
        extractor = Textractor(region_name=self._aws_region)
        document  = extractor.start_document_analysis(
            file_source=s3_uri,
            features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES],
            save_image=False,  # PyMuPDF rasterizes pages — do NOT set True (needs poppler)
        )
        logger.info("Textract complete. Pages: %d", document.num_pages)

        # Rasterize from the local file — no need to re-download from S3
        page_images = self._rasterize(pdf_path)
        page_results: List[PageResult] = []

        for page in document.pages:
            page_num   = page.page_num
            page_image = page_images.get(page_num)
            raw_elements: List[ParsedElement] = []

            # --- Layout blocks (text, titles, figures; LAYOUT_TABLE skipped) ---
            for layout in page.layouts:
                bt = _get_block_type(layout)
                if bt == "LAYOUT_TABLE":
                    continue  # TABLE objects below provide structured markdown

                label = _BLOCK_TO_LABEL.get(bt, "text")
                text  = layout.get_text().strip()
                bbox  = bbox_dict(layout.bbox)
                image_b64 = None

                if label == "figure":
                    image_b64 = crop_base64(page_image, bbox)
                    text = ""  # OCR inside figure region is noise; GPT vision handles captioning
                elif not text:
                    continue

                raw_elements.append(ParsedElement(
                    label=label,
                    text=text,
                    bbox=bbox,
                    score=round(getattr(layout, "confidence", 1.0) or 1.0, 4),
                    reading_order=0,
                    page_number=page_num,
                    image_base64=image_b64,
                ))

            # --- Table blocks (structured markdown rendering + image crop) ---
            for table in page.tables:
                try:
                    text = table.to_markdown()
                except Exception:
                    text = table.get_text()

                table_bbox = bbox_dict(table.bbox)
                raw_elements.append(ParsedElement(
                    label="table",
                    text=text,
                    bbox=table_bbox,
                    score=round(getattr(table, "confidence", 1.0) or 1.0, 4),
                    reading_order=0,
                    page_number=page_num,
                    image_base64=crop_base64(page_image, table_bbox),
                ))

            # Sort by vertical then horizontal position to restore reading order
            raw_elements.sort(key=lambda e: (
                e.bbox["top"]  if e.bbox else 0.0,
                e.bbox["left"] if e.bbox else 0.0,
            ))
            for i, el in enumerate(raw_elements):
                el.reading_order = i

            page_results.append(PageResult(
                page_number=page_num,
                elements=raw_elements,
                markdown=assemble_markdown(raw_elements),
            ))
            logger.info("Page %d: %d elements", page_num, len(raw_elements))

        total = sum(len(p.elements) for p in page_results)
        logger.info("Parse complete. Total elements: %d", total)
        return ParseResult(source_file=s3_uri, pages=page_results, total_elements=total)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upload(self, pdf_path: str, s3_key: str) -> None:
        logger.info("Uploading '%s' → s3://%s/%s", pdf_path, self._s3_bucket, s3_key)
        boto3.client("s3", region_name=self._aws_region).upload_file(
            pdf_path, self._s3_bucket, s3_key
        )
        logger.info("Upload complete.")

    def _rasterize(self, pdf_path: str, dpi: int = 250) -> Dict[int, Image.Image]:
        """Rasterize every page of a local PDF with PyMuPDF (no poppler required)."""
        scale  = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        images: Dict[int, Image.Image] = {}
        with fitz.open(pdf_path) as pdf:
            for i in range(len(pdf)):
                pix = pdf[i].get_pixmap(matrix=matrix)
                images[i + 1] = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        logger.info("Rasterized %d pages at %d DPI", len(images), dpi)
        return images
