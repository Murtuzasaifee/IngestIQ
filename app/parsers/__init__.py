"""
parsers/__init__.py
-------------------
Factory that returns the configured document parser backend.

Controlled by the DOCUMENT_PARSER environment variable (default: textract).
Adding a new backend requires only:
  1. A new file in parsers/ that subclasses BaseDocumentParser
  2. A new branch in get_parser() below
"""

import logging

from parsers.base import BaseDocumentParser

logger = logging.getLogger(__name__)


def get_parser(cfg: dict) -> BaseDocumentParser:
    """Instantiate and return the parser backend specified by DOCUMENT_PARSER.

    Args:
        cfg: Config dict produced by main.load_config().

    Returns:
        A ready-to-use BaseDocumentParser instance.

    Raises:
        ValueError: If DOCUMENT_PARSER names an unknown backend.
        RuntimeError: If a backend's required dependencies are not installed.
    """
    backend = (cfg.get("DOCUMENT_PARSER") or "textract").lower()
    logger.info("Document parser backend: %s", backend)

    if backend == "textract":
        _require(cfg, ["S3_BUCKET", "AWS_REGION"], backend)
        from parsers.textract_parser import TextractParser
        return TextractParser(
            s3_bucket=cfg["S3_BUCKET"],
            aws_region=cfg["AWS_REGION"],
            s3_prefix=cfg.get("TEXTRACT_S3_PREFIX") or "",
        )

    if backend == "azure":
        _require(cfg, ["AZURE_DI_ENDPOINT", "AZURE_DI_KEY"], backend)
        from parsers.azure_di_parser import AzureDocumentIntelligenceParser
        return AzureDocumentIntelligenceParser(
            endpoint=cfg["AZURE_DI_ENDPOINT"],
            api_key=cfg["AZURE_DI_KEY"],
        )

    raise ValueError(
        f"Unknown DOCUMENT_PARSER: '{backend}'. Valid values: textract, azure"
    )


def _require(cfg: dict, keys: list[str], backend: str) -> None:
    """Exit with a clear message if backend-specific config keys are missing."""
    import sys
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        logger.error(
            "DOCUMENT_PARSER=%s requires these env vars: %s",
            backend, ", ".join(missing),
        )
        sys.exit(1)
