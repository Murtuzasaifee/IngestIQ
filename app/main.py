"""
main.py
-------
Entry point for the document RAG pipeline. All configuration is read from .env.

Usage (from the app/ directory):

    # Ingest the PDF specified by INGEST_PDF_PATH in .env:
    uv run python main.py ingest

    # Launch an interactive query session:
    uv run python main.py query
"""

import argparse
import logging
import os
import sys
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

from parsers import get_parser
from chunker import chunk_document
from enrichment import enrich_chunks
from vector_store import get_qdrant_client, ensure_collection, upsert_chunks
from rag_query import answer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load all configuration from .env.

    Core vars are validated eagerly. Backend-specific vars (S3_BUCKET, AWS_REGION,
    etc.) are validated lazily inside get_parser() so non-Textract users don't
    need AWS credentials at all.
    """
    load_dotenv()

    core_required = [
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_CHAT_MODEL",
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "VECTOR_SIZE",
        "MAX_CHUNK_TOKENS",
    ]

    cfg: dict = {k: os.getenv(k) for k in core_required}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)

    cfg["VECTOR_SIZE"]      = int(cfg["VECTOR_SIZE"])
    cfg["MAX_CHUNK_TOKENS"] = int(cfg["MAX_CHUNK_TOKENS"])
    cfg["TOP_K"]            = int(os.getenv("TOP_K") or 5)

    # Backend-specific and optional vars — passed through as-is
    for key in [
        "DOCUMENT_PARSER",
        "S3_BUCKET",
        "AWS_REGION",
        "TEXTRACT_S3_PREFIX",
        "AZURE_DI_ENDPOINT",
        "AZURE_DI_KEY",
        "INGEST_PDF_PATH",
        "CROPS_DIR",
    ]:
        cfg[key] = os.getenv(key, "")

    return cfg


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_ingest(cfg: dict) -> None:
    """Run the full ingestion pipeline for the PDF at INGEST_PDF_PATH."""
    pdf_path = cfg.get("INGEST_PDF_PATH", "").strip()
    if not pdf_path:
        logger.error("INGEST_PDF_PATH is not set in .env")
        sys.exit(1)

    # 1. Parse (backend owned by DOCUMENT_PARSER env var)
    parser       = get_parser(cfg)
    parse_result = parser.parse(pdf_path)
    logger.info(
        "Parsed %d pages, %d total elements.",
        len(parse_result.pages),
        parse_result.total_elements,
    )

    # 2. Document-aware chunking
    chunks = chunk_document(parse_result, max_chunk_tokens=cfg["MAX_CHUNK_TOKENS"])

    # 3. Enrich chunks (GPT vision captions for table/image + word/char counts)
    openai_client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
    chunks = enrich_chunks(
        chunks=chunks,
        openai_client=openai_client,
        chat_model=cfg["OPENAI_CHAT_MODEL"],
    )

    # 4. Embed and index into Qdrant
    qdrant = get_qdrant_client(cfg["QDRANT_URL"])
    ensure_collection(qdrant, cfg["QDRANT_COLLECTION"], cfg["VECTOR_SIZE"])
    upsert_chunks(
        qdrant=qdrant,
        collection_name=cfg["QDRANT_COLLECTION"],
        chunks=chunks,
        openai_client=openai_client,
        embedding_model=cfg["OPENAI_EMBEDDING_MODEL"],
        crops_dir=cfg["CROPS_DIR"] or None,
    )

    modality_counts = Counter(c.modality for c in chunks)
    logger.info("Ingestion complete. %d chunks indexed. Breakdown: %s", len(chunks), dict(modality_counts))


def cmd_query(cfg: dict, question: str) -> None:
    """Run a single RAG query against the indexed collection."""
    openai_client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
    qdrant        = get_qdrant_client(cfg["QDRANT_URL"])

    response = answer(
        query=question,
        qdrant=qdrant,
        collection_name=cfg["QDRANT_COLLECTION"],
        openai_client=openai_client,
        embedding_model=cfg["OPENAI_EMBEDDING_MODEL"],
        chat_model=cfg["OPENAI_CHAT_MODEL"],
        top_k=cfg["TOP_K"],
    )

    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    print(f"A: {response}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Document RAG pipeline")
    sub    = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Ingest the PDF at INGEST_PDF_PATH in .env")

    query_p = sub.add_parser("query", help="Ask a question about indexed documents")
    query_p.add_argument("--question", required=True, help="Natural-language question")

    args = parser.parse_args()
    cfg  = load_config()

    if args.command == "ingest":
        cmd_ingest(cfg)
    elif args.command == "query":
        cmd_query(cfg, args.question)


if __name__ == "__main__":
    main()
