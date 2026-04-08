"""
main.py
-------
Entry point for the Naive RAG pipeline.

Usage
-----
# Ingest a PDF (uploads to S3 then processes with Textract):
    python main.py ingest --pdf path/to/document.pdf --s3-key docs/document.pdf

# Query the indexed document:
    python main.py query --question "What is the total revenue for Q3?"

Environment
-----------
All configuration is read from a .env file (see .env.example).
"""

import argparse
import logging
import os
import sys

import boto3
from dotenv import load_dotenv
from openai import OpenAI

from document_processor import extract_chunks_from_s3
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
# Config loader
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load all required variables from .env into a dict."""
    load_dotenv()

    required = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET",
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_CHAT_MODEL",
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "VECTOR_SIZE",
    ]

    config = {}
    missing = []
    for key in required:
        value = os.getenv(key)
        if not value:
            missing.append(key)
        config[key] = value

    if missing:
        logger.error("Missing environment variables: %s", ", ".join(missing))
        sys.exit(1)

    config["VECTOR_SIZE"] = int(config["VECTOR_SIZE"])
    return config


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_ingest(args, cfg: dict) -> None:
    """Upload PDF to S3 and run the full ingestion pipeline."""

    # 1. Upload local PDF to S3
    logger.info("Uploading '%s' to s3://%s/%s …", args.pdf, cfg["S3_BUCKET"], args.s3_key)
    s3 = boto3.client(
        "s3",
        region_name=cfg["AWS_REGION"],
        aws_access_key_id=cfg["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=cfg["AWS_SECRET_ACCESS_KEY"],
    )
    s3.upload_file(args.pdf, cfg["S3_BUCKET"], args.s3_key)
    logger.info("Upload complete.")

    # 2. Extract chunks via Textract
    chunks = extract_chunks_from_s3(
        s3_bucket=cfg["S3_BUCKET"],
        s3_key=args.s3_key,
        aws_region=cfg["AWS_REGION"],
        aws_access_key_id=cfg["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=cfg["AWS_SECRET_ACCESS_KEY"],
    )

    # 3. Enrich chunks (captions, metadata)
    openai_client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
    chunks = enrich_chunks(
        chunks=chunks,
        openai_client=openai_client,
        chat_model=cfg["OPENAI_CHAT_MODEL"],
    )

    # 4. Upsert into Qdrant
    qdrant = get_qdrant_client(cfg["QDRANT_URL"])
    ensure_collection(qdrant, cfg["QDRANT_COLLECTION"], cfg["VECTOR_SIZE"])
    upsert_chunks(
        qdrant=qdrant,
        collection_name=cfg["QDRANT_COLLECTION"],
        chunks=chunks,
        openai_client=openai_client,
        embedding_model=cfg["OPENAI_EMBEDDING_MODEL"],
    )

    logger.info("Ingestion pipeline complete. %d chunks indexed.", len(chunks))

    # Print a short summary
    modality_counts = {}
    for c in chunks:
        modality_counts[c.modality] = modality_counts.get(c.modality, 0) + 1
    logger.info("Modality breakdown: %s", modality_counts)


def cmd_query(args, cfg: dict) -> None:
    """Run a single RAG query against the indexed collection."""
    openai_client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
    qdrant        = get_qdrant_client(cfg["QDRANT_URL"])

    response = answer(
        query=args.question,
        qdrant=qdrant,
        collection_name=cfg["QDRANT_COLLECTION"],
        openai_client=openai_client,
        embedding_model=cfg["OPENAI_EMBEDDING_MODEL"],
        chat_model=cfg["OPENAI_CHAT_MODEL"],
        top_k=args.top_k,
    )

    print("\n" + "=" * 60)
    print(f"Q: {args.question}")
    print("=" * 60)
    print(f"A: {response}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Naive RAG with Textract + Qdrant")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = sub.add_parser("ingest", help="Process and index a PDF document")
    ingest_p.add_argument("--pdf",    required=True, help="Local path to the PDF file")
    ingest_p.add_argument("--s3-key", required=True, help="S3 object key (e.g. docs/file.pdf)")

    # query
    query_p = sub.add_parser("query", help="Ask a question about indexed documents")
    query_p.add_argument("--question", required=True, help="Natural-language question")
    query_p.add_argument("--top-k",    type=int, default=5, help="Chunks to retrieve (default: 5)")

    return parser


def main():
    cfg    = load_config()
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args, cfg)
    elif args.command == "query":
        cmd_query(args, cfg)


if __name__ == "__main__":
    main()