# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Commands

### First-time setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter the project
git clone <repo-url> && cd aws-textract-rag

# 3. Install all dependencies (creates .venv automatically using Python 3.13)
uv sync

# 4. Copy and populate environment variables
cp .env.example .env   # then edit .env with your credentials
```

### Running the app

All `uv run` commands must be executed from the `app/` directory — modules use flat (non-package) imports relative to that directory. All configuration is in `.env` — no CLI arguments needed.

```bash
cd app

# Ingest the PDF at INGEST_PDF_PATH in .env
uv run python main.py ingest

# Interactive query session (type 'exit' or Ctrl+C to quit)
uv run python main.py query
```

### Validate imports

```bash
cd app
uv run python -c "from parsers.base import ParsedElement, ParseResult; from parsers import get_parser; from chunker import chunk_document; from enrichment import enrich_chunks; from vector_store import get_qdrant_client; from rag_query import answer; print('All imports OK')"
```

### Dependency management

```bash
# Add a new package
uv add <package-name>

# Remove a package
uv remove <package-name>

# Update all packages
uv sync --upgrade

# Show installed packages
uv pip list
```

## Environment

Key `.env` variables:

```
# Parser backend — controls which backend get_parser() returns
DOCUMENT_PARSER=textract          # textract | docling (default: textract)

# Ingest / query
INGEST_PDF_PATH=/path/to/doc.pdf
TOP_K=5

# OpenAI (always required)
OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL

# Qdrant (always required)
QDRANT_URL, QDRANT_COLLECTION, VECTOR_SIZE, MAX_CHUNK_TOKENS

# AWS / Textract (only when DOCUMENT_PARSER=textract)
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET
TEXTRACT_S3_PREFIX=aws-textract-input   # S3 key prefix; filename appended automatically
```

`VECTOR_SIZE` must match the embedding model dimension (1536 for `text-embedding-3-small`).
`MAX_CHUNK_TOKENS` controls the maximum estimated tokens per text chunk (default: 512).

## Architecture

```
PDF → S3 → Textract (async, LAYOUT + TABLES features)
              │
              ├── LAYOUT_* blocks  → text chunks
              ├── TABLE blocks     → markdown table chunks
              └── LAYOUT_FIGURE   → image chunks (bbox crop as base64 JPEG)
                                                │
                                         GPT-4o-mini vision → caption
                                                │
                                    OpenAI Embeddings (text / caption)
                                                │
                                          Qdrant (local on-disk)
                                                │
                              Query → embed → search → GPT-4o-mini → Answer
```

### Module responsibilities

| Module | Role |
|---|---|
| `parsers/base.py` | Shared data contract: `ParsedElement`, `PageResult`, `ParseResult` dataclasses + `BaseDocumentParser` ABC + shared utilities (`bbox_dict`, `crop_base64`, `assemble_markdown`) |
| `parsers/textract_parser.py` | `TextractParser`: uploads local PDF to S3, calls Textract async API (LAYOUT + TABLES), rasterizes pages locally with PyMuPDF, returns `ParseResult` |
| `parsers/docling_parser.py` | `DoclingParser`: local Docling-based parsing, no AWS required — maps Docling block types to the same `ParseResult` contract |
| `parsers/__init__.py` | `get_parser(cfg)` factory — reads `DOCUMENT_PARSER` env var, validates backend-specific config, returns the correct parser instance |
| `chunker.py` | Document-aware chunker; imports data types from `parsers.base`; produces `Chunk` list |
| `enrichment.py` | GPT-4o-mini vision captions for table + image chunks; `word_count`/`char_count` on all chunks |
| `vector_store.py` | Qdrant collection management; embeds via OpenAI; upserts/searches `PointStruct` payloads |
| `rag_query.py` | Retrieves top-k chunks, formats context block with modality labels, calls GPT for answer generation |
| `main.py` | CLI entry point — reads all config from `.env`, routes `ingest` / `query`, no CLI arguments |

### Key data model

`ParsedElement` / `PageResult` / `ParseResult` (defined in `parsers/base.py`) are the shared contract between any parser backend and the rest of the pipeline. `Chunk` (defined in `chunker.py`) is the transfer object for enrichment and vector store. `chunk.metadata` dict carries `source`, `word_count`, `char_count` and is flattened into the Qdrant payload on upsert.

### Textract integration notes

- Multi-page PDFs **must** be on S3 before calling Textract — `TextractParser.parse()` handles the upload internally using `TEXTRACT_S3_PREFIX/{filename}` as the S3 key.
- `save_image=False` is mandatory in `start_document_analysis()` — page rasterization is done from the **local PDF** using PyMuPDF (`fitz`). **Do not set `save_image=True`** — it delegates to `pdf2image` which requires the `poppler` system binary and will raise `PDFInfoNotInstalledError`.
- The `block_type` attribute name varies across `amazon-textract-textractor` versions; `textract_parser.py` checks `layout_type` → `block_type` → `raw_object["BlockType"]` defensively.

### Qdrant storage

Qdrant is used in local on-disk mode (`QdrantClient(path=...)`). The collection uses cosine distance. Each point ID is a random UUID; the original `chunk_id` (`p<page>_<index>`) is stored in the payload. Re-ingesting the same document appends new points (no deduplication).


---

## Mistake Log — What Not To Try

Running research log for this project. Every time an approach fails, a wrong assumption is made, or a dead end is hit, append an entry here. Future Claude instances must read this section before starting work to avoid repeating the same mistakes.

**Format:** `[YYYY-MM-DD] [area] — what was tried → why it failed → what to do instead.`

When a mistake is identified — caught by Murtuza or self-detected — append it here immediately. Do not wait until end of session.

<!-- entries below -->
