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

All `uv run` commands must be executed from the `app/` directory — modules use flat (non-package) imports relative to that directory.

```bash
cd app

# Ingest a PDF (uploads to S3 then indexes via Textract)
uv run python main.py ingest --pdf /path/to/report.pdf --s3-key docs/report.pdf

# Query indexed documents
uv run python main.py query --question "What is the net profit margin?"
uv run python main.py query --question "Summarise tables on page 3" --top-k 8
```

### Validate imports

```bash
cd app
uv run python -c "from document_processor import DocumentChunk; from enrichment import enrich_chunks; from vector_store import get_qdrant_client; from rag_query import answer; print('All imports OK')"
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

Copy `.env` (or create from the fields below) before running:

```
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET
OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL
QDRANT_URL, QDRANT_COLLECTION, VECTOR_SIZE
```

`VECTOR_SIZE` must match the embedding model dimension (1536 for `text-embedding-3-small`).

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
| `document_processor.py` | Calls Textract async API via `amazon-textract-textractor`; produces `DocumentChunk` dataclass instances with modality (`text`/`table`/`image`), bbox, and page number |
| `enrichment.py` | Adds GPT-4o-mini vision captions for image chunks; computes `word_count`/`char_count` on all chunks |
| `vector_store.py` | Manages Qdrant client and collection; embeds text via OpenAI; upserts/searches `PointStruct` payloads |
| `rag_query.py` | Retrieves top-k chunks, formats context block with modality labels, calls GPT for answer generation |
| `main.py` | CLI entry point — validates env, routes `ingest` / `query` subcommands |

### Key data model

`DocumentChunk` (defined in `document_processor.py`) is the central transfer object through the pipeline. `extra_metadata` dict carries fields that vary by modality (`source`, `confidence`, `num_rows`/`num_cols` for tables, `word_count`/`char_count` after enrichment) and is flattened into the Qdrant payload on upsert.

### Textract integration notes

- Multi-page PDFs **must** be on S3 before calling Textract (single-page can be local, but this pipeline always uploads first).
- `save_image=False` is set on `start_document_analysis()` — page rasterization is handled separately by `_rasterize_pdf_from_s3()` using `fitz` (PyMuPDF). **Do not set `save_image=True`** — it delegates to `pdf2image` which requires the `poppler` system binary and will raise `PDFInfoNotInstalledError`.
- The `block_type` attribute name varies across `amazon-textract-textractor` versions; `document_processor.py` checks `layout_type` → `block_type` → `raw_object["BlockType"]` defensively.

### Qdrant storage

Qdrant is used in local on-disk mode (`QdrantClient(path=...)`). The collection uses cosine distance. Each point ID is a random UUID; the original `chunk_id` (`p<page>_<index>`) is stored in the payload. Re-ingesting the same document appends new points (no deduplication).


---

## Mistake Log — What Not To Try

Running research log for this project. Every time an approach fails, a wrong assumption is made, or a dead end is hit, append an entry here. Future Claude instances must read this section before starting work to avoid repeating the same mistakes.

**Format:** `[YYYY-MM-DD] [area] — what was tried → why it failed → what to do instead.`

When a mistake is identified — caught by Murtuza or self-detected — append it here immediately. Do not wait until end of session.

<!-- entries below -->

[2026-04-08] [document_processor / rasterization] — Used `save_image=True` in `extractor.start_document_analysis()` → `PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?`. Textractor delegates rasterization to `pdf2image` which requires the `poppler` system binary. → Set `save_image=False` and rasterize pages ourselves in `_rasterize_pdf_from_s3()` using `fitz` (PyMuPDF, already a project dependency). Download PDF bytes from S3 via boto3, open with `fitz.open(stream=..., filetype="pdf")`, render each page with `get_pixmap()`, convert to `PIL.Image`. Result stored in a `Dict[int, Image.Image]` keyed by 1-based page number.

[2026-04-08] [document_processor / Table API] — Accessed `table.rows` and `table.rows[0].cells` to get row/col counts → `AttributeError: 'Table' object has no attribute 'rows'`. The Textractor `Table` entity exposes `row_count` and `column_count` integer properties directly. → Use `table.row_count` and `table.column_count`.

[2026-04-08] [vector_store / Qdrant API] — Called `qdrant.search(collection_name=..., query_vector=..., limit=..., with_payload=True)` → `AttributeError: 'QdrantClient' object has no attribute 'search'`. The `search` method was removed in qdrant-client v1.x. → Use `qdrant.query_points(collection_name=..., query=<vector>, limit=..., with_payload=True)` which returns a `QueryResponse`; iterate over `response.points` (list of `ScoredPoint`) instead of the result directly.

[2026-04-08] [document_processor / Textractor] — Passed `aws_access_key_id` and `aws_secret_access_key` as kwargs to `Textractor()` → `TypeError: Textractor.__init__() got an unexpected keyword argument 'aws_access_key_id'`. `Textractor` only accepts `profile_name`, `region_name`, `kms_key_id`; it builds its own `boto3.session.Session` internally. → Do not pass explicit credentials to `Textractor`. Rely on env vars (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) which `load_dotenv()` already sets before this code runs — boto3 picks them up automatically.