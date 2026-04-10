# IngestIQ

Intelligent document ingestion and Retrieval-Augmented Generation pipeline using:
- **Amazon Textract** or **Azure Document Intelligence** for layout-aware document parsing
- **Qdrant** (local on-disk) as the vector store
- **OpenAI** for embeddings, image captioning, and answer generation

---

## Architecture

```
PDF → S3 → Textract (LAYOUT + TABLES)
              │
              ├── LAYOUT_* blocks → text chunks
              ├── TABLE blocks    → table chunks (markdown)
              └── LAYOUT_FIGURE  → image chunks (bbox crop as base64 JPEG)
                                          │
                              ┌───────────┴────────────┐
                         table chunks             image chunks
                      GPT-4o-mini vision       GPT-4o-mini vision
                           caption                  caption
                              └───────────┬────────────┘
                                          │
                              document-aware chunker
                                          │
                                OpenAI Embeddings
                                (caption for table/image,
                                 raw text for text chunks)
                                          │
                                    Qdrant (local)
                                          │
                            Query → Retrieve → GPT-4o-mini → Answer
```

### Modalities extracted

| Textract block type | Modality in metadata |
|---------------------|----------------------|
| LAYOUT_TEXT, LAYOUT_TITLE, LAYOUT_SECTION_HEADER, etc. | `text` |
| TABLE | `table` |
| LAYOUT_FIGURE | `image` |

---

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management (Python 3.13).

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install all dependencies (auto-creates .venv with Python 3.13)
uv sync

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your credentials (see required variables below)
```

### Required `.env` variables

```
# Parser backend (textract | azure — default: textract)
DOCUMENT_PARSER=textract

# PDF to ingest
INGEST_PDF_PATH=/path/to/your/document.pdf

# Query
TOP_K=5

# OpenAI (always required)
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

# Qdrant (always required)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                         # leave empty for local Qdrant
QDRANT_COLLECTION=ingest_iq
VECTOR_SIZE=1536                        # must match embedding model dimension
MAX_CHUNK_TOKENS=512

# AWS / Textract (only required when DOCUMENT_PARSER=textract)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
TEXTRACT_S3_PREFIX=aws-textract-input

# CloudWatch logging (only used when DOCUMENT_PARSER=textract)
CW_LOG_GROUP=/ingestiq/textract        # default if not set
CW_LOG_STREAM=textract-parser          # default if not set

# Azure Document Intelligence (only required when DOCUMENT_PARSER=azure)
AZURE_DI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DI_KEY=...
```

### AWS requirements (Textract backend only)
- IAM user/role with `textract:*` and `s3:*` permissions
- S3 bucket in the same region as Textract (multi-page PDFs must be on S3)

---

## Usage

All configuration lives in `.env` — no command-line arguments needed.

### Ingest a PDF

Set `INGEST_PDF_PATH` in `.env` to the local PDF you want to process, then:

```bash
cd app
uv run python main.py ingest
```

Uploads the PDF to S3 (Textract backend), runs async document analysis, enriches table and image chunks with GPT-4o-mini vision captions, and indexes everything into the local Qdrant store.

### Query

```bash
cd app
uv run python main.py query
```

Launches an interactive session — type your question at the prompt, `exit` or `Ctrl+C` to quit. Retrieval depth is controlled by `TOP_K` in `.env` (default: 5).

### Switching parser backends

Change one line in `.env`:

```
DOCUMENT_PARSER=azure   # local, no AWS required — install: uv add azure-ai-documentintelligence
DOCUMENT_PARSER=textract  # default, requires S3_BUCKET + AWS credentials
```

---

## Chunk metadata payload (stored in Qdrant)

| Field | Description |
|-------|-------------|
| `chunk_id` | `p<page>_<index>` |
| `modality` | `text` / `table` / `image` |
| `chunk_text` | GPT caption for table/image chunks; raw text for text chunks — matches the embedded vector |
| `caption` | GPT-4o-mini vision caption (table and image chunks only) |
| `page_number` | 1-based page index |
| `elements` | Textract block labels that make up the chunk (e.g. `["table"]`, `["title", "text"]`) |
| `bbox` | Normalised bounding box `{left, top, width, height}` (atomic chunks only) |
| `image_base64` | JPEG crop as base64 (table and image chunks) |
| `source` | `s3://bucket/key` |
| `word_count` | Word count of `chunk_text` |
| `char_count` | Character count of `chunk_text` |

---

## File structure

```
app/
├── main.py                        # CLI entry point (ingest / query — config from .env)
├── parsers/
│   ├── __init__.py                # get_parser(cfg) factory — reads DOCUMENT_PARSER env var
│   ├── base.py                    # ParsedElement, PageResult, ParseResult, BaseDocumentParser
│   ├── textract_parser.py         # TextractParser: S3 upload + Textract async analysis
│   └── azure_di_parser.py          # AzureDocumentIntelligenceParser: Azure Document Intelligence-based parsing, uses Azure Document Intelligence API — maps Azure Document Intelligence block types to the same `ParseResult` contract
├── chunker.py                     # Document-aware chunker → Chunk list
├── enrichment.py                  # GPT vision captions (table + image) & word/char counts
├── vector_store.py                # Qdrant collection management, embedding, upsert & search
└── rag_query.py                   # Retrieval + GPT-4o-mini answer generation
```