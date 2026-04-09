# Naive RAG — Textract + Qdrant

Minimal Retrieval-Augmented Generation pipeline using:
- **Amazon Textract** (via `amazon-textract-textractor`) for layout-aware document parsing
- **Qdrant** (local on-disk) as the vector store
- **OpenAI** `gpt-5.4-mini` for embeddings, image captioning, and answer generation

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
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name

OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

QDRANT_URL=./qdrant_data
QDRANT_COLLECTION=naive_rag
VECTOR_SIZE=1536
```

### AWS requirements
- IAM user/role with `textract:*` and `s3:*` permissions
- S3 bucket in the same region as Textract (multi-page PDFs must be on S3)

---

## Usage

All commands are run from the `app/` directory using `uv run`.

### Ingest a PDF

```bash
cd app
uv run python app/main.py ingest --pdf data/docling_report-3-5.pdf --s3-key aws-textract-input/doc.pdf
```

This uploads the PDF to S3, runs Textract async analysis, enriches image chunks with GPT captions, and indexes everything into the local Qdrant store.

### Query

```bash
cd app
uv run python app/main.py query --question "Intel(R) Xeon E5-2690"
uv run python app/main.py query --question "Summarise the tables on page 3." --top-k 8
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
├── main.py               # CLI entry point (ingest / query subcommands)
├── document_processor.py # Textract extraction → ParsedElement / ParseResult
├── chunker.py            # Document-aware chunker → Chunk list
├── enrichment.py         # GPT vision captions (table + image) & word/char counts
├── vector_store.py       # Qdrant collection management, embedding, upsert & search
├── rag_query.py          # Retrieval + GPT-5.4-mini answer generation
└── .env.example
```