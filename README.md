# Naive RAG — Textract + Qdrant

Minimal Retrieval-Augmented Generation pipeline using:
- **Amazon Textract** (via `amazon-textract-textractor`) for layout-aware document parsing
- **Qdrant** (local on-disk) as the vector store
- **OpenAI** `gpt-4o-mini` for embeddings, image captioning, and answer generation

---

## Architecture

```
PDF → S3 → Textract (LAYOUT + TABLES)
              │
              ├── text   chunks  ──────────────────┐
              ├── table  chunks (markdown)          │  enrich
              └── image  chunks (bbox crop + caption)│
                                                   ▼
                                         OpenAI Embeddings
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
uv run python main.py ingest --pdf /path/to/report.pdf --s3-key docs/report.pdf
```

This uploads the PDF to S3, runs Textract async analysis, enriches image chunks with GPT captions, and indexes everything into the local Qdrant store.

### Query

```bash
cd app
uv run python main.py query --question "What is the net profit margin?"
uv run python main.py query --question "Summarise the tables on page 3." --top-k 8
```

---

## Chunk metadata payload (stored in Qdrant)

| Field | Description |
|-------|-------------|
| `chunk_id` | `p<page>_<index>` |
| `modality` | `text` / `table` / `image` |
| `chunk_text` | Raw extracted text |
| `page_number` | 1-based page index |
| `layout_type` | Textract layout block type |
| `bbox` | Normalised bounding box `{left, top, width, height}` |
| `image_base64` | JPEG crop as base64 (image only) |
| `image_caption` | GPT-generated caption (image only) |
| `source` | `s3://bucket/key` |
| `confidence` | Textract confidence score |
| `word_count` | Word count of chunk text |
| `char_count` | Character count |
| `num_rows` / `num_cols` | Table dimensions (table only) |

---

## File structure

```
naive_rag/
├── main.py               # Entry point & CLI
├── document_processor.py # Textract extraction → DocumentChunk list
├── enrichment.py         # Image captioning & metadata enrichment
├── vector_store.py       # Qdrant upsert & search
├── rag_query.py          # Retrieval + GPT generation
├── requirements.txt
└── .env.example
```