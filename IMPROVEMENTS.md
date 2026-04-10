# IngestIQ — Improvement Tracker

All pipeline improvement suggestions are tracked here with status, priority, and context.

**Status legend:** `[ ]` open · `[~]` in progress · `[x]` done · `[-]` rejected

---

## Batch 1 — Pipeline Analysis (2026-04-10)

### Stage 4 — Embed & Store (`vector_store.py`)

| # | Status | Priority | Title |
|---|--------|----------|-------|
| 1 | [x] | High | Batch embeddings: replace per-chunk `_embed()` calls with a single batched OpenAI request |
| 2 | [x] | Medium | Remove `image_base64` from Qdrant payload — store a file path or bbox reference instead to avoid 5–15 MB of binary bloat per document |
| 3 | [x] | Low | Re-ingest deduplication — delete existing points for the same `source` before upserting to prevent index doubling |

**#1 Detail:** `_embed()` is called once per chunk — O(N) API round trips. OpenAI `embeddings.create` accepts a list of strings. Batch all embed texts in one call (or in batches of ≤2048). Single highest-ROI performance change in the pipeline.

**#2 Detail:** A 250 DPI JPEG crop is typically 50–300 KB per table/figure. Store crops as files under `data/crops/<doc_name>/` and save the relative path in the payload instead of raw base64. Regenerate on demand using `source` + `bbox` + `page_number` if needed.

**#3 Detail:** No guard against double-ingest. Add a pre-upsert delete filtered on `source == s3_uri` using `qdrant.delete(collection_name, points_selector=Filter(...))`.

---

### Stage 3 — Enrich (`enrichment.py`)

| # | Status | Priority | Title |
|---|--------|----------|-------|
| 4 | [ ] | High | Parallelize caption generation with `ThreadPoolExecutor` |
| 5 | [ ] | Low | Fix `enrich_chunks` mutation + return inconsistency |

**#4 Detail:** Caption generation is fully sequential — 15 tables/figures = 15 serial vision API calls (~2–5 s each). Use `ThreadPoolExecutor(max_workers=6)` over the image/table chunks. No downstream logic changes needed.

**#5 Detail:** `enrich_chunks` mutates chunks in-place AND returns the list. Pick one: either remove the return value (in-place mutation is clear) or make it a pure function. Current pattern implies the original is replaced but it isn't.

---

### Stage 1 — Parse (`parsers/`)

| # | Status | Priority | Title |
|---|--------|----------|-------|
| 6 | [x] | Low | Move `_rasterize()` to `base.py` — identical code in both `textract_parser.py` and `azure_di_parser.py` |
| 7 | [x] | Low | Remove dead `PageResult.markdown` field and `assemble_markdown()` calls from both parsers |
| 8 | [x] | Low | Fix `_call_azure_di` passing `DocumentContentFormat` class as a parameter — move the try/import to module level |

**#6 Detail:** `_rasterize()` is verbatim identical in both parsers (same signature, same `fitz.Matrix`, same `Image.frombytes`). Extract to `base.py` as a standalone function.

**#7 Detail:** `assemble_markdown()` is called on every page by both parsers, result stored in `PageResult.markdown`. Nothing downstream reads `page_result.markdown` — chunker, enrichment, vector store, and rag_query all ignore it. Dead code that runs on every ingest.

**#8 Detail:** `DocumentContentFormat` is imported inside `parse()` and threaded as a parameter into `_call_azure_di`. Either hoist the import to a lazy module-level helper or inline the `DocumentContentFormat.MARKDOWN` reference directly.

---

### Stage 2 — Chunk (`chunker.py`)

| # | Status | Priority | Title |
|---|--------|----------|-------|
| 9 | [ ] | Low | Fix orphan title page number — uses current element's page, not the title's actual page |
| 10 | [x] | Low | Deduplicate `_SKIP_LABELS` — defined identically in both `base.py` and `chunker.py` |

**#9 Detail:** When `pending_title` is flushed as a standalone chunk, `el.page_number` (the *next* element) is used instead of the page where the title appeared. Add `pending_title_page: Optional[int]` to track the title's origin page.

**#10 Detail:** `_SKIP_LABELS = {"page_header", "page_footer", "page_number"}` is declared in both `base.py:85` and `chunker.py:28`. The chunker already imports from `parsers.base` — import the set instead of redefining it.

---

### Stage 5 — Query (`rag_query.py`)

| # | Status | Priority | Title |
|---|--------|----------|-------|
| 11 | [ ] | Medium | Add minimum score threshold to filter low-relevance chunks before sending to GPT |
| 12 | [ ] | Low | Replace `print()` in `_print_retrieved_chunks` with `logger.debug()` |
| 13 | [ ] | Low | Remove misleading `[table check] chunk_text == caption` assertion |
| 14 | [ ] | Low | Increase `max_completion_tokens` from 512 to 1024 |

**#11 Detail:** All `top_k` results go to GPT regardless of score. A chunk with score 0.45 and one with 0.92 get equal weight. Add `MIN_SCORE` env var (default 0.5); filter hits before building context.

**#12 Detail:** `_print_retrieved_chunks` uses raw `print()` — cannot be suppressed without modifying the function. Replace with `logger.debug()` so it only appears at `LOG_LEVEL=DEBUG`.

**#13 Detail:** `chunk_text == caption` is always `True` for table chunks because `vector_store.py` sets `chunk_text = embed_text = chunk.caption or chunk.text`. The check is a no-op that misleads readers.

**#14 Detail:** 512 tokens can truncate multi-part answers. 1024 is safer without meaningful cost impact given that context retrieval is already the dominant cost.

---

### Stage 6 — Orchestration (`main.py`)

| # | Status | Priority | Title |
|---|--------|----------|-------|
| 15 | [x] | Low | Move `TOP_K` int-parsing into `load_config()` alongside `VECTOR_SIZE` / `MAX_CHUNK_TOKENS` |

**#15 Detail:** `TOP_K` is parsed with `int(cfg.get("TOP_K") or 5)` at call site. Move it into the `load_config()` int-parsing block for consistency.
