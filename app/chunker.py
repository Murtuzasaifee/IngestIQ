"""
chunker.py
----------
Document-aware chunker that processes a full ParseResult in a single pass
across ALL pages, preserving cross-page context.

Chunk modalities:
  "text"  — one or more text/title/list/key_value elements merged up to max_chunk_tokens
  "table" — always a single atomic chunk (one TABLE element)
  "image" — always a single atomic chunk (one LAYOUT_FIGURE element)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from parsers.base import ParsedElement, ParseResult

logger = logging.getLogger(__name__)

# Elements that always become their own standalone chunk — never merged
_ATOMIC_LABELS = {"table", "figure"}

# Elements that act as forward-attaching headings
_TITLE_LABELS = {"title", "section_title"}

# Elements skipped entirely (no retrieval value)
_SKIP_LABELS = {"page_header", "page_footer", "page_number"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str                          # "p<first_page>_<index>"
    text: str                              # chunk content (Markdown for tables)
    modality: str                          # "text" | "table" | "image"
    page: int                              # page of the first element in the chunk
    elements: List[str]                    # labels of constituent elements
    bbox: Optional[dict] = None            # bbox of the primary element (atomic chunks only)
    image_base64: Optional[str] = None     # JPEG crop as base64 (image chunks only)
    caption: Optional[str] = None          # GPT caption — filled in by enrichment step
    metadata: dict = field(default_factory=dict)   # source, word_count, char_count, …


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words / 0.75 (no tokenizer dependency)."""
    return int(len(text.split()) / 0.75)


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_document(
    parse_result: ParseResult,
    max_chunk_tokens: int = 512,
) -> List[Chunk]:
    """Chunk a whole document across ALL pages in a single pass.

    Unlike single-page chunking, this processes every page as one continuous
    element stream so that:

    - A section heading on the last line of page N attaches to content on page N+1
      instead of becoming an orphan chunk.
    - A ``figure_title`` (figure caption label) is linked directly to the
      following ``figure`` atomic chunk rather than floating into a surrounding
      text chunk.

    Rules (applied in order per element):
    - Skip decorative elements (page_header, page_footer, page_number).
    - ``figure_title`` is intercepted and held as a pending caption for the
      next figure chunk.
    - Atomic elements (table, figure) always get their own chunk; the pending
      caption is prepended when present.
    - Title elements (title, section_title) flush the current text chunk and
      are carried forward to attach to the next content element.
    - Text elements accumulate up to ``max_chunk_tokens``; overflow starts a
      new chunk.
    - Chunk ``page`` is set to the page of the first element that entered it.

    Args:
        parse_result: Output of parse_document_from_s3.
        max_chunk_tokens: Maximum estimated tokens per text chunk (default 512).

    Returns:
        List of Chunk objects in document reading order.
    """
    # Flatten all elements across all pages into one ordered stream
    all_elements: List[ParsedElement] = []
    for page_result in parse_result.pages:
        all_elements.extend(page_result.elements)

    chunks: List[Chunk] = []
    chunk_index = 0

    # --- Text accumulation state ---
    pending_texts: List[str] = []
    pending_labels: List[str] = []
    pending_page: Optional[int] = None
    pending_tokens: int = 0

    # --- Forward-carry state ---
    pending_title: Optional[str] = None    # title text waiting to attach to next text chunk
    pending_caption: Optional[str] = None  # figure_title text waiting for next figure

    def flush_text() -> None:
        """Emit the accumulated text chunk and reset accumulators."""
        nonlocal pending_texts, pending_labels, pending_page, pending_tokens, chunk_index
        if not pending_texts:
            return
        body = "\n\n".join(t.strip() for t in pending_texts if t.strip())
        chunks.append(Chunk(
            chunk_id=f"p{pending_page}_{chunk_index}",
            text=body,
            modality="text",
            page=pending_page,
            elements=list(pending_labels),
            metadata={"source": parse_result.source_file},
        ))
        chunk_index += 1
        pending_texts.clear()
        pending_labels.clear()
        pending_page = None  # type: ignore[assignment]
        pending_tokens = 0

    for el in all_elements:

        # 1. Skip decorative elements
        if el.label in _SKIP_LABELS:
            continue

        # 2. Intercept figure_title — hold for the next figure chunk
        if el.label == "figure_title":
            pending_caption = el.text.strip()
            continue

        # 3. Atomic elements — flush text, emit standalone chunk
        if el.label in _ATOMIC_LABELS:
            flush_text()
            caption = pending_caption
            pending_caption = None
            pending_title = None  # a heading before a figure is consumed here

            body = f"{caption}\n\n{el.text}".strip() if caption else el.text
            modality = "image" if el.label == "figure" else el.label
            el_labels = ["figure_title", el.label] if caption else [el.label]

            chunks.append(Chunk(
                chunk_id=f"p{el.page_number}_{chunk_index}",
                text=body,
                modality=modality,
                page=el.page_number,
                elements=el_labels,
                bbox=el.bbox,
                image_base64=el.image_base64,
                metadata={"source": parse_result.source_file},
            ))
            chunk_index += 1
            continue

        # 4. Title elements — flush text, carry forward to next content element
        if el.label in _TITLE_LABELS:
            flush_text()
            # Consecutive headings: emit previous orphan title rather than silently
            # overwriting it. Without this, all but the last heading in a sequence
            # of headings (e.g. table-of-contents, multi-level section headers)
            # would be lost.
            if pending_title:
                chunks.append(Chunk(
                    chunk_id=f"p{el.page_number}_{chunk_index}",
                    text=pending_title,
                    modality="text",
                    page=el.page_number,
                    elements=["title"],
                    metadata={"source": parse_result.source_file},
                ))
                chunk_index += 1
            pending_title = el.text.strip()
            continue

        # 5. Text / list / key_value — accumulate up to max_chunk_tokens
        text = el.text.strip()
        if not text:
            continue

        # Prepend any pending title so heading and first paragraph are co-located
        if pending_title:
            text = f"{pending_title}\n\n{text}"
            pending_title = None

        tok = _estimate_tokens(text)

        if pending_page is None:
            # Start a fresh accumulation
            pending_page = el.page_number
            pending_texts.append(text)
            pending_labels.append(el.label)
            pending_tokens = tok
        elif pending_tokens + tok > max_chunk_tokens:
            # Token budget exceeded — flush and start a new chunk
            flush_text()
            pending_page = el.page_number
            pending_texts.append(text)
            pending_labels.append(el.label)
            pending_tokens = tok
        else:
            pending_texts.append(text)
            pending_labels.append(el.label)
            pending_tokens += tok

    # Flush any remaining text (or orphan title at end of document)
    if pending_title and not pending_texts:
        pending_texts = [pending_title]
        pending_labels = ["title"]
        pending_page = parse_result.pages[-1].page_number if parse_result.pages else 1
    flush_text()

    logger.info(
        "Chunked %d elements into %d chunks (%d text, %d table, %d image)",
        len(all_elements),
        len(chunks),
        sum(1 for c in chunks if c.modality == "text"),
        sum(1 for c in chunks if c.modality == "table"),
        sum(1 for c in chunks if c.modality == "image"),
    )
    return chunks
