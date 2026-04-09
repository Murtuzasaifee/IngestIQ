"""
rag_query.py
------------
Retrieval-Augmented Generation: retrieve relevant chunks from Qdrant,
build a context-enriched prompt, and generate an answer via GPT-4o-mini.
"""

import logging
from typing import List, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient

from vector_store import search

logger = logging.getLogger(__name__)

_SEPARATOR = "-" * 60


def _print_retrieved_chunks(hits: List[Tuple[float, dict]]) -> None:
    """Print retrieved chunks to stdout for inspection.

    For table chunks, side-by-side comparison of chunk_text and caption
    is printed so it is easy to verify whether they carry the same content.
    """
    print(f"\n{'=' * 60}")
    print(f"RETRIEVED CHUNKS ({len(hits)} total)")
    print(f"{'=' * 60}")

    for rank, (score, payload) in enumerate(hits, start=1):
        modality   = payload.get("modality", "text")
        page       = payload.get("page_number", "?")
        chunk_id   = payload.get("chunk_id", "?")
        chunk_text = payload.get("chunk_text", "")
        caption    = payload.get("caption", "") or ""

        print(f"\n[{rank}] chunk_id={chunk_id}  page={page}  modality={modality}  score={score:.4f}")
        print(_SEPARATOR)

        if modality in ("image", "table"):
            # Show chunk_text (markdown / extracted text)
            print("  chunk_text:")
            preview = chunk_text[:400].replace("\n", "\n    ")
            print(f"    {preview}" + ("…" if len(chunk_text) > 400 else ""))

            # Show caption (GPT vision caption)
            print("  caption:")
            cap_preview = caption[:400].replace("\n", "\n    ")
            print(f"    {cap_preview}" + ("…" if len(caption) > 400 else ""))

            if modality == "table":
                # Explicit confirmation: are they the same?
                same = chunk_text.strip() == caption.strip()
                print(f"  [table check] chunk_text == caption: {same}")
        else:
            preview = chunk_text[:400].replace("\n", "\n  ")
            print(f"  {preview}" + ("…" if len(chunk_text) > 400 else ""))

    print(f"\n{'=' * 60}\n")


def _build_context(hits: List[Tuple[float, dict]]) -> str:
    """Format retrieved chunks into a readable context block for the LLM prompt."""
    parts = []
    for rank, (score, payload) in enumerate(hits, start=1):
        modality = payload.get("modality", "text")
        page     = payload.get("page_number", "?")
        text     = payload.get("chunk_text", "")

        # For image and table chunks prefer the GPT caption as LLM context
        # (richer natural-language description than raw markdown)
        if modality in ("image", "table"):
            text = payload.get("caption") or text

        parts.append(
            f"[{rank}] (page={page}, modality={modality}, score={score:.3f})\n{text}"
        )

    return "\n\n---\n\n".join(parts)


def answer(
    query: str,
    qdrant: QdrantClient,
    collection_name: str,
    openai_client: OpenAI,
    embedding_model: str,
    chat_model: str,
    top_k: int = 5,
) -> str:
    """
    Full RAG pipeline: retrieve → print chunks → prompt → generate.

    Parameters
    ----------
    query            : User's natural-language question.
    qdrant           : Connected QdrantClient.
    collection_name  : Target collection.
    openai_client    : Initialised OpenAI client.
    embedding_model  : Embedding model for query vectorisation.
    chat_model       : Chat model for answer generation.
    top_k            : Number of chunks to retrieve.

    Returns
    -------
    Generated answer string.
    """
    hits = search(
        qdrant, collection_name, query, openai_client, embedding_model, top_k
    )

    if not hits:
        return "No relevant content found in the document."

    _print_retrieved_chunks(hits)

    context = _build_context(hits)

    system_prompt = (
        "You are a document QA assistant. Answer the user's question using ONLY "
        "the provided document context. If the answer is not in the context, say so."
    )

    user_prompt = (
        f"Document context:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_completion_tokens=512,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
