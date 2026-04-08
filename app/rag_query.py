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


def _build_context(hits: List[Tuple[float, dict]]) -> str:
    """
    Format retrieved chunks into a readable context block for the LLM prompt.
    Includes modality label so the model is aware of content type.
    """
    parts = []
    for rank, (score, payload) in enumerate(hits, start=1):
        modality = payload.get("modality", "text")
        page     = payload.get("page_number", "?")
        text     = payload.get("chunk_text", "")

        # For image chunks prefer the generated caption as context
        if modality == "image":
            text = payload.get("image_caption") or text

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
    Full RAG pipeline: retrieve → prompt → generate.

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
        max_tokens=512,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()