"""
vector_store.py
---------------
Handles all interactions with a local Qdrant instance:
  - Collection creation
  - Upserting enriched DocumentChunks with embeddings
  - Semantic search
"""

import logging
import uuid
from typing import List, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from document_processor import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def get_qdrant_client(url: str) -> QdrantClient:
    """
    Return a QdrantClient

    Parameters
    ----------
    url : str
        URL of the Qdrant instance (e.g. "http://localhost:6333").
    """
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Uses cosine similarity — standard for text embeddings.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", collection_name)
    else:
        logger.info("Qdrant collection '%s' already exists", collection_name)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_text(client: OpenAI, text: str, model: str) -> List[float]:
    """Return the embedding vector for a single text string."""
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    qdrant: QdrantClient,
    collection_name: str,
    chunks: List[DocumentChunk],
    openai_client: OpenAI,
    embedding_model: str,
) -> None:
    """
    Embed each chunk's text and upsert it into Qdrant with full metadata payload.

    Payload structure stored per point
    -----------------------------------
    chunk_id       : str
    modality       : "text" | "table" | "image"
    chunk_text     : str
    page_number    : int
    layout_type    : str
    bbox           : dict | None
    image_base64   : str | None   (image modality only)
    image_caption  : str | None   (image modality only)
    **extra_metadata fields flattened in
    """
    points: List[PointStruct] = []

    for chunk in chunks:
        # Text to embed: prefer caption for images, otherwise chunk_text
        embed_text = (
            chunk.image_caption or chunk.chunk_text
            if chunk.modality == "image"
            else chunk.chunk_text
        )

        if not embed_text.strip():
            logger.debug("Skipping empty chunk %s", chunk.chunk_id)
            continue

        vector = _embed_text(openai_client, embed_text, embedding_model)

        # Build flat payload
        payload = {
            "chunk_id":     chunk.chunk_id,
            "modality":     chunk.modality,
            "chunk_text":   chunk.chunk_text,
            "page_number":  chunk.page_number,
            "layout_type":  chunk.layout_type,
            "bbox":         chunk.bbox,
            "image_base64": chunk.image_base64,
            "image_caption": chunk.image_caption,
            **chunk.extra_metadata,
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)
        logger.info("Upserted %d points into '%s'", len(points), collection_name)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    qdrant: QdrantClient,
    collection_name: str,
    query: str,
    openai_client: OpenAI,
    embedding_model: str,
    top_k: int = 5,
) -> List[Tuple[float, dict]]:
    """
    Embed the query and return the top-k matching chunks.

    Returns
    -------
    List of (score, payload) tuples ordered by relevance (highest score first).
    """
    query_vector = _embed_text(openai_client, query, embedding_model)

    # qdrant-client >= 1.x removed .search(); use .query_points() instead.
    # Returns QueryResponse with a .points list of ScoredPoint objects.
    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [(hit.score, hit.payload) for hit in response.points]