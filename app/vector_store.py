"""
vector_store.py
---------------
Handles all interactions with Qdrant:
  - Collection creation
  - Upserting enriched Chunks with embeddings
  - Semantic search

Uses qdrant-client >= 1.x API: query_points() (search() was removed in v1).
"""

import logging
import uuid
from typing import List, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_qdrant_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s'", collection_name)
    else:
        logger.info("Collection '%s' already exists", collection_name)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed(client: OpenAI, text: str, model: str) -> List[float]:
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    qdrant: QdrantClient,
    collection_name: str,
    chunks: List[Chunk],
    openai_client: OpenAI,
    embedding_model: str,
) -> None:
    """Embed each chunk and upsert into Qdrant with full metadata payload."""
    points: List[PointStruct] = []

    for chunk in chunks:
        # For image/table chunks prefer the GPT caption for embedding (richer semantics)
        embed_text = (
            chunk.caption or chunk.text
            if chunk.modality in ("image", "table")
            else chunk.text
        )
        if not embed_text.strip():
            logger.debug("Skipping empty chunk %s", chunk.chunk_id)
            continue

        vector = _embed(openai_client, embed_text, embedding_model)
        payload = {
            "chunk_id":      chunk.chunk_id,
            "modality":      chunk.modality,
            "chunk_text":    chunk.text,
            "page_number":   chunk.page,
            "elements":      chunk.elements,
            "bbox":          chunk.bbox,
            "image_base64":  chunk.image_base64,
            "caption": chunk.caption,
            **chunk.metadata,
        }
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

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
    """Embed the query and return the top-k matching chunks as (score, payload) tuples."""
    query_vector = _embed(openai_client, query, embedding_model)
    # qdrant-client >= 1.x: search() removed; use query_points()
    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [(hit.score, hit.payload) for hit in response.points]
