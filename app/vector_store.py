"""
vector_store.py
---------------
Handles all interactions with Qdrant:
  - Collection creation
  - Upserting enriched Chunks with embeddings (batched — one API call)
  - Semantic search
  - Re-ingest deduplication by source

Uses qdrant-client >= 1.x API: query_points() (search() was removed in v1).
"""

import base64
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

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
    """Embed a single text string — used at query time."""
    return client.embeddings.create(input=[text], model=model).data[0].embedding


_EMBED_BATCH_SIZE = 2048  # OpenAI hard limit per request


def _embed_batch(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """Embed multiple texts and return vectors in input order.

    Splits into pages of up to 2048 (OpenAI hard limit) and concatenates
    results, so callers never need to think about batch sizing.
    """
    vectors: List[List[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    return vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_crop(image_base64: str, crops_dir: str, chunk_id: str) -> str:
    """Decode a base64 JPEG and write it to <crops_dir>/<chunk_id>.jpg.

    Returns the absolute file path so it can be stored in the Qdrant payload.
    """
    path = Path(crops_dir) / f"{chunk_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(image_base64))
    return str(path.resolve())


def delete_by_source(qdrant: QdrantClient, collection_name: str, source: str) -> None:
    """Delete all Qdrant points whose payload `source` matches the given value.

    Used for re-ingest deduplication: call before upserting to prevent the
    same document from accumulating duplicate points across multiple ingests.
    """
    qdrant.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )
        ),
    )
    logger.info("Deleted existing points for source '%s'", source)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    qdrant: QdrantClient,
    collection_name: str,
    chunks: List[Chunk],
    openai_client: OpenAI,
    embedding_model: str,
    crops_dir: Optional[str] = None,
    deduplicate: bool = True,
) -> None:
    """Embed all chunks in one batched API call and upsert into Qdrant.

    Args:
        qdrant:           Connected QdrantClient.
        collection_name:  Target Qdrant collection.
        chunks:           Enriched Chunk objects from enrich_chunks().
        openai_client:    Initialised OpenAI client.
        embedding_model:  Embedding model name (must match VECTOR_SIZE).
        crops_dir:        Directory to write image/table JPEG crops. When set,
                          the file path is stored in the payload as `image_path`
                          instead of the raw base64 — avoids binary bloat in
                          the Qdrant index. When None, crops are not persisted.
        deduplicate:      Delete existing points with the same `source` before
                          inserting. Prevents index doubling on re-ingest.
    """
    # --- Select embed text per chunk; drop empties ---
    embed_texts: List[str] = []
    valid_chunks: List[Chunk] = []

    for chunk in chunks:
        # For image/table chunks prefer the GPT caption — richer semantics than
        # raw markdown or empty string. Fall back to chunk.text if no caption.
        embed_text = (
            chunk.caption or chunk.text
            if chunk.modality in ("image", "table")
            else chunk.text
        )
        if not embed_text.strip():
            logger.debug("Skipping empty chunk %s", chunk.chunk_id)
            continue
        embed_texts.append(embed_text)
        valid_chunks.append(chunk)

    if not valid_chunks:
        logger.info("No valid chunks to upsert.")
        return

    # --- Deduplication: remove stale points for each source ---
    if deduplicate:
        sources = {c.metadata.get("source") for c in valid_chunks if c.metadata.get("source")}
        for source in sources:
            delete_by_source(qdrant, collection_name, source)

    # --- Batch embed: single API call for all chunks ---
    logger.info("Embedding %d chunks in one batch request…", len(valid_chunks))
    vectors = _embed_batch(openai_client, embed_texts, embedding_model)

    # --- Build Qdrant points ---
    points: List[PointStruct] = []
    for chunk, embed_text, vector in zip(valid_chunks, embed_texts, vectors):
        image_path: Optional[str] = None
        if chunk.image_base64 and crops_dir:
            image_path = _save_crop(chunk.image_base64, crops_dir, chunk.chunk_id)

        payload = {
            "chunk_id":    chunk.chunk_id,
            "modality":    chunk.modality,
            # chunk_text always matches the embedded vector for consistent retrieval
            "chunk_text":  embed_text,
            "caption":     chunk.caption,
            "page_number": chunk.page,
            "elements":    chunk.elements,
            "bbox":        chunk.bbox,
            # file path to the JPEG crop (set when crops_dir is configured)
            "image_path":  image_path,
            **chunk.metadata,
        }
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

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
    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [(hit.score, hit.payload) for hit in response.points]
