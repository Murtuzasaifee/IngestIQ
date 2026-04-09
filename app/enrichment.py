"""
enrichment.py
-------------
Enriches Chunk objects with:
  - image_caption (GPT-4o-mini vision) for image modality chunks
  - word_count and char_count in metadata for all chunks
"""

import logging
from typing import List

from openai import OpenAI

from chunker import Chunk

logger = logging.getLogger(__name__)


def _generate_caption(client: OpenAI, image_b64: str, model: str) -> str:
    """Call the OpenAI vision API to caption a cropped document figure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "low",   # minimise token cost
                        },
                    },
                    {
                        "type": "text",
                        "text": "Provide a concise one-sentence caption for this document figure.",
                    },
                ],
            }],
            max_tokens=128,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Caption generation failed: %s", exc)
        return ""


def enrich_chunks(
    chunks: List[Chunk],
    openai_client: OpenAI,
    chat_model: str,
) -> List[Chunk]:
    """Enrich chunks in-place with captions (image chunks) and token counts (all chunks).

    Args:
        chunks: Chunks produced by chunk_document.
        openai_client: Initialised OpenAI client.
        chat_model: Vision-capable model for image captioning (e.g. gpt-4o-mini).

    Returns:
        The same list with enriched chunks.
    """
    for chunk in chunks:
        chunk.metadata["word_count"] = len(chunk.text.split())
        chunk.metadata["char_count"] = len(chunk.text)

        if chunk.modality == "image" and chunk.image_base64:
            logger.info("Generating caption for chunk %s …", chunk.chunk_id)
            chunk.image_caption = _generate_caption(openai_client, chunk.image_base64, chat_model)
            # Fall back to caption as searchable text if the figure had no extracted text
            if not chunk.text.strip() and chunk.image_caption:
                chunk.text = chunk.image_caption

    return chunks
