"""
enrichment.py
-------------
Enriches Chunk objects with:
  - caption (GPT vision) for image AND table modality chunks
  - word_count and char_count in metadata for all chunks
"""

import logging
from typing import List

from openai import OpenAI

from chunker import Chunk

logger = logging.getLogger(__name__)

# Prompts tailored per modality so the model focuses on the right content
_CAPTION_PROMPTS: dict[str, str] = {
    "image": (
        "You are a document analyst. Describe this figure in detail: what it shows, "
        "key data points or trends visible, and its likely purpose in the document. "
        "Be thorough but concise."
    ),
    "table": (
        "You are a document analyst. Describe this table in detail: its headers, "
        "the type of data it contains, notable values or patterns, and its likely "
        "purpose in the document. Be thorough but concise."
    ),
}


def _generate_caption(client: OpenAI, image_b64: str, model: str, modality: str) -> str:
    """Call the OpenAI vision API to caption a cropped document region.

    Args:
        client: Initialised OpenAI client.
        image_b64: Base64-encoded JPEG of the cropped region.
        model: Vision-capable chat model to use.
        modality: "image" or "table" — selects the caption prompt.

    Returns:
        Caption string, or empty string on failure.
    """
    prompt = _CAPTION_PROMPTS.get(modality, _CAPTION_PROMPTS["image"])
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
                            "detail": "low",  # minimise token cost
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }],
            max_completion_tokens=1000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Caption generation failed for %s chunk: %s", modality, exc)
        return ""


def enrich_chunks(
    chunks: List[Chunk],
    openai_client: OpenAI,
    chat_model: str,
) -> List[Chunk]:
    """Enrich chunks in-place with captions (image + table chunks) and token counts (all).

    Args:
        chunks: Chunks produced by chunk_document.
        openai_client: Initialised OpenAI client.
        chat_model: Vision-capable model for captioning (e.g. gpt-4o-mini).

    Returns:
        The same list with enriched chunks.
    """
    for chunk in chunks:
        chunk.metadata["word_count"] = len(chunk.text.split())
        chunk.metadata["char_count"] = len(chunk.text)

        if chunk.modality in ("image", "table") and chunk.image_base64:
            logger.info(
                "Generating %s caption for chunk %s …", chunk.modality, chunk.chunk_id
            )
            chunk.caption = _generate_caption(
                openai_client, chunk.image_base64, chat_model, chunk.modality
            )
            # For image chunks with no extracted text, fall back to caption as searchable text
            if chunk.modality == "image" and not chunk.text.strip() and chunk.caption:
                chunk.text = chunk.caption

    return chunks
