"""
enrichment.py
-------------
Enriches DocumentChunks with:
  - Image captions (GPT-4o-mini vision) for image modality chunks.
  - Additional computed metadata fields.
"""

import logging
from typing import List

from openai import OpenAI

from document_processor import DocumentChunk

logger = logging.getLogger(__name__)


def _generate_image_caption(client: OpenAI, image_b64: str, model: str) -> str:
    """
    Call the OpenAI vision API to generate a short caption for a cropped image.
    Returns an empty string on failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",   # keeps token cost minimal
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Provide a concise one-sentence caption describing "
                                "this image from a document."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=128,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Image captioning failed: %s", exc)
        return ""


def enrich_chunks(
    chunks: List[DocumentChunk],
    openai_client: OpenAI,
    chat_model: str,
) -> List[DocumentChunk]:
    """
    Mutates each DocumentChunk in-place by adding:
      - image_caption  (image modality only)
      - word_count     (text / table)
      - char_count

    Parameters
    ----------
    chunks : List[DocumentChunk]
    openai_client : OpenAI
        Initialised OpenAI client.
    chat_model : str
        Model name (e.g. "gpt-4o-mini").

    Returns
    -------
    The same list with enriched chunks.
    """
    for chunk in chunks:
        # ---- universal metadata ----
        chunk.extra_metadata["word_count"] = len(chunk.chunk_text.split())
        chunk.extra_metadata["char_count"] = len(chunk.chunk_text)

        # ---- image-specific enrichment ----
        if chunk.modality == "image" and chunk.image_base64:
            logger.info("Generating caption for chunk %s …", chunk.chunk_id)
            chunk.image_caption = _generate_image_caption(
                openai_client, chunk.image_base64, chat_model
            )
            # Use caption as searchable text if layout text is sparse
            if not chunk.chunk_text and chunk.image_caption:
                chunk.chunk_text = chunk.image_caption

    return chunks