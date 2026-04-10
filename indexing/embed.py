"""
Embed entity names via the text-embedding-3-large API.

Handles batching, rate limiting (50 RPM), and retries.
"""
import logging
import time
from typing import List

import numpy as np
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    RATE_LIMIT_RPM,
)

logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    """Create and return an OpenAI client pointed at the MAESTRO LiteLLM proxy."""
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)


def embed_batch(client: OpenAI, texts: List[str], retries: int = 3) -> np.ndarray:
    """
    Embed a single batch of texts and return L2-normalised embeddings.

    Args:
        client: OpenAI client instance.
        texts: List of strings to embed (max EMBEDDING_BATCH_SIZE).
        retries: Number of retry attempts on API failure.

    Returns:
        Float32 array of shape (len(texts), EMBEDDING_DIMENSIONS), L2-normalised.
    """
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            embeddings = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    "Embedding batch failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                raise


def embed_names(names: List[str]) -> np.ndarray:
    """
    Embed a list of entity names with batching and rate limiting.

    Names are uppercased before embedding for consistency.

    Args:
        names: List of entity name strings.

    Returns:
        Float32 array of shape (len(names), EMBEDDING_DIMENSIONS), L2-normalised.
    """
    client = _get_client()
    min_interval = 60.0 / RATE_LIMIT_RPM  # minimum seconds between requests

    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(names), EMBEDDING_BATCH_SIZE):
        batch = [n.upper() for n in names[i : i + EMBEDDING_BATCH_SIZE]]
        t_start = time.time()

        embeddings = embed_batch(client, batch)
        all_embeddings.append(embeddings)

        elapsed = time.time() - t_start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        logger.info(
            "Embedded %d / %d names",
            min(i + EMBEDDING_BATCH_SIZE, len(names)),
            len(names),
        )

    return np.vstack(all_embeddings)
