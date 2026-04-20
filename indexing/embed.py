"""
Embed entity names via the text-embedding-3-large API.

Handles batching, rate limiting (50 RPM), retries, and S3 checkpointing.
"""
import logging
import time
from typing import List

import boto3
import numpy as np
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    RATE_LIMIT_RPM,
    S3_BUCKET,
    S3_CHECKPOINT_PREFIX,
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


def _get_last_checkpoint() -> int:
    """Return the last completed batch number from S3, or -1 if none."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"{S3_CHECKPOINT_PREFIX}last_completed.txt",
        )
        return int(obj["Body"].read().decode().strip())
    except s3.exceptions.NoSuchKey:
        return -1
    except Exception:
        return -1


def _save_checkpoint(batch_num: int, embeddings: np.ndarray) -> None:
    """Save a batch's embeddings and update the progress marker in S3."""
    s3 = boto3.client("s3")
    path = f"/tmp/checkpoint_{batch_num}.npy"
    np.save(path, embeddings)
    s3.upload_file(path, S3_BUCKET, f"{S3_CHECKPOINT_PREFIX}batch_{batch_num}.npy")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{S3_CHECKPOINT_PREFIX}last_completed.txt",
        Body=str(batch_num).encode(),
    )


def _load_checkpoints(num_batches: int) -> List[np.ndarray]:
    """Load all completed checkpoint embeddings from S3."""
    s3 = boto3.client("s3")
    results = []
    for i in range(num_batches):
        path = f"/tmp/checkpoint_{i}.npy"
        s3.download_file(S3_BUCKET, f"{S3_CHECKPOINT_PREFIX}batch_{i}.npy", path)
        results.append(np.load(path))
    return results


def clear_checkpoints() -> None:
    """Delete all checkpoint files from S3. Call after indexing completes."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_CHECKPOINT_PREFIX):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
    logger.info("Cleared checkpoints from s3://%s/%s", S3_BUCKET, S3_CHECKPOINT_PREFIX)


def embed_names(names: List[str], use_checkpoints: bool = True) -> np.ndarray:
    """
    Embed a list of entity names with batching, rate limiting, and optional checkpointing.

    Args:
        names: List of entity name strings.
        use_checkpoints: If True, save/resume from S3 checkpoints. Set False for small test runs.

    Returns:
        Float32 array of shape (len(names), EMBEDDING_DIMENSIONS), L2-normalised.
    """
    client = _get_client()
    min_interval = 60.0 / RATE_LIMIT_RPM

    batches = [
        names[i : i + EMBEDDING_BATCH_SIZE]
        for i in range(0, len(names), EMBEDDING_BATCH_SIZE)
    ]
    total_batches = len(batches)

    last_completed = _get_last_checkpoint() if use_checkpoints else -1
    if last_completed >= 0:
        logger.info(
            "Resuming from checkpoint: %d / %d batches already done",
            last_completed + 1,
            total_batches,
        )
        all_embeddings = _load_checkpoints(last_completed + 1)
    else:
        all_embeddings = []

    for batch_num in range(last_completed + 1, total_batches):
        batch = [n.upper() for n in batches[batch_num]]
        t_start = time.time()

        embeddings = embed_batch(client, batch)
        all_embeddings.append(embeddings)
        if use_checkpoints:
            _save_checkpoint(batch_num, embeddings)

        elapsed = time.time() - t_start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        logger.info(
            "Embedded %d / %d names (batch %d/%d%s)",
            min((batch_num + 1) * EMBEDDING_BATCH_SIZE, len(names)),
            len(names),
            batch_num + 1,
            total_batches,
            ", checkpointed" if use_checkpoints else "",
        )

    return np.vstack(all_embeddings)
