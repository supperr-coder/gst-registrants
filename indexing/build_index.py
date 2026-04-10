"""
Build a FAISS IndexFlatIP from pre-computed embeddings and save artifacts to S3.
"""
import io
import json
import logging

import boto3
import faiss
import numpy as np
import pandas as pd

from config import (
    EMBEDDING_DIMENSIONS,
    S3_BUCKET,
    S3_CONFIG_KEY,
    S3_FAISS_KEY,
    S3_METADATA_KEY,
    TMP_FAISS_PATH,
    TMP_METADATA_PATH,
)

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP index from L2-normalised embeddings.

    IndexFlatIP on normalised vectors gives exact cosine similarity search.

    Args:
        embeddings: Float32 array of shape (n, EMBEDDING_DIMENSIONS), L2-normalised.

    Returns:
        Populated faiss.IndexFlatIP.
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors", index.ntotal)
    return index


def save_artifacts_to_s3(
    index: faiss.IndexFlatIP, metadata: pd.DataFrame
) -> None:
    """
    Persist the FAISS index, metadata parquet, and a config JSON to S3.

    Args:
        index: Populated FAISS index.
        metadata: DataFrame with at least an 'entity_name' column.
    """
    s3 = boto3.client("s3")

    # FAISS index
    faiss.write_index(index, TMP_FAISS_PATH)
    s3.upload_file(TMP_FAISS_PATH, S3_BUCKET, S3_FAISS_KEY)
    logger.info("Uploaded FAISS index → s3://%s/%s", S3_BUCKET, S3_FAISS_KEY)

    # Metadata parquet
    metadata.to_parquet(TMP_METADATA_PATH, index=False)
    s3.upload_file(TMP_METADATA_PATH, S3_BUCKET, S3_METADATA_KEY)
    logger.info("Uploaded metadata → s3://%s/%s", S3_BUCKET, S3_METADATA_KEY)

    # Config JSON (useful for validation / auditing)
    config_data = {
        "embedding_model": "text-embedding-3-large",
        "dimensions": EMBEDDING_DIMENSIONS,
        "num_entities": len(metadata),
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_CONFIG_KEY,
        Body=json.dumps(config_data),
        ContentType="application/json",
    )
    logger.info("Uploaded config → s3://%s/%s", S3_BUCKET, S3_CONFIG_KEY)
