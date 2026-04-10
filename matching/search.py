"""
Load the FAISS index from S3 and search for top-k candidate matches.

The index and metadata are cached in module-level variables so they are only
downloaded once per process lifetime (important for SageMaker endpoints and
long-running Streamlit sessions).
"""
import io
import logging
from typing import List

import boto3
import faiss
import numpy as np
import pandas as pd

from config import (
    FAISS_TOP_K,
    S3_BUCKET,
    S3_FAISS_KEY,
    S3_METADATA_KEY,
    TMP_FAISS_PATH,
    TMP_METADATA_PATH,
)
from indexing.embed import embed_names

logger = logging.getLogger(__name__)

# Module-level cache — populated on first call to load_index()
_index: faiss.IndexFlatIP | None = None
_metadata: pd.DataFrame | None = None


def load_index() -> tuple[faiss.IndexFlatIP, pd.DataFrame]:
    """
    Download and cache the FAISS index and entity metadata from S3.

    Subsequent calls return the cached objects without hitting S3 again.

    Returns:
        Tuple of (faiss index, metadata DataFrame).
    """
    global _index, _metadata

    if _index is not None and _metadata is not None:
        return _index, _metadata

    s3 = boto3.client("s3")

    logger.info("Downloading FAISS index from s3://%s/%s", S3_BUCKET, S3_FAISS_KEY)
    s3.download_file(S3_BUCKET, S3_FAISS_KEY, TMP_FAISS_PATH)
    _index = faiss.read_index(TMP_FAISS_PATH)
    logger.info("FAISS index loaded (%d vectors)", _index.ntotal)

    logger.info("Downloading metadata from s3://%s/%s", S3_BUCKET, S3_METADATA_KEY)
    body = s3.get_object(Bucket=S3_BUCKET, Key=S3_METADATA_KEY)["Body"].read()
    _metadata = pd.read_parquet(io.BytesIO(body))
    logger.info("Metadata loaded (%d rows)", len(_metadata))

    return _index, _metadata


def search(query_names: List[str], top_k: int = FAISS_TOP_K) -> List[List[dict]]:
    """
    Embed query names and retrieve the top-k nearest neighbours from the FAISS index.

    Args:
        query_names: List of entity name strings to search for.
        top_k: Number of candidates to retrieve per query.

    Returns:
        A list (one entry per query) of candidate lists. Each candidate is a dict:
            - entity_name (str): matched GST-registered name
            - score (float): cosine similarity in [−1, 1]; higher is better
    """
    index, metadata = load_index()

    query_embeddings = embed_names(query_names)  # already uppercased + normalised inside embed_names
    scores, indices = index.search(query_embeddings, top_k)

    results: List[List[dict]] = []
    for score_row, idx_row in zip(scores, indices):
        candidates = [
            {
                "entity_name": metadata.iloc[idx]["entity_name"],
                "score": float(score),
            }
            for score, idx in zip(score_row, idx_row)
            if idx != -1
        ]
        results.append(candidates)

    return results
