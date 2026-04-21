"""
Entrypoint for the one-time (or periodic) indexing pipeline.

Usage (from project root):
    python -m indexing.run_indexing
    python -m indexing.run_indexing --entity-column entity_name
"""
import argparse
import io
import logging

import boto3
import pandas as pd

from config import S3_BUCKET, S3_GST_FILE
from indexing.build_index import build_faiss_index, save_artifacts_to_s3
from indexing.embed import embed_names, clear_checkpoints

logger = logging.getLogger(__name__)


def _detect_entity_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect which column holds entity names.

    Tries a list of common column name patterns before falling back to the
    first object-typed column.

    Args:
        df: DataFrame whose columns are inspected.

    Returns:
        The column name to use.

    Raises:
        ValueError: If no suitable column is found.
    """
    candidates = [
        "entity_name",
        "name",
        "company_name",
        "business_name",
        "registrant_name",
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_cols:
            return lower_cols[candidate]

    string_cols = df.select_dtypes(include="object").columns.tolist()
    if string_cols:
        logger.warning(
            "Could not find a known entity column; defaulting to '%s'", string_cols[0]
        )
        return string_cols[0]

    raise ValueError(
        f"Cannot detect entity name column. Available columns: {df.columns.tolist()}"
    )


def load_gst_entities(entity_column: str | None = None) -> list[str]:
    """
    Load entity names from the file specified by S3_GST_FILE in config.

    Args:
        entity_column: Explicit column name to use. Auto-detected if None.

    Returns:
        Deduplicated list of uppercased entity name strings.
    """
    s3 = boto3.client("s3")

    logger.info("Loading s3://%s/%s", S3_BUCKET, S3_GST_FILE)
    body = s3.get_object(Bucket=S3_BUCKET, Key=S3_GST_FILE)["Body"].read()

    if S3_GST_FILE.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(body))
    elif S3_GST_FILE.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(body))
    else:
        raise ValueError(f"Unsupported file type: {S3_GST_FILE}. Use .csv or .parquet.")

    col = entity_column or _detect_entity_column(df)
    all_names = df[col].dropna().str.upper().tolist()

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_names = [n for n in all_names if not (n in seen or seen.add(n))]  # type: ignore[func-returns-value]
    logger.info("Loaded %d unique entity names from column '%s'", len(unique_names), col)
    return unique_names


def run_indexing(entity_column: str | None = None) -> None:
    """
    Full indexing pipeline: load → embed → build FAISS index → save to S3.

    Args:
        entity_column: Explicit entity column name. Auto-detected if None.
    """
    logger.info("=== Starting indexing pipeline ===")

    names = load_gst_entities(entity_column=entity_column)

    logger.info("Embedding %d entity names (this may take a while)...", len(names))
    embeddings = embed_names(names)

    logger.info("Building FAISS index...")
    index = build_faiss_index(embeddings)

    metadata = pd.DataFrame({"entity_name": names})
    save_artifacts_to_s3(index, metadata)

    clear_checkpoints()
    logger.info("=== Indexing complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GST entity indexing pipeline")
    parser.add_argument(
        "--entity-column",
        default=None,
        help="Column name containing entity names (auto-detected if omitted)",
    )
    args = parser.parse_args()
    run_indexing(entity_column=args.entity_column)
