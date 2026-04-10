"""
Helper utilities for the Streamlit app.
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def detect_entity_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect which column in a DataFrame contains entity names.

    Tries a list of common column name patterns (case-insensitive) before
    falling back to the first object-typed column.

    Args:
        df: DataFrame to inspect.

    Returns:
        The column name to use for entity names.

    Raises:
        ValueError: If no suitable column can be identified.
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
        return string_cols[0]

    raise ValueError(
        f"Cannot detect entity name column. Available columns: {df.columns.tolist()}"
    )


def parse_uploaded_csv(
    uploaded_file, entity_column: Optional[str] = None
) -> tuple[pd.DataFrame, str]:
    """
    Parse a Streamlit-uploaded CSV file.

    Args:
        uploaded_file: File-like object from st.file_uploader.
        entity_column: Explicit column name to use. Auto-detected if None or empty.

    Returns:
        Tuple of (DataFrame, resolved entity column name).

    Raises:
        ValueError: If the specified column does not exist or cannot be detected.
    """
    df = pd.read_csv(uploaded_file)

    if entity_column and entity_column in df.columns:
        return df, entity_column
    elif entity_column and entity_column not in df.columns:
        raise ValueError(
            f"Column '{entity_column}' not found. Available: {df.columns.tolist()}"
        )

    col = detect_entity_column(df)
    return df, col


def results_to_csv_bytes(results_df: pd.DataFrame) -> bytes:
    """
    Serialise a results DataFrame to UTF-8 CSV bytes for st.download_button.

    Args:
        results_df: DataFrame returned by match_entities().

    Returns:
        CSV-encoded bytes.
    """
    return results_df.to_csv(index=False).encode("utf-8")
