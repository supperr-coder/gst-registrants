"""
End-to-end matching pipeline: query names → ranked results DataFrame.
"""
import logging
from typing import List

import pandas as pd

from config import MATCH_TOP_N, SIMILARITY_THRESHOLD
from matching.search import search

logger = logging.getLogger(__name__)


def match_entities(query_names: List[str]) -> pd.DataFrame:
    """
    Match a list of query entity names against the GST-registered index.

    For each query name, retrieves up to MATCH_TOP_N candidates whose cosine
    similarity meets or exceeds SIMILARITY_THRESHOLD, ranked best-first.
    Queries with no match above the threshold produce a single row with
    matched_entity=None.

    Args:
        query_names: List of entity name strings to match.

    Returns:
        DataFrame with columns:
            - query_name (str)
            - matched_entity (str | None)
            - score (float | None): cosine similarity
            - rank (int | None): 1-based rank among matches for this query
    """
    candidates_per_query = search(query_names)

    rows: List[dict] = []
    for query_name, candidates in zip(query_names, candidates_per_query):
        filtered = [c for c in candidates if c["score"] >= SIMILARITY_THRESHOLD]
        top = filtered[:MATCH_TOP_N]

        if not top:
            rows.append(
                {
                    "query_name": query_name,
                    "matched_entity": None,
                    "score": None,
                    "rank": None,
                }
            )
        else:
            for rank, candidate in enumerate(top, start=1):
                rows.append(
                    {
                        "query_name": query_name,
                        "matched_entity": candidate["entity_name"],
                        "score": round(candidate["score"], 4),
                        "rank": rank,
                    }
                )

    return pd.DataFrame(rows)
