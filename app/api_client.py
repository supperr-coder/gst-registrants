"""
API client for calling the GST Entity Matcher SageMaker endpoint on MAESTRO.

The Streamlit app on Airbase uses this module instead of importing matching
logic directly — all heavy computation (embedding, FAISS search) happens on
the SageMaker endpoint.
"""
import logging
import os
from typing import List

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# The SageMaker endpoint URL exposed via MAESTRO's API Gateway.
# Set this as an environment variable on Airbase.
SAGEMAKER_ENDPOINT_URL: str = os.getenv("SAGEMAKER_ENDPOINT_URL", "")


def match_entities(query_names: List[str]) -> pd.DataFrame:
    """
    Send entity names to the SageMaker endpoint and return matched results.

    Args:
        query_names: List of entity name strings to match.

    Returns:
        DataFrame with columns: query_name, matched_entity, score, rank

    Raises:
        ConnectionError: If the endpoint URL is not configured.
        requests.HTTPError: If the endpoint returns a non-2xx status.
    """
    if not SAGEMAKER_ENDPOINT_URL:
        raise ConnectionError(
            "SAGEMAKER_ENDPOINT_URL is not set. "
            "Configure it as an environment variable on Airbase."
        )

    response = requests.post(
        SAGEMAKER_ENDPOINT_URL,
        json={"entity_names": query_names},
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=120,
    )
    response.raise_for_status()

    records = response.json()
    return pd.DataFrame(records)
