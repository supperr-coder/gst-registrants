"""
SageMaker real-time endpoint handler.

SageMaker calls these four functions in order:
  model_fn     → load artefacts into memory
  input_fn     → deserialise the HTTP request body
  predict_fn   → run inference
  output_fn    → serialise the response

Expected request format (application/json):
    ["ENTITY A", "ENTITY B", ...]
  or
    {"entity_names": ["ENTITY A", "ENTITY B", ...]}

Supported accept types: application/json, text/csv
"""
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _install_requirements(model_dir: str) -> None:
    """Install packages from requirements.txt if present in the model archive."""
    req_path = os.path.join(model_dir, "requirements.txt")
    if os.path.exists(req_path):
        logger.info("Installing dependencies from %s", req_path)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", req_path, "--quiet"]
        )


def model_fn(model_dir: str) -> dict:
    """
    Load the FAISS index and metadata into memory.

    SageMaker calls this once when the endpoint starts up.

    Args:
        model_dir: Path to the directory where model artefacts are unpacked.

    Returns:
        A dict with 'index' and 'metadata' keys (the in-memory cache).
    """
    # Install extra packages (faiss-cpu, openai, python-dotenv, etc.)
    _install_requirements(model_dir)

    # Add model_dir to Python path so our modules are importable
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    import pandas as pd
    from matching.search import load_index

    index, metadata = load_index()
    logger.info("Model loaded: %d entities indexed", len(metadata))
    return {"index": index, "metadata": metadata}


def input_fn(request_body: str, content_type: str = "application/json") -> list[str]:
    """
    Deserialise the incoming HTTP request body.

    Args:
        request_body: Raw request body string.
        content_type: MIME type of the request.

    Returns:
        List of entity name strings.

    Raises:
        ValueError: For unsupported content types or malformed payloads.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")

    data = json.loads(request_body)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "entity_names" in data:
        return data["entity_names"]

    raise ValueError(
        'Expected a JSON array or {"entity_names": [...]} object.'
    )


def predict_fn(data: list[str], model: dict):
    """
    Run the matching pipeline against the loaded index.

    Args:
        data: List of entity name strings from input_fn.
        model: The dict returned by model_fn (index + metadata cache).

    Returns:
        Results DataFrame from match_entities().
    """
    from matching.pipeline import match_entities

    return match_entities(data)


def output_fn(prediction, accept: str = "application/json") -> str:
    """
    Serialise the prediction DataFrame for the HTTP response.

    Args:
        prediction: DataFrame returned by predict_fn.
        accept: Desired response MIME type.

    Returns:
        Serialised string.

    Raises:
        ValueError: For unsupported accept types.
    """
    if accept == "application/json":
        return prediction.to_json(orient="records")
    if accept == "text/csv":
        return prediction.to_csv(index=False)

    raise ValueError(f"Unsupported accept type: {accept}. Use application/json or text/csv.")
