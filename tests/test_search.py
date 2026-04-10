"""
Tests for matching/search.py
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _make_index(n_vectors: int = 10, dims: int = 1024):
    """Build a small in-memory FAISS index for testing."""
    import faiss

    index = faiss.IndexFlatIP(dims)
    vectors = np.random.randn(n_vectors, dims).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index.add(vectors)
    return index


def _make_metadata(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"entity_name": [f"ENTITY_{i}" for i in range(n)]})


def test_search_returns_one_list_per_query():
    """search() should return exactly one candidate list per input query."""
    import matching.search as search_module

    mock_index = _make_index()
    mock_metadata = _make_metadata()
    mock_embeddings = np.random.randn(3, 1024).astype(np.float32)
    mock_embeddings /= np.linalg.norm(mock_embeddings, axis=1, keepdims=True)

    with patch.object(search_module, "_index", mock_index), \
         patch.object(search_module, "_metadata", mock_metadata), \
         patch("matching.search.embed_names", return_value=mock_embeddings):

        results = search_module.search(["A", "B", "C"], top_k=5)

    assert len(results) == 3


def test_search_candidate_structure():
    """Each candidate dict must have 'entity_name' and 'score' keys."""
    import matching.search as search_module

    mock_index = _make_index()
    mock_metadata = _make_metadata()
    mock_embeddings = np.random.randn(1, 1024).astype(np.float32)
    mock_embeddings /= np.linalg.norm(mock_embeddings, axis=1, keepdims=True)

    with patch.object(search_module, "_index", mock_index), \
         patch.object(search_module, "_metadata", mock_metadata), \
         patch("matching.search.embed_names", return_value=mock_embeddings):

        results = search_module.search(["TEST ENTITY"], top_k=5)

    for candidate in results[0]:
        assert "entity_name" in candidate
        assert "score" in candidate
        assert isinstance(candidate["score"], float)


def test_search_respects_top_k():
    """search() should return at most top_k candidates per query."""
    import matching.search as search_module

    mock_index = _make_index(n_vectors=10)
    mock_metadata = _make_metadata(n=10)
    mock_embeddings = np.random.randn(1, 1024).astype(np.float32)
    mock_embeddings /= np.linalg.norm(mock_embeddings, axis=1, keepdims=True)

    with patch.object(search_module, "_index", mock_index), \
         patch.object(search_module, "_metadata", mock_metadata), \
         patch("matching.search.embed_names", return_value=mock_embeddings):

        results = search_module.search(["TEST"], top_k=3)

    assert len(results[0]) <= 3


def test_search_scores_are_valid_cosine_range():
    """Cosine similarity scores should be in [−1, 1]."""
    import matching.search as search_module

    mock_index = _make_index()
    mock_metadata = _make_metadata()
    mock_embeddings = np.random.randn(1, 1024).astype(np.float32)
    mock_embeddings /= np.linalg.norm(mock_embeddings, axis=1, keepdims=True)

    with patch.object(search_module, "_index", mock_index), \
         patch.object(search_module, "_metadata", mock_metadata), \
         patch("matching.search.embed_names", return_value=mock_embeddings):

        results = search_module.search(["TEST"])

    for candidate in results[0]:
        assert -1.0 <= candidate["score"] <= 1.0
