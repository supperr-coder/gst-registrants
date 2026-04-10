"""
Tests for indexing/embed.py
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_response(n: int, dims: int = 1024) -> MagicMock:
    """Helper: build a mock openai embeddings response with n items."""
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1] * dims) for _ in range(n)]
    return response


def test_embed_batch_returns_correct_shape():
    """embed_batch should return shape (n, EMBEDDING_DIMENSIONS)."""
    from indexing.embed import embed_batch

    client = MagicMock()
    client.embeddings.create.return_value = _make_mock_response(3)

    result = embed_batch(client, ["A", "B", "C"])

    assert result.shape == (3, 1024)
    assert result.dtype == np.float32


def test_embed_batch_returns_normalised_vectors():
    """Returned embeddings should be L2-normalised (norm ≈ 1.0)."""
    from indexing.embed import embed_batch

    client = MagicMock()
    client.embeddings.create.return_value = _make_mock_response(2)

    result = embed_batch(client, ["X", "Y"])
    norms = np.linalg.norm(result, axis=1)

    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_batch_retries_on_failure():
    """embed_batch should retry on API error and succeed on second attempt."""
    from indexing.embed import embed_batch

    client = MagicMock()
    client.embeddings.create.side_effect = [
        Exception("transient API error"),
        _make_mock_response(1),
    ]

    with patch("time.sleep"):
        result = embed_batch(client, ["TEST"], retries=2)

    assert result.shape == (1, 1024)
    assert client.embeddings.create.call_count == 2


def test_embed_batch_raises_after_all_retries_exhausted():
    """embed_batch should re-raise if every attempt fails."""
    from indexing.embed import embed_batch

    client = MagicMock()
    client.embeddings.create.side_effect = Exception("persistent error")

    with patch("time.sleep"):
        with pytest.raises(Exception, match="persistent error"):
            embed_batch(client, ["TEST"], retries=2)


def test_embed_names_uppercases_input():
    """embed_names should uppercase all names before sending to the API."""
    from indexing import embed as embed_module

    with patch.object(embed_module, "_get_client") as mock_get_client, \
         patch("time.sleep"):
        client = MagicMock()
        client.embeddings.create.return_value = _make_mock_response(2)
        mock_get_client.return_value = client

        embed_module.embed_names(["abc trading", "xyz pte ltd"])

        call_args = client.embeddings.create.call_args
        sent_texts = call_args.kwargs.get("input") or call_args.args[0]
        # input is passed as a keyword argument in the real call
        assert all(t == t.upper() for t in sent_texts)
