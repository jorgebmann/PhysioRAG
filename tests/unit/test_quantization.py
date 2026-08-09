"""Tests for the quantization entrypoint (float16 stub path)."""

from __future__ import annotations

import numpy as np
import pytest

from physiorag.embeddings.quantization import quantize_embeddings


def test_quantize_embeddings_stub_shape() -> None:
    embeddings = np.random.default_rng(1).normal(size=(4, 16)).astype(np.float32)
    result = quantize_embeddings(embeddings, bits=8)
    assert result.original_shape == (4, 16)
    assert result.codec in {"pyturboquant", "float16_stub"}
    data = np.asarray(result.data)
    assert data.shape == (4, 16)


def test_quantize_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        quantize_embeddings(np.zeros(8, dtype=np.float32))
