"""Tests for the quantization entrypoint (pyturboquant.core or float16 stub)."""

from __future__ import annotations

import numpy as np
import pytest

from physiorag.embeddings.quantization import (
    has_pyturboquant_core,
    is_real_quant_codec,
    quantize_embeddings,
)


def test_quantize_embeddings_shape_and_stats() -> None:
    embeddings = np.random.default_rng(1).normal(size=(4, 16)).astype(np.float32)
    result = quantize_embeddings(embeddings, bits=8)
    assert result.original_shape == (4, 16)
    assert result.codec in {"pyturboquant", "float16_stub"}
    data = np.asarray(result.data)
    assert data.shape == (4, 16)
    assert result.metadata is not None
    assert result.metadata["original_bytes"] == 4 * 16 * 4
    assert result.metadata["compressed_bytes"] > 0
    assert result.metadata["compression_ratio"] >= 1.0
    assert "mean_cosine" in result.metadata
    assert is_real_quant_codec(result.codec) == (result.codec == "pyturboquant")


def test_has_pyturboquant_core_matches_codec_when_installed() -> None:
    embeddings = np.random.default_rng(2).normal(size=(2, 8)).astype(np.float32)
    result = quantize_embeddings(embeddings, bits=4)
    if has_pyturboquant_core():
        assert result.codec == "pyturboquant"
        assert is_real_quant_codec(result.codec)
    else:
        assert result.codec == "float16_stub"


def test_quantize_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        quantize_embeddings(np.zeros(8, dtype=np.float32))
