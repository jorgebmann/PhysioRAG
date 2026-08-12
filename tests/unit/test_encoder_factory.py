"""Tests for the signal-encoder registry/factory."""

from __future__ import annotations

import pytest

from physiorag.embeddings.base import TimeSeriesEncoder
from physiorag.embeddings.factory import available_encoders, build_encoder


def test_baseline_cnn_is_registered() -> None:
    assert "baseline_cnn" in available_encoders()


def test_build_baseline_cnn_honors_config_dim() -> None:
    encoder = build_encoder({"embeddings": {"encoder": "baseline_cnn", "embedding_dim": 16}})
    assert isinstance(encoder, TimeSeriesEncoder)
    assert encoder.embedding_dim == 16


def test_build_encoder_rejects_unknown() -> None:
    with pytest.raises(ValueError) as exc:
        build_encoder({"embeddings": {"encoder": "chronos"}})
    message = str(exc.value)
    assert "chronos" in message
    assert "baseline_cnn" in message  # lists what is available
