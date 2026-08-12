"""Strict-mode ingest fails loudly when real compression is unavailable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import physiorag.pipeline.ingest as ingest_mod
from physiorag.embeddings.quantization import QuantizedEmbedding
from physiorag.storage.memory_store import InMemoryVectorStore


def _stub_quant(embeddings: np.ndarray, *, bits: int = 8) -> QuantizedEmbedding:
    arr = np.asarray(embeddings, dtype=np.float16)
    return QuantizedEmbedding(
        data=arr,
        original_shape=tuple(arr.shape),
        bits=16,
        codec="float16_stub",
    )


def _config(tmp_path: Path) -> dict:
    return {
        "data": {"demo_dir": str(tmp_path / "demo")},
        "embeddings": {
            "encoder": "baseline_cnn",
            "embedding_dim": 16,
            "device": "cpu",
            "batch_size": 2,
            "text_encoder_enabled": False,
        },
        "quantization": {"enabled": True, "bits": 8},
        "storage": {"backend": "memory", "array_dir": str(tmp_path / "arrays")},
    }


def test_strict_ingest_rejects_stub_codec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest_mod, "quantize_embeddings", _stub_quant)
    with pytest.raises(RuntimeError) as exc:
        ingest_mod.run_ingest(
            dataset="mimic_demo",
            modality="ventilator",
            config=_config(tmp_path),
            store=InMemoryVectorStore(),
            strict=True,
        )
    assert "pyturboquant" in str(exc.value)


def test_nonstrict_ingest_allows_stub_codec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest_mod, "quantize_embeddings", _stub_quant)
    result = ingest_mod.run_ingest(
        dataset="mimic_demo",
        modality="ventilator",
        config=_config(tmp_path),
        store=InMemoryVectorStore(),
        strict=False,
    )
    assert result["status"] == "ok"
    assert result["quant_codec"] == "float16_stub"
    assert result["strict"] is False
