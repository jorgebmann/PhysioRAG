"""Baseline ingest path against the in-memory vector store."""

from __future__ import annotations

from pathlib import Path

from physiorag.pipeline.ingest import run_ingest
from physiorag.storage.memory_store import InMemoryVectorStore


def test_run_ingest_demo_to_memory(tmp_path: Path) -> None:
    store = InMemoryVectorStore()
    config = {
        "data": {"demo_dir": str(tmp_path / "demo"), "raw_dir": str(tmp_path / "raw")},
        "ingestion": {
            "window_seconds": 10.0,
            "target_sample_rate_hz": 125,
        },
        "embeddings": {
            "encoder": "baseline_cnn",
            "embedding_dim": 32,
            "device": "cpu",
            "batch_size": 2,
            "text_encoder_enabled": False,
        },
        "quantization": {"enabled": True, "bits": 8},
        "storage": {
            "backend": "memory",
            "array_dir": str(tmp_path / "arrays"),
        },
    }
    (tmp_path / "demo").mkdir()

    result = run_ingest(
        dataset="mimic_demo",
        modality="ventilator",
        config=config,
        store=store,
        text_encoder=None,
    )

    assert result["status"] == "ok"
    assert result["epochs_written"] == 4
    hits = store.search_text("ARDS pressure spike", top_k=3)
    assert hits
    assert any("ARDS" in (h.text or "") for h in hits)
    assert Path(hits[0].array_ref).exists()
