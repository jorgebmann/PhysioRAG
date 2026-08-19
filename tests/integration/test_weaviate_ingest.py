"""Integration test against a local Weaviate instance (skipped if unavailable)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physiorag.pipeline.ingest import run_ingest
from physiorag.storage.weaviate_store import WeaviateVectorStore


def _weaviate_available() -> bool:
    try:
        import weaviate

        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        ready = client.is_ready()
        client.close()
        return bool(ready)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _weaviate_available(),
    reason="Local Weaviate not reachable on :8080",
)


def test_ingest_and_search_weaviate(tmp_path: Path) -> None:
    collection = "WaveformEpochTest"
    store = WeaviateVectorStore(
        url="http://127.0.0.1:8080",
        grpc_port=50051,
        collection=collection,
        waveform_dim=32,
        text_dim=384,
    )
    try:
        if store._client.collections.exists(collection):
            store._client.collections.delete(collection)
        store.ensure_schema()

        config = {
            "data": {"demo_dir": str(tmp_path / "demo"), "raw_dir": str(tmp_path / "raw")},
            "ingestion": {"window_seconds": 10.0, "target_sample_rate_hz": 125},
            "embeddings": {
                "encoder": "baseline_cnn",
                "embedding_dim": 32,
                "device": "cpu",
                "batch_size": 8,
                "text_encoder_enabled": False,
            },
            "quantization": {"enabled": True, "bits": 8},
            "storage": {
                "backend": "weaviate",
                "array_dir": str(tmp_path / "arrays"),
                "collection": collection,
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
        assert result["epochs_written"] == 8

        hits = store.search_text("ARDS spontaneous pressure spike", top_k=3)
        assert hits
        assert any("ARDS" in (h.text or "") for h in hits)

        # get_by_epoch_id round-trip for the plotting endpoint
        one = store.get_by_epoch_id(hits[0].epoch_id)
        assert one is not None
        assert one.array_ref
    finally:
        if store._client.collections.exists(collection):
            store._client.collections.delete(collection)
        store.close()
