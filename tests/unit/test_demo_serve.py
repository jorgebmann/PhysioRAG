"""Public demo path: API auto-ingests synthetic data into memory on startup.

Proves `scripts/serve_demo.py` works as a single command without Weaviate,
Ollama, or a downloaded text encoder: the in-memory store is populated by the
lifespan handler and `/search` returns plottable hits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


def _demo_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "runtime": {"strict": False, "auto_ingest_demo": True},
        "data": {"demo_dir": str(tmp_path / "demo"), "raw_dir": str(tmp_path / "raw")},
        "ingestion": {
            "dataset": "mimic_demo",
            "modality": "ventilator",
            "window_seconds": 10.0,
            "target_sample_rate_hz": 125,
        },
        "embeddings": {
            "encoder": "baseline_cnn",
            "embedding_dim": 32,
            "device": "cpu",
            "batch_size": 2,
            # No network: keep MiniLM out of the unit test. Search degrades to
            # keyword/BM25 over the captions, which is enough to return hits.
            "text_encoder_enabled": False,
        },
        "quantization": {"enabled": True, "bits": 8},
        "storage": {
            "backend": "memory",
            "array_dir": str(tmp_path / "arrays"),
            "collection": "WaveformEpochDemo",
        },
        "retrieval": {"mode": "hybrid_text", "top_k": 5},
        # No Ollama in the unit test.
        "synthesis": {"enabled": False},
    }


def test_serve_demo_auto_ingests_and_searches(monkeypatch, tmp_path: Path) -> None:
    import api.main as api_main

    monkeypatch.setattr(api_main, "load_config", lambda *a, **k: _demo_config(tmp_path))

    with TestClient(api_main.app) as client:
        # The store was populated by the lifespan auto-ingest, not the test.
        store = api_main._state["store"]
        assert len(getattr(store, "_records", {})) > 0

        health = client.get("/health")
        assert health.status_code == 200
        hbody = health.json()
        assert hbody["status"] == "ok"  # non-strict, degraded components tolerated
        assert hbody["store"] == "memory"

        resp = client.post(
            "/search",
            json={"query": "ARDS pressure spike asynchrony", "top_k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["hits"], "expected at least one hit from the synthetic demo"
        top = body["hits"][0]
        assert top["plot_url"].startswith("/waveforms/")
        # No Ollama configured -> no synthesized answer, but search still works.
        assert body["answer"] is None

        png = client.get(top["plot_url"].split("?")[0], params={"format": "png"})
        assert png.status_code == 200
        assert png.headers["content-type"] == "image/png"
        assert png.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_auto_ingest_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    """Without the auto_ingest_demo flag, the store starts empty (default.yaml)."""
    import api.main as api_main

    cfg = _demo_config(tmp_path)
    cfg["runtime"] = {"strict": False}  # no auto_ingest_demo

    monkeypatch.setattr(api_main, "load_config", lambda *a, **k: cfg)

    with TestClient(api_main.app) as client:
        store = api_main._state["store"]
        assert len(getattr(store, "_records", {})) == 0
        resp = client.post("/search", json={"query": "ARDS", "top_k": 3})
        assert resp.status_code == 200
        assert resp.json()["hits"] == []
