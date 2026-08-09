"""API tests using an in-memory store (offline, no Weaviate/Ollama/model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient


def _memory_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "storage": {"backend": "memory", "array_dir": str(tmp_path / "arrays")},
        "embeddings": {"text_encoder_enabled": False},
        "synthesis": {"enabled": False},
    }


def _seed(store: Any, arrays: Any) -> None:
    from physiorag.storage.base import StoredRecord

    signal = np.random.default_rng(0).normal(size=(2, 1250)).astype(np.float32)
    ref = arrays.save("demo-ards-001_0", signal)
    store.upsert(
        [
            StoredRecord(
                record_id="demo-ards-001",
                epoch_id="demo-ards-001_0",
                modality="ventilator",
                embedding=np.ones(8, dtype=np.float32),
                array_ref=ref,
                metadata={"start_time_s": 0.0, "sample_rate_hz": 125.0, "channels": ["Paw", "Flow"]},
                text="ARDS pressure spike asynchrony",
            )
        ]
    )


def test_search_and_plot(monkeypatch, tmp_path: Path) -> None:
    import api.main as api_main

    monkeypatch.setattr(api_main, "load_config", lambda *a, **k: _memory_config(tmp_path))

    with TestClient(api_main.app) as client:
        _seed(api_main._state["store"], api_main._state["arrays"])

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        resp = client.post("/search", json={"query": "ARDS pressure spike", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["hits"], "expected at least one hit"
        top = body["hits"][0]
        assert top["epoch_id"] == "demo-ards-001_0"
        assert top["plot_url"] == "/waveforms/demo-ards-001_0?format=png"
        assert body["answer"] is None  # synthesis disabled

        j = client.get("/waveforms/demo-ards-001_0", params={"format": "json"})
        assert j.status_code == 200
        jd = j.json()
        assert jd["channels"] == ["Paw", "Flow"]
        assert len(jd["signal"]) == 2 and len(jd["signal"][0]) == 1250

        png = client.get("/waveforms/demo-ards-001_0", params={"format": "png"})
        assert png.status_code == 200
        assert png.headers["content-type"] == "image/png"
        assert png.content[:8] == b"\x89PNG\r\n\x1a\n"

        missing = client.get("/waveforms/does-not-exist")
        assert missing.status_code == 404
