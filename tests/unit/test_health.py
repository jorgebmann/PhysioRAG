"""/health surfaces store/collection/strict and only reports degraded when an
enabled component is missing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


def _memory_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "storage": {"backend": "memory", "array_dir": str(tmp_path / "arrays"), "collection": "DemoX"},
        "embeddings": {"text_encoder_enabled": False},
        "synthesis": {"enabled": False},
        "quantization": {"enabled": False},
    }


def test_health_reports_fields_and_ok_when_disabled(monkeypatch, tmp_path: Path) -> None:
    import api.main as api_main

    monkeypatch.setattr(api_main, "load_config", lambda *a, **k: _memory_config(tmp_path))
    with TestClient(api_main.app) as client:
        body = client.get("/health").json()
        # Config-disabled components are not "degraded".
        assert body["status"] == "ok"
        assert body["text_encoder"] is False
        assert body["llm"] is False
        assert body["store"] == "memory"
        assert body["store_ok"] is True
        assert body["collection"] == "DemoX"
        assert "quant_available" in body
        assert body["strict"] is False
