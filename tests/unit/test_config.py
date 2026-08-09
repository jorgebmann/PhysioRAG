"""Config loading smoke tests."""

from __future__ import annotations

from physiorag.config import load_config


def test_load_default_config() -> None:
    cfg = load_config()
    assert cfg["ingestion"]["window_seconds"] == 10.0
    assert cfg["quantization"]["enabled"] is True
    assert cfg["synthesis"]["backend"] == "ollama"
