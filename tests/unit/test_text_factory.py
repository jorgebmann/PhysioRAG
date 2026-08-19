"""Query (text) encoder factory selection."""

from __future__ import annotations

from typing import Any

import pytest

from physiorag.embeddings.text_factory import build_query_encoder


def test_defaults_to_minilm_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeTextEncoder:
        def __init__(self, *, model_name: str, device: str, embedding_dim: int) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["embedding_dim"] = embedding_dim

    import physiorag.embeddings.text_encoder as te

    monkeypatch.setattr(te, "TextEncoder", _FakeTextEncoder)

    build_query_encoder({"embeddings": {"device": "cpu", "text_embedding_dim": 384}})
    assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["embedding_dim"] == 384


def test_merl_selects_merl_text_tower(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    # The factory imports build_merl_text_encoder lazily from the merl module.
    from physiorag.embeddings import merl

    monkeypatch.setattr(merl, "build_merl_text_encoder", lambda config: sentinel)

    result = build_query_encoder({"embeddings": {"text_encoder": "merl"}})
    assert result is sentinel
