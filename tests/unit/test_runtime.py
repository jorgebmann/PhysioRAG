"""Tests for strict-mode resolution precedence."""

from __future__ import annotations

import pytest

from physiorag.runtime import is_strict


def test_override_beats_env_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSIORAG_STRICT", "0")
    assert is_strict({"runtime": {"strict": False}}, override=True) is True
    assert is_strict({"runtime": {"strict": True}}, override=False) is False


def test_env_beats_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSIORAG_STRICT", "1")
    assert is_strict({"runtime": {"strict": False}}) is True
    monkeypatch.setenv("PHYSIORAG_STRICT", "off")
    assert is_strict({"runtime": {"strict": True}}) is False


def test_config_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHYSIORAG_STRICT", raising=False)
    assert is_strict({"runtime": {"strict": True}}) is True
    assert is_strict({"runtime": {"strict": False}}) is False
    assert is_strict({}) is False
    assert is_strict(None) is False
