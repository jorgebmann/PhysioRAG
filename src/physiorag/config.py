"""Load YAML configuration for PhysioRAG pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
CONFIG_ENV = "PHYSIORAG_CONFIG"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file.

    Precedence: explicit ``path`` > ``PHYSIORAG_CONFIG`` env var (used to point
    the API at e.g. ``configs/ecg_merl.yaml``) > ``configs/default.yaml``.
    """
    if path:
        config_path = Path(path)
    elif os.environ.get(CONFIG_ENV):
        config_path = Path(os.environ[CONFIG_ENV])
    else:
        config_path = DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a mapping")
    return data
