"""Runtime-mode helpers (strict vs. soft-degrade).

Strict mode makes the pipeline fail loudly when a required component (text
encoder, vector store, Ollama synthesis, or real ``pyturboquant`` compression)
is unavailable, instead of silently degrading. This is what the Phase A demo /
air-gap acceptance runs use so a broken offline setup cannot pass unnoticed.

Precedence (highest first):
  1. explicit ``override`` argument (e.g. an ingest ``--strict`` flag)
  2. the ``PHYSIORAG_STRICT`` environment variable
  3. ``runtime.strict`` in the loaded config
  4. ``False``
"""

from __future__ import annotations

import os
from typing import Any

STRICT_ENV = "PHYSIORAG_STRICT"
_TRUTHY = {"1", "true", "yes", "on"}


def is_strict(config: dict[str, Any] | None = None, *, override: bool | None = None) -> bool:
    """Resolve the effective strict-mode flag from override, env, then config."""
    if override is not None:
        return override
    env = os.environ.get(STRICT_ENV)
    if env is not None:
        return env.strip().lower() in _TRUTHY
    if config:
        return bool(config.get("runtime", {}).get("strict", False))
    return False
