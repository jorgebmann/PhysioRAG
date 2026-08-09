"""Vector compression via pyturboquant (core embedding-pipeline step)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class QuantizedEmbedding:
    """Compressed embedding plus codec metadata needed for later search/decode."""

    data: bytes | np.ndarray
    original_shape: tuple[int, ...]
    bits: int
    codec: str = "pyturboquant"
    metadata: dict[str, Any] | None = None


def quantize_embeddings(
    embeddings: np.ndarray,
    *,
    bits: int = 8,
) -> QuantizedEmbedding:
    """Quantize a batch of dense embeddings.

    This is intentionally a thin wrapper so the rest of the pipeline always
    goes through a single compression entrypoint. When ``pyturboquant`` is
    available it is preferred; otherwise a float16 cast is used as a stub so
    local scaffolding and tests can run without the native package.
    """
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    try:
        import pyturboquant  # type: ignore[import-untyped]
    except ImportError:
        compressed = embeddings.astype(np.float16, copy=False)
        return QuantizedEmbedding(
            data=compressed,
            original_shape=tuple(embeddings.shape),
            bits=16,
            codec="float16_stub",
            metadata={"warning": "pyturboquant not installed; using float16 stub"},
        )

    # pyturboquant API may evolve; keep call site centralized here.
    if hasattr(pyturboquant, "quantize"):
        payload = pyturboquant.quantize(embeddings, bits=bits)
        return QuantizedEmbedding(
            data=payload,
            original_shape=tuple(embeddings.shape),
            bits=bits,
            codec="pyturboquant",
        )

    # Installed package without a public API yet — keep the pipeline runnable.
    compressed = embeddings.astype(np.float16, copy=False)
    return QuantizedEmbedding(
        data=compressed,
        original_shape=tuple(embeddings.shape),
        bits=16,
        codec="float16_stub",
        metadata={
            "warning": (
                "pyturboquant is installed but exposes no `quantize` function; "
                "using float16 stub"
            )
        },
    )
