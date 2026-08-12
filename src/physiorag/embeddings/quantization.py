"""Vector compression via pyturboquant (core embedding-pipeline step)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Codec name produced by a real TurboQuant implementation (not the float16 scaffold).
REAL_QUANT_CODECS = frozenset({"pyturboquant"})


def is_real_quant_codec(codec: str | None) -> bool:
    return codec in REAL_QUANT_CODECS


def has_pyturboquant_core() -> bool:
    """True when ``pyturboquant.core`` (MSE quantizer) is importable."""
    try:
        from pyturboquant.core import MSEQuantizer  # noqa: F401
    except ImportError:
        try:
            from pyturboquant.core import mse_quantize  # noqa: F401
        except ImportError:
            return False
    return True


@dataclass(slots=True)
class QuantizedEmbedding:
    """Compressed embedding plus codec metadata needed for later search/decode."""

    data: bytes | np.ndarray
    original_shape: tuple[int, ...]
    bits: int
    codec: str = "pyturboquant"
    metadata: dict[str, Any] | None = None


def _fidelity_stats(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Mean cosine similarity and MSE between float32 original and reconstruction."""
    orig = original.astype(np.float32, copy=False)
    recon = reconstructed.astype(np.float32, copy=False)
    # Per-row cosine, then mean (avoids one giant vector dominating).
    cosines: list[float] = []
    for i in range(orig.shape[0]):
        a, b = orig[i], recon[i]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        cosines.append(float(np.dot(a, b) / denom))
    mse = float(np.mean((orig - recon) ** 2))
    return {
        "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "mse": mse,
    }


def _float32_nbytes(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * 4


def _quantize_pyturboquant_core(
    embeddings: np.ndarray, *, bits: int
) -> QuantizedEmbedding | None:
    """Use ``pyturboquant.core`` (class or functional API) when installed via ``.[quant]``."""
    try:
        import torch
        from pyturboquant.core import MSEQuantizer  # type: ignore[import-untyped]
    except ImportError:
        try:
            import torch
            from pyturboquant.core import mse_dequantize, mse_quantize  # type: ignore[import-untyped]
        except ImportError:
            return None
        else:
            x = torch.from_numpy(embeddings.astype(np.float32, copy=False))
            qt = mse_quantize(x, bits=bits, seed=42)
            reconstructed = mse_dequantize(qt).detach().cpu().numpy().astype(np.float32, copy=False)
            compressed_bytes = _packed_nbytes(qt)
    else:
        dim = int(embeddings.shape[1])
        quantizer = MSEQuantizer(dim=dim, bits=bits, seed=42)
        x = torch.from_numpy(embeddings.astype(np.float32, copy=False))
        qt = quantizer.quantize(x)
        reconstructed = quantizer.dequantize(qt).detach().cpu().numpy().astype(np.float32, copy=False)
        compressed_bytes = _packed_nbytes(qt)

    original_bytes = _float32_nbytes(tuple(embeddings.shape))
    fidelity = _fidelity_stats(embeddings, reconstructed)
    ratio = (original_bytes / compressed_bytes) if compressed_bytes else 0.0
    return QuantizedEmbedding(
        data=reconstructed,
        original_shape=tuple(embeddings.shape),
        bits=bits,
        codec="pyturboquant",
        metadata={
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": round(ratio, 3),
            "mean_cosine": round(fidelity["mean_cosine"], 6),
            "mse": fidelity["mse"],
            # Weaviate ANN still indexes the dequantized float32 reconstruction.
            "stored_as": "float32_reconstruction",
        },
    )


def _packed_nbytes(qt: Any) -> int:
    """Byte size of the quantized payload (indices + norms when present)."""
    total = 0
    packed = getattr(qt, "packed_indices", None)
    if packed is not None and hasattr(packed, "numel"):
        total += int(packed.numel() * packed.element_size())
    norms = getattr(qt, "norms", None)
    if norms is not None and hasattr(norms, "numel"):
        total += int(norms.numel() * norms.element_size())
    if total == 0:
        # Fallback theoretical size: bits per dim + float32 norm per row.
        dim = int(getattr(qt, "dim", 0) or 0)
        bits = int(getattr(qt, "bits", 8) or 8)
        n = int(getattr(packed, "shape", [0])[0]) if packed is not None else 0
        # packed_indices is usually flat; prefer norms batch size when available.
        if norms is not None and hasattr(norms, "shape") and len(norms.shape) >= 1:
            n = int(norms.shape[0])
        total = max(1, (n * dim * bits + 7) // 8 + n * 4)
    return total


def quantize_embeddings(
    embeddings: np.ndarray,
    *,
    bits: int = 8,
) -> QuantizedEmbedding:
    """Quantize a batch of dense embeddings.

    Thin wrapper so the rest of the pipeline always goes through one compression
    entrypoint. Resolution order:

    1. ``pyturboquant.core`` (MSEQuantizer / mse_quantize) — required for Phase A
    2. ``float16_stub`` when ``pyturboquant`` with ``.core`` is not installed

    Note: Weaviate search still stores the **dequantized** float32 reconstruction
    (ANN needs dense vectors). Codec metadata reports theoretical compressed
    size / fidelity so "TurboQuant on" is measurable at ingest time.
    """
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    result = _quantize_pyturboquant_core(embeddings, bits=bits)
    if result is not None:
        return result

    compressed = embeddings.astype(np.float16, copy=False)
    original_bytes = _float32_nbytes(tuple(embeddings.shape))
    compressed_bytes = int(compressed.nbytes)
    fidelity = _fidelity_stats(embeddings, compressed.astype(np.float32))
    return QuantizedEmbedding(
        data=compressed,
        original_shape=tuple(embeddings.shape),
        bits=16,
        codec="float16_stub",
        metadata={
            "warning": (
                "pyturboquant.core not available; using float16 stub. "
                'Install with: pip install ".[quant]" (pyturboquant>=0.1.1)'
            ),
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": round(original_bytes / compressed_bytes, 3)
            if compressed_bytes
            else 0.0,
            "mean_cosine": round(fidelity["mean_cosine"], 6),
            "mse": fidelity["mse"],
            "stored_as": "float16_stub",
        },
    )
