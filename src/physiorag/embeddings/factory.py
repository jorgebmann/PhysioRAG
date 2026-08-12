"""Registry/factory for signal (waveform) encoders.

Keeps encoder selection config-driven and pluggable: baselines (1D-CNN today)
and future modality-specific / reused dual-encoders (Phase B+) register here and
are swapped via ``embeddings.encoder`` without touching ingest, storage, or the
API. Unknown names fail loudly with the list of available encoders.
"""

from __future__ import annotations

from typing import Any, Callable

from physiorag.embeddings.base import TimeSeriesEncoder

EncoderBuilder = Callable[[dict[str, Any]], TimeSeriesEncoder]

_ENCODER_BUILDERS: dict[str, EncoderBuilder] = {}


def register_encoder(name: str, builder: EncoderBuilder) -> None:
    """Register a builder under ``name`` (idempotent overwrite allowed)."""
    _ENCODER_BUILDERS[name] = builder


def available_encoders() -> list[str]:
    return sorted(_ENCODER_BUILDERS)


def build_encoder(config: dict[str, Any]) -> TimeSeriesEncoder:
    """Construct the signal encoder named by ``embeddings.encoder``."""
    emb = config.get("embeddings", {})
    name = str(emb.get("encoder", "baseline_cnn"))
    try:
        builder = _ENCODER_BUILDERS[name]
    except KeyError:
        available = ", ".join(available_encoders()) or "(none)"
        raise ValueError(
            f"Unsupported encoder '{name}'. Available encoders: {available}"
        ) from None
    return builder(config)


def _build_baseline_cnn(config: dict[str, Any]) -> TimeSeriesEncoder:
    from physiorag.embeddings.baseline_cnn import BaselineCNNEncoder

    emb = config.get("embeddings", {})
    return BaselineCNNEncoder(
        embedding_dim=int(emb.get("embedding_dim", 128)),
        device=str(emb.get("device", "cpu")),
    )


register_encoder("baseline_cnn", _build_baseline_cnn)
