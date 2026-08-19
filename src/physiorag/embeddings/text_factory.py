"""Registry/factory for query (text) encoders.

Keeps the natural-language query tower config-driven and pluggable so the
Phase A MiniLM path and Phase B matched dual-encoder text towers (e.g. MERL's
Med-CPT projection) can be swapped via ``embeddings.text_encoder`` without
touching ingest, storage, or the API.

A query encoder is any object exposing the small ``TextEncoder`` protocol:
``embedding_dim`` (int property), ``encode(texts) -> np.ndarray`` and
``encode_one(text) -> np.ndarray``. Both :class:`TextEncoder` (MiniLM) and the
MERL text tower implement it.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class QueryEncoder(Protocol):
    """Minimal contract shared by MiniLM and matched dual-encoder text towers."""

    @property
    def embedding_dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_one(self, text: str) -> np.ndarray: ...


def build_query_encoder(config: dict[str, Any]) -> QueryEncoder:
    """Construct the query text encoder named by ``embeddings.text_encoder``.

    ``merl`` selects the MERL Med-CPT projection tower (must land in the same
    space as the MERL ECG signal tower). Anything else is treated as a local
    SentenceTransformer model name (default MiniLM).
    """
    emb = config.get("embeddings", {})
    name = str(emb.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"))

    if name == "merl":
        from physiorag.embeddings.merl import build_merl_text_encoder

        return build_merl_text_encoder(config)

    from physiorag.embeddings.text_encoder import TextEncoder

    return TextEncoder(
        model_name=name,
        device=str(emb.get("device", "cpu")),
        embedding_dim=int(emb.get("text_embedding_dim", 384)),
    )
