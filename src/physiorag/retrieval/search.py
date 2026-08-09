"""Retrieval orchestration over a local vector store."""

from __future__ import annotations

from typing import Any

import numpy as np

from physiorag.storage.base import StoredRecord, VectorStore


class Retriever:
    """Thin search facade; query encoding stays pluggable."""

    def __init__(self, store: VectorStore, *, text_encoder: Any | None = None) -> None:
        self._store = store
        self._text_encoder = text_encoder

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[StoredRecord]:
        return self._store.search(query_embedding, top_k=top_k, filters=filters)

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[StoredRecord]:
        """Text/NL search. Uses the text encoder for a cross-modal vector when
        available (hybrid), else falls back to keyword search.
        """
        query_embedding = None
        if self._text_encoder is not None:
            query_embedding = self._text_encoder.encode_one(query)
        if not hasattr(self._store, "search_text"):
            raise NotImplementedError("Store does not support text search")
        return self._store.search_text(  # type: ignore[attr-defined]
            query,
            top_k=top_k,
            filters=filters,
            query_embedding=query_embedding,
        )
