"""In-memory vector store for unit tests and offline dry-runs."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from physiorag.embeddings.quantization import QuantizedEmbedding
from physiorag.storage.base import StoredRecord, VectorStore


def _as_vector(embedding: QuantizedEmbedding | np.ndarray) -> np.ndarray:
    if isinstance(embedding, QuantizedEmbedding):
        return np.asarray(embedding.data, dtype=np.float32).reshape(-1)
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class InMemoryVectorStore(VectorStore):
    """Cosine-similarity store holding both waveform and text vectors."""

    def __init__(self) -> None:
        self._records: dict[str, StoredRecord] = {}
        self._wave_vectors: dict[str, np.ndarray] = {}
        self._text_vectors: dict[str, np.ndarray] = {}

    def upsert(self, records: Sequence[StoredRecord]) -> int:
        for record in records:
            self._records[record.epoch_id] = record
            self._wave_vectors[record.epoch_id] = _as_vector(record.embedding)
            if record.text_embedding is not None:
                self._text_vectors[record.epoch_id] = np.asarray(
                    record.text_embedding, dtype=np.float32
                ).reshape(-1)
        return len(records)

    def get_by_epoch_id(self, epoch_id: str) -> StoredRecord | None:
        return self._records.get(epoch_id)

    def _clone_with_score(self, record: StoredRecord, score: float) -> StoredRecord:
        meta = dict(record.metadata)
        meta["_score"] = score
        return StoredRecord(
            record_id=record.record_id,
            epoch_id=record.epoch_id,
            modality=record.modality,
            embedding=record.embedding,
            array_ref=record.array_ref,
            metadata=meta,
            text=record.text,
            text_embedding=record.text_embedding,
        )

    def _passes(self, record: StoredRecord, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        modality = filters.get("modality")
        return not (modality and record.modality != modality)

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[StoredRecord]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        scored: list[tuple[float, StoredRecord]] = []
        for epoch_id, vector in self._wave_vectors.items():
            record = self._records[epoch_id]
            if not self._passes(record, filters):
                continue
            scored.append((_cosine(query, vector), record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._clone_with_score(rec, s) for s, rec in scored[:top_k]]

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[StoredRecord]:
        # Prefer text-vector cosine when a query vector and stored text vectors exist.
        if query_embedding is not None and self._text_vectors:
            q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            scored: list[tuple[float, StoredRecord]] = []
            for epoch_id, vector in self._text_vectors.items():
                record = self._records[epoch_id]
                if not self._passes(record, filters):
                    continue
                scored.append((_cosine(q, vector), record))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [self._clone_with_score(rec, s) for s, rec in scored[:top_k]]

        # Fallback: keyword overlap over text + metadata.
        tokens = {t.lower() for t in query.split() if t}
        scored = []
        for record in self._records.values():
            if not self._passes(record, filters):
                continue
            hay = f"{record.text or ''} {record.metadata}".lower()
            score = float(sum(1 for tok in tokens if tok in hay))
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._clone_with_score(rec, s) for s, rec in scored[:top_k]]
