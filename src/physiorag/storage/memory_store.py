"""In-memory vector store for unit tests and offline dry-runs."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from physiorag.embeddings.quantization import QuantizedEmbedding
from physiorag.storage.base import StoredRecord, VectorStore
from physiorag.storage.weaviate_store import METADATA_TEXT_PROPERTIES

# Tiny EN/DE stopword set so common connectors do not inflate keyword overlap.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "for",
        "is", "are", "at", "by", "as", "that", "this",
        "der", "die", "das", "und", "oder", "mit", "im", "am", "auf", "einen",
        "eine", "ein", "den", "dem", "des", "dadurch", "gegen", "bei", "durch",
    }
)


def _as_vector(embedding: QuantizedEmbedding | np.ndarray) -> np.ndarray:
    if isinstance(embedding, QuantizedEmbedding):
        return np.asarray(embedding.data, dtype=np.float32).reshape(-1)
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _fold(text: str) -> str:
    t = (text or "").lower()
    return t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")


def _tokenize(text: str) -> set[str]:
    """Umlaut-folded alphanumeric tokens, minus a small stopword set.

    Folding mirrors the vent query glossary so an umlaut caption and a folded
    query token still match. Splitting on non-alphanumerics turns structured
    labels like ``double_triggering`` into ``{double, triggering}``.
    """
    return {
        tok
        for tok in re.split(r"[^a-z0-9]+", _fold(text))
        if tok and tok not in _STOPWORDS
    }


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

    def _candidates(self, filters: dict[str, Any] | None) -> list[StoredRecord]:
        return [r for r in self._records.values() if self._passes(r, filters)]

    @staticmethod
    def _keyword_haystack(record: StoredRecord) -> str:
        """Caption + the same promoted metadata fields Weaviate BM25 scores."""
        meta = record.metadata or {}
        parts = [record.text or ""]
        for name in METADATA_TEXT_PROPERTIES:
            value = meta.get(name)
            if value is not None:
                parts.append(str(value))
        return " ".join(parts)

    def _keyword_scores(self, query: str, records: list[StoredRecord]) -> dict[str, float]:
        query_tokens = _tokenize(query)
        scores: dict[str, float] = {}
        if not query_tokens:
            return scores
        for record in records:
            overlap = query_tokens & _tokenize(self._keyword_haystack(record))
            if overlap:
                scores[record.epoch_id] = float(len(overlap))
        return scores

    def _cosine_scores(
        self, query_embedding: np.ndarray, records: list[StoredRecord]
    ) -> dict[str, float]:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        scores: dict[str, float] = {}
        for record in records:
            vector = self._text_vectors.get(record.epoch_id)
            if vector is not None:
                scores[record.epoch_id] = _cosine(q, vector)
        return scores

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[StoredRecord]:
        """Hybrid dense + keyword search.

        When a query embedding and stored text vectors exist, dense (text-vector
        cosine) and keyword (token overlap over caption + metadata) rankings are
        fused with reciprocal rank fusion so an exact keyword hit can rescue a
        weak cosine match and vice versa. Without an embedding this degrades to
        keyword-only search.
        """
        records = self._candidates(filters)
        keyword = self._keyword_scores(query, records)

        if query_embedding is None or not self._text_vectors:
            scored = [(keyword[r.epoch_id], r) for r in records if r.epoch_id in keyword]
            scored.sort(key=lambda item: item[0], reverse=True)
            return [self._clone_with_score(rec, s) for s, rec in scored[:top_k]]

        cosine = self._cosine_scores(query_embedding, records)
        fused = _reciprocal_rank_fusion([cosine, keyword])
        ranked = sorted(records, key=lambda r: fused.get(r.epoch_id, 0.0), reverse=True)
        ranked = [r for r in ranked if fused.get(r.epoch_id, 0.0) > 0.0]
        return [self._clone_with_score(rec, fused[rec.epoch_id]) for rec in ranked[:top_k]]


def _reciprocal_rank_fusion(
    rankings: list[dict[str, float]], *, k0: int = 60
) -> dict[str, float]:
    """Combine several ``epoch_id -> score`` maps into one RRF score map.

    Each map is turned into a descending rank; an id's fused score is the sum of
    ``1 / (k0 + rank)`` across the maps in which it appears.
    """
    fused: dict[str, float] = {}
    for scores in rankings:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for rank, (epoch_id, _score) in enumerate(ordered):
            fused[epoch_id] = fused.get(epoch_id, 0.0) + 1.0 / (k0 + rank)
    return fused
