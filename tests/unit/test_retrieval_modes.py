"""Retrieval-mode behavior: hybrid_text (Phase A) vs signal_aligned (Phase B)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from physiorag.retrieval.search import SIGNAL_ALIGNED, Retriever
from physiorag.storage.base import StoredRecord
from physiorag.storage.memory_store import InMemoryVectorStore


class _StubQueryEncoder:
    """Maps a couple of phrases to fixed unit vectors in a tiny signal space."""

    def __init__(self) -> None:
        self._table = {
            "atrial fibrillation": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "normal sinus rhythm": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }

    @property
    def embedding_dim(self) -> int:
        return 3

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode_one(t) for t in texts], axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self._table.get(text.lower(), np.zeros(3, dtype=np.float32))


def _rec(epoch_id: str, wave: list[float], text: str) -> StoredRecord:
    return StoredRecord(
        record_id=epoch_id,
        epoch_id=epoch_id,
        modality="ecg",
        embedding=np.asarray(wave, dtype=np.float32),
        array_ref=f"/tmp/{epoch_id}.npy",
        metadata={"start_time_s": 0.0},
        text=text,
    )


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert(
        [
            _rec("afib", [1.0, 0.0, 0.0], "12-lead ECG showing atrial fibrillation"),
            _rec("nsr", [0.0, 1.0, 0.0], "12-lead ECG with normal sinus rhythm"),
        ]
    )
    return store


class _RecordingEncoder(_StubQueryEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def encode_one(self, text: str) -> np.ndarray:
        self.seen.append(text)
        return super().encode_one(text)


def test_signal_aligned_queries_waveform_vectors() -> None:
    retriever = Retriever(_store(), text_encoder=_StubQueryEncoder(), mode=SIGNAL_ALIGNED)
    hits = retriever.search_text("atrial fibrillation", top_k=1)
    assert hits and hits[0].epoch_id == "afib"


def test_signal_aligned_rewrites_german_ptbxl_query() -> None:
    encoder = _RecordingEncoder()
    retriever = Retriever(_store(), text_encoder=encoder, mode=SIGNAL_ALIGNED)
    hits = retriever.search_text("vorhofflimmern", top_k=1)
    assert encoder.seen == ["atrial fibrillation"]
    assert hits and hits[0].epoch_id == "afib"


def test_signal_aligned_leaves_english_query_unchanged() -> None:
    encoder = _RecordingEncoder()
    retriever = Retriever(_store(), text_encoder=encoder, mode=SIGNAL_ALIGNED)
    retriever.search_text("Atrial Fibrillation", top_k=1)
    assert encoder.seen == ["Atrial Fibrillation"]


def test_hybrid_text_does_not_apply_ptbxl_glossary() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [_rec("de", [0.0, 0.0, 1.0], "sinusrhythmus normales ekg")]
    )
    retriever = Retriever(store)
    hits = retriever.search_text("sinusrhythmus", top_k=1)
    assert hits and hits[0].epoch_id == "de"


def test_signal_aligned_requires_query_encoder() -> None:
    retriever = Retriever(_store(), text_encoder=None, mode=SIGNAL_ALIGNED)
    with pytest.raises(NotImplementedError):
        retriever.search_text("atrial fibrillation", top_k=1)


def test_hybrid_text_is_default_mode() -> None:
    retriever = Retriever(_store())
    assert retriever.mode == "hybrid_text"
    # Without a text encoder it falls back to keyword search over text metadata.
    hits = retriever.search_text("sinus", top_k=1)
    assert hits and hits[0].epoch_id == "nsr"
