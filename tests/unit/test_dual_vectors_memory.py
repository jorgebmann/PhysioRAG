"""Dual (waveform + text) vector behavior in the in-memory store."""

from __future__ import annotations

import numpy as np

from physiorag.storage.base import StoredRecord
from physiorag.storage.memory_store import InMemoryVectorStore


def _rec(epoch_id: str, text_vec: list[float], text: str) -> StoredRecord:
    return StoredRecord(
        record_id=epoch_id,
        epoch_id=epoch_id,
        modality="ventilator",
        embedding=np.ones(4, dtype=np.float32),
        array_ref=f"/tmp/{epoch_id}.npy",
        metadata={"start_time_s": 0.0},
        text=text,
        text_embedding=np.asarray(text_vec, dtype=np.float32),
    )


def test_text_vector_search_prefers_nearest() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            _rec("a", [1.0, 0.0, 0.0], "ARDS asynchrony"),
            _rec("b", [0.0, 1.0, 0.0], "normal breaths"),
            _rec("c", [0.0, 0.0, 1.0], "air trapping"),
        ]
    )
    query = np.asarray([0.9, 0.1, 0.0], dtype=np.float32)
    hits = store.search_text("ards", top_k=2, query_embedding=query)
    assert hits[0].epoch_id == "a"
    assert hits[0].metadata["_score"] > hits[1].metadata["_score"]


def test_text_search_keyword_fallback_without_vector() -> None:
    store = InMemoryVectorStore()
    store.upsert([_rec("a", [1.0, 0.0, 0.0], "ARDS asynchrony pressure spike")])
    hits = store.search_text("pressure spike", top_k=1)
    assert hits and hits[0].epoch_id == "a"
