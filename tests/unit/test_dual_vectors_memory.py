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


def test_keyword_matches_promoted_metadata_field() -> None:
    store = InMemoryVectorStore()
    rec = StoredRecord(
        record_id="a",
        epoch_id="a",
        modality="ventilator",
        embedding=np.ones(4, dtype=np.float32),
        array_ref="/tmp/a.npy",
        metadata={"asynchrony_type": "double_triggering", "vent_mode": "PSV"},
        text="ventilator pressure flow window",
    )
    store.upsert([rec])
    # "double triggering" lives only in the promoted asynchrony_type field.
    hits = store.search_text("double triggering", top_k=1)
    assert hits and hits[0].epoch_id == "a"


def test_keyword_ignores_metadata_keys_and_stopwords() -> None:
    store = InMemoryVectorStore()
    store.upsert([_rec("a", [1.0, 0.0, 0.0], "controlled ventilation")])
    # "with" is a stopword; "type" was only matchable via the old str(metadata)
    # substring hack (asynchrony_type key) and must no longer score.
    assert store.search_text("with type", top_k=1) == []


def test_keyword_folds_umlauts() -> None:
    store = InMemoryVectorStore()
    store.upsert([_rec("a", [1.0, 0.0, 0.0], "Beatmungsgerät Druckanstieg")])
    # A folded query token still matches an umlaut caption.
    hits = store.search_text("beatmungsgeraet", top_k=1)
    assert hits and hits[0].epoch_id == "a"


def test_rrf_lets_keyword_rescue_a_weak_cosine_match() -> None:
    """A record whose cosine is mediocre but whose caption keyword-matches the
    query should be pulled up by reciprocal rank fusion."""
    store = InMemoryVectorStore()
    store.upsert(
        [
            _rec("a", [0.0, 1.0, 0.0], "double triggering pressure spike"),
            _rec("b", [1.0, 0.0, 0.0], "air trapping auto peep"),
            _rec("c", [0.9, 0.1, 0.0], "normal controlled ventilation"),
        ]
    )
    # Query vector is closest to "b"/"c" by cosine, but the words only match "a".
    query_embedding = np.asarray([0.95, 0.05, 0.0], dtype=np.float32)
    hits = store.search_text("double triggering", top_k=1, query_embedding=query_embedding)
    assert hits and hits[0].epoch_id == "a"
