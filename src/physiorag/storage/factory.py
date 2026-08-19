"""Construct vector stores from config."""

from __future__ import annotations

from typing import Any

from physiorag.storage.base import VectorStore
from physiorag.storage.memory_store import InMemoryVectorStore

# Same names as ingest: mimic_demo lands in Demo; real WFDB stays on V2.
WFDB_COLLECTION = "WaveformEpochV2"
DEMO_COLLECTION = "WaveformEpochDemo"


def _collection_count(store: Any) -> int | None:
    try:
        return int(store.collection.aggregate.over_all(total_count=True).total_count)
    except Exception:
        return None


def _prefer_demo_if_wfdb_empty(store: Any, storage: dict[str, Any]) -> Any:
    """Point the API at WaveformEpochDemo when V2 exists but has no epochs.

    ``mimic_demo`` ingest writes Demo; ``default.yaml`` (and therefore uvicorn)
    still names V2. Opening Weaviate creates an empty V2, so every UI query
    returns zero hits even though the demo collection is populated.
    """
    if getattr(store, "collection_name", None) != WFDB_COLLECTION:
        return store
    n_v2 = _collection_count(store)
    if n_v2:
        return store
    client = getattr(store, "_client", None)
    if client is None or not client.collections.exists(DEMO_COLLECTION):
        return store
    demo_n = 0
    try:
        demo_n = int(
            client.collections.get(DEMO_COLLECTION)
            .aggregate.over_all(total_count=True)
            .total_count
        )
    except Exception:
        demo_n = 0
    if demo_n <= 0:
        return store

    from physiorag.storage.weaviate_store import WeaviateVectorStore

    url = storage.get("weaviate_url", "http://127.0.0.1:8080")
    grpc_port = int(storage.get("weaviate_grpc_port", 50051))
    skip = bool(storage.get("skip_init_checks", False))
    waveform_dim = int(store.waveform_dim)
    text_dim = int(store.text_dim)
    if hasattr(store, "close"):
        store.close()
    storage["collection"] = DEMO_COLLECTION
    print(
        f"[store] {WFDB_COLLECTION} is empty; using populated {DEMO_COLLECTION} "
        f"({demo_n} epochs) for search."
    )
    return WeaviateVectorStore(
        url=url,
        grpc_port=grpc_port,
        collection=DEMO_COLLECTION,
        waveform_dim=waveform_dim,
        text_dim=text_dim,
        skip_init_checks=skip,
    )


def build_vector_store(config: dict[str, Any]) -> VectorStore:
    storage = config.get("storage", {})
    backend = storage.get("backend", "memory")
    emb = config.get("embeddings", {})
    waveform_dim = int(emb.get("embedding_dim", 128))
    text_dim = int(emb.get("text_embedding_dim", 384))

    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "weaviate":
        from physiorag.storage.weaviate_store import WeaviateVectorStore

        store = WeaviateVectorStore(
            url=storage.get("weaviate_url", "http://127.0.0.1:8080"),
            grpc_port=int(storage.get("weaviate_grpc_port", 50051)),
            collection=storage.get("collection", WFDB_COLLECTION),
            waveform_dim=waveform_dim,
            text_dim=text_dim,
            skip_init_checks=bool(storage.get("skip_init_checks", False)),
        )
        return _prefer_demo_if_wfdb_empty(store, storage)
    raise ValueError(f"Unsupported storage backend: {backend}")
