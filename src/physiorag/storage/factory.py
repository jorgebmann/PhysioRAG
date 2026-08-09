"""Construct vector stores from config."""

from __future__ import annotations

from typing import Any

from physiorag.storage.base import VectorStore
from physiorag.storage.memory_store import InMemoryVectorStore


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

        return WeaviateVectorStore(
            url=storage.get("weaviate_url", "http://127.0.0.1:8080"),
            grpc_port=int(storage.get("weaviate_grpc_port", 50051)),
            collection=storage.get("collection", "WaveformEpochV2"),
            waveform_dim=waveform_dim,
            text_dim=text_dim,
            skip_init_checks=bool(storage.get("skip_init_checks", False)),
        )
    raise ValueError(f"Unsupported storage backend: {backend}")
