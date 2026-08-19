"""Weaviate-backed vector store for waveform epoch records.

Uses two named vectors per object:
  - ``waveform``: signal encoder output (optionally quantized)
  - ``text``: local text-encoder output for cross-modal search
"""

from __future__ import annotations

import json
from typing import Any, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np

from physiorag.embeddings.quantization import QuantizedEmbedding
from physiorag.storage.base import StoredRecord, VectorStore

DEFAULT_COLLECTION = "WaveformEpochV2"
WAVEFORM_VECTOR = "waveform"
TEXT_VECTOR = "text"

# Structured metadata promoted to first-class TEXT properties so BM25 / hybrid
# search can match on them (not just the free-text caption). Kept in sync with
# ``vent_captions.build_metadata`` and the ventilator demo.
METADATA_TEXT_PROPERTIES: tuple[str, ...] = (
    "asynchrony_type",
    "diagnosis",
    "finding",
    "vent_mode",
    "label",
)
# Properties BM25 scores against for a natural-language query.
BM25_QUERY_PROPERTIES: list[str] = ["text", *METADATA_TEXT_PROPERTIES]


def _as_vector(embedding: QuantizedEmbedding | np.ndarray) -> list[float]:
    if isinstance(embedding, QuantizedEmbedding):
        arr = np.asarray(embedding.data, dtype=np.float32).reshape(-1)
    else:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return arr.tolist()


def _stable_uuid(epoch_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"physiorag:{epoch_id}"))


class WeaviateVectorStore(VectorStore):
    """Local Weaviate adapter (v4 client) with named waveform/text vectors."""

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:8080",
        grpc_port: int = 50051,
        collection: str = DEFAULT_COLLECTION,
        waveform_dim: int = 128,
        text_dim: int = 384,
        skip_init_checks: bool = False,
        client: Any | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.grpc_port = grpc_port
        self.collection_name = collection
        self.waveform_dim = waveform_dim
        self.text_dim = text_dim
        self.skip_init_checks = skip_init_checks
        self._external_client = client is not None
        self._client = client if client is not None else self._connect()
        self.ensure_schema()

    def _connect(self) -> Any:
        from urllib.parse import urlparse

        import weaviate

        parsed = urlparse(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080
        return weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=self.grpc_port,
            skip_init_checks=self.skip_init_checks,
        )

    def close(self) -> None:
        if not self._external_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> "WeaviateVectorStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def reset_schema(self) -> None:
        """Drop and recreate the collection (use when vector dims change)."""
        if self._client.collections.exists(self.collection_name):
            self._client.collections.delete(self.collection_name)
        self.ensure_schema()

    def health(self) -> bool:
        """True when the Weaviate client can see (or create) this collection."""
        try:
            return bool(self._client.collections.exists(self.collection_name))
        except Exception:
            return False

    def ensure_schema(self) -> None:
        from weaviate.classes.config import Configure, DataType, Property

        if self._client.collections.exists(self.collection_name):
            self._verify_searchable_properties()
            return
        self._client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="epoch_id", data_type=DataType.TEXT),
                Property(name="record_id", data_type=DataType.TEXT),
                Property(name="modality", data_type=DataType.TEXT),
                Property(name="array_ref", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                *[
                    Property(name=name, data_type=DataType.TEXT)
                    for name in METADATA_TEXT_PROPERTIES
                ],
                Property(name="metadata_json", data_type=DataType.TEXT),
                Property(name="start_time_s", data_type=DataType.NUMBER),
            ],
            vector_config=[
                Configure.Vectors.self_provided(name=WAVEFORM_VECTOR),
                Configure.Vectors.self_provided(name=TEXT_VECTOR),
            ],
        )

    def _verify_searchable_properties(self) -> None:
        """Fail loudly if an existing collection predates the promoted BM25 props.

        Weaviate ignores property additions once a collection exists, so a
        collection created before Phase C would silently lack the searchable
        metadata fields and BM25 queries against them would return nothing (or
        error). Point the operator at ``--reset-collection`` instead.
        """
        try:
            config = self.collection.config.get()
            existing = {p.name for p in (config.properties or [])}
        except Exception:
            return  # introspection unavailable; do not block startup
        missing = [name for name in METADATA_TEXT_PROPERTIES if name not in existing]
        if missing:
            raise RuntimeError(
                f"Weaviate collection '{self.collection_name}' is missing searchable "
                f"metadata properties {missing}. It predates the Phase C BM25 "
                "metadata fields; re-ingest with --reset-collection to rebuild it."
            )

    @property
    def collection(self) -> Any:
        return self._client.collections.get(self.collection_name)

    def upsert(self, records: Sequence[StoredRecord]) -> int:
        if not records:
            return 0
        with self.collection.batch.dynamic() as batch:
            for record in records:
                wave = _as_vector(record.embedding)
                if self.waveform_dim > 0 and len(wave) != self.waveform_dim:
                    raise ValueError(
                        f"Waveform dim {len(wave)} != configured {self.waveform_dim}"
                    )
                vectors: dict[str, list[float]] = {WAVEFORM_VECTOR: wave}
                if record.text_embedding is not None:
                    text_vec = np.asarray(record.text_embedding, dtype=np.float32).reshape(-1)
                    if self.text_dim > 0 and len(text_vec) != self.text_dim:
                        raise ValueError(
                            f"Text dim {len(text_vec)} != configured {self.text_dim}"
                        )
                    vectors[TEXT_VECTOR] = text_vec.tolist()

                meta = record.metadata or {}
                props = {
                    "epoch_id": record.epoch_id,
                    "record_id": record.record_id,
                    "modality": record.modality,
                    "array_ref": record.array_ref,
                    "text": record.text or "",
                    "metadata_json": json.dumps(meta),
                    "start_time_s": float(meta.get("start_time_s", 0.0)),
                }
                for name in METADATA_TEXT_PROPERTIES:
                    value = meta.get(name)
                    props[name] = str(value) if value is not None else ""
                batch.add_object(
                    properties=props,
                    vector=vectors,
                    uuid=UUID(_stable_uuid(record.epoch_id)),
                )
        failed = list(self.collection.batch.failed_objects or [])
        if failed:
            raise RuntimeError(f"Weaviate batch upsert had failures: {failed[:3]}")
        return len(records)

    def get_by_epoch_id(self, epoch_id: str) -> StoredRecord | None:
        from weaviate.classes.query import Filter

        result = self.collection.query.fetch_objects(
            filters=Filter.by_property("epoch_id").equal(epoch_id),
            limit=1,
        )
        if not result.objects:
            return None
        return self._to_record(result.objects[0])

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[StoredRecord]:
        """Waveform-vector nearest-neighbor search."""
        from weaviate.classes.query import MetadataQuery

        vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1).tolist()
        result = self.collection.query.near_vector(
            near_vector=vector,
            target_vector=WAVEFORM_VECTOR,
            limit=top_k,
            filters=self._build_filter(filters),
            return_metadata=MetadataQuery(distance=True),
        )
        return [self._to_record(obj) for obj in result.objects]

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[StoredRecord]:
        """Hybrid BM25 + text-vector search when a query vector is provided,
        otherwise pure BM25. Both score over ``BM25_QUERY_PROPERTIES`` (the
        caption ``text`` plus the promoted metadata fields), not ``text`` alone.
        """
        from weaviate.classes.query import MetadataQuery

        weaviate_filter = self._build_filter(filters)
        if query_embedding is None:
            result = self.collection.query.bm25(
                query=query,
                query_properties=BM25_QUERY_PROPERTIES,
                limit=top_k,
                filters=weaviate_filter,
                return_metadata=MetadataQuery(score=True),
            )
        else:
            result = self.collection.query.hybrid(
                query=query,
                vector=np.asarray(query_embedding, dtype=np.float32).reshape(-1).tolist(),
                target_vector=TEXT_VECTOR,
                query_properties=BM25_QUERY_PROPERTIES,
                limit=top_k,
                filters=weaviate_filter,
                return_metadata=MetadataQuery(score=True, distance=True),
            )
        return [self._to_record(obj) for obj in result.objects]

    def _build_filter(self, filters: dict[str, Any] | None) -> Any:
        if not filters:
            return None
        from weaviate.classes.query import Filter

        modality = filters.get("modality")
        if modality:
            return Filter.by_property("modality").equal(modality)
        return None

    def _to_record(self, obj: Any) -> StoredRecord:
        props = obj.properties or {}
        raw_meta = props.get("metadata_json") or "{}"
        try:
            metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
        except json.JSONDecodeError:
            metadata = {"metadata_json": raw_meta}

        score = None
        if obj.metadata is not None:
            score = getattr(obj.metadata, "score", None)
            if score is None and getattr(obj.metadata, "distance", None) is not None:
                score = 1.0 - float(obj.metadata.distance)
        if score is not None:
            metadata["_score"] = float(score)

        vectors = obj.vector if isinstance(obj.vector, dict) else {}
        wave = np.asarray(vectors.get(WAVEFORM_VECTOR, []), dtype=np.float32)
        text_vec_raw = vectors.get(TEXT_VECTOR)
        text_vec = np.asarray(text_vec_raw, dtype=np.float32) if text_vec_raw else None

        return StoredRecord(
            record_id=str(props.get("record_id", "")),
            epoch_id=str(props.get("epoch_id", "")),
            modality=str(props.get("modality", "")),
            embedding=wave,
            array_ref=str(props.get("array_ref", "")),
            metadata=metadata,
            text=str(props.get("text") or "") or None,
            text_embedding=text_vec,
        )
