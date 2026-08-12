"""Storage abstractions for multi-vector waveform records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from physiorag.embeddings.quantization import QuantizedEmbedding


@dataclass(slots=True)
class StoredRecord:
    """One retrievable unit: waveform embedding + optional text embedding +
    raw array ref + clinical metadata.
    """

    record_id: str
    epoch_id: str
    modality: str
    embedding: QuantizedEmbedding | np.ndarray  # waveform (signal) vector
    array_ref: str  # path or object key to the plottable numpy array
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None
    text_embedding: np.ndarray | None = None  # aligned text vector for cross-modal search


class VectorStore(ABC):
    """Local vector database adapter (Weaviate, pgvector, Milvus, or in-memory)."""

    @abstractmethod
    def upsert(self, records: Sequence[StoredRecord]) -> int:
        """Insert or update records; return number written."""

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[StoredRecord]:
        """Nearest-neighbor search over stored waveform vectors."""

    def get_by_epoch_id(self, epoch_id: str) -> StoredRecord | None:
        """Fetch a single record by epoch id (for plotting). Default: unsupported."""
        return None

    def health(self) -> bool:
        """Return True when the backend is reachable / usable."""
        return True
