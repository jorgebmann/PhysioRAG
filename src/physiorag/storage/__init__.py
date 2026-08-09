"""Local vector and array persistence."""

from physiorag.storage.array_store import ArrayStore
from physiorag.storage.base import StoredRecord, VectorStore
from physiorag.storage.factory import build_vector_store
from physiorag.storage.memory_store import InMemoryVectorStore

__all__ = [
    "ArrayStore",
    "StoredRecord",
    "VectorStore",
    "InMemoryVectorStore",
    "build_vector_store",
]
