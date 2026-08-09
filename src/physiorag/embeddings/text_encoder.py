"""Local text encoder for the cross-modal (text) named vector.

Wraps a small SentenceTransformer so natural-language queries and epoch
descriptions land in the same vector space. Runs fully offline once the model
is cached locally (see README for air-gapped setup).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


class TextEncoder:
    """Thin, lazily-loaded wrapper around a local SentenceTransformer."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        embedding_dim: int = DEFAULT_DIM,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._embedding_dim = embedding_dim
        self._model = None  # lazy

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        dim = self._model.get_sentence_embedding_dimension()
        if dim:
            self._embedding_dim = int(dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized embeddings with shape ``(len(texts), dim)``."""
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        self._ensure_model()
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
