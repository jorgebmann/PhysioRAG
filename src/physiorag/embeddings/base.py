"""Abstract interface for physiological time-series encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from physiorag.ingestion.base import WaveformEpoch


class TimeSeriesEncoder(ABC):
    """Encode waveform epochs into dense semantic vectors.

    Keep this interface stable so baselines (1D-CNN) and foundation models
    (Chronos, modality-specific TSFMs) can be swapped without touching storage
    or retrieval.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of vectors produced by :meth:`encode`."""

    @abstractmethod
    def encode(self, epochs: Sequence[WaveformEpoch]) -> np.ndarray:
        """Return embeddings with shape ``(len(epochs), embedding_dim)``."""

    def encode_one(self, epoch: WaveformEpoch) -> np.ndarray:
        """Encode a single epoch; returns shape ``(embedding_dim,)``."""
        return self.encode([epoch])[0]
