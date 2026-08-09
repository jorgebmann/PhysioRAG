"""Contract tests for TimeSeriesEncoder."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from physiorag.embeddings.base import TimeSeriesEncoder
from physiorag.ingestion.base import WaveformEpoch


class _DummyEncoder(TimeSeriesEncoder):
    @property
    def embedding_dim(self) -> int:
        return 8

    def encode(self, epochs: Sequence[WaveformEpoch]) -> np.ndarray:
        return np.zeros((len(epochs), self.embedding_dim), dtype=np.float32)


def test_encode_one_shape(synthetic_epoch: WaveformEpoch) -> None:
    encoder = _DummyEncoder()
    vector = encoder.encode_one(synthetic_epoch)
    assert vector.shape == (8,)


def test_encode_batch_shape(synthetic_epoch: WaveformEpoch) -> None:
    encoder = _DummyEncoder()
    batch = encoder.encode([synthetic_epoch, synthetic_epoch])
    assert batch.shape == (2, 8)
