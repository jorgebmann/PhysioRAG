"""Lightweight 1D-CNN baseline encoder for waveform epochs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from physiorag.embeddings.base import TimeSeriesEncoder
from physiorag.ingestion.base import WaveformEpoch


class _Tiny1DCNN(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.proj(h)


class BaselineCNNEncoder(TimeSeriesEncoder):
    """Deterministically initialized 1D-CNN (no pretrained weights).

    Useful as a drop-in baseline before Chronos / modality-specific TSFMs.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 768,
        in_channels: int = 2,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self._embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.model = _Tiny1DCNN(in_channels=in_channels, embedding_dim=embedding_dim).to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, epochs: Sequence[WaveformEpoch]) -> np.ndarray:
        if not epochs:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        batch = np.stack([self._prepare(epoch.signal) for epoch in epochs], axis=0)
        x = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            vectors = self.model(x).cpu().numpy().astype(np.float32)
        # L2-normalize for cosine-friendly retrieval.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        return vectors / norms

    def _prepare(self, signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        channels, length = arr.shape
        if channels < self.in_channels:
            pad = np.zeros((self.in_channels - channels, length), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif channels > self.in_channels:
            arr = arr[: self.in_channels]
        return arr
