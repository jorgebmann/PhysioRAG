"""Filesystem persistence for plottable waveform arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ArrayStore:
    """Save/load epoch numpy arrays under a root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, epoch_id: str) -> Path:
        safe = epoch_id.replace("/", "_")
        return self.root / f"{safe}.npy"

    def save(self, epoch_id: str, signal: np.ndarray) -> str:
        path = self.path_for(epoch_id)
        np.save(path, np.asarray(signal))
        return str(path)

    def load(self, array_ref: str) -> np.ndarray:
        return np.load(array_ref)
