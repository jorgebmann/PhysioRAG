"""Abstract interface for waveform loading, epoching, and cleaning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Iterator

import numpy as np


@dataclass(slots=True)
class WaveformEpoch:
    """A fixed-length window of physiological signal plus linked metadata."""

    record_id: str
    modality: str
    start_time_s: float
    sample_rate_hz: float
    signal: np.ndarray  # shape: (n_channels, n_samples) or (n_samples,)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal)


class WaveformProcessor(ABC):
    """Load WFDB (or similar) records and yield memory-safe epochs.

    Implementations must stream or chunk large records — never load an entire
    MIMIC-IV corpus into RAM.
    """

    @abstractmethod
    def iter_epochs(self, source: str | PathLike[str]) -> Iterator[WaveformEpoch]:
        """Yield cleaned, fixed-window epochs from ``source``."""

    @abstractmethod
    def process_epoch(self, epoch: WaveformEpoch) -> WaveformEpoch:
        """Apply resampling, normalization, and artifact handling to one epoch."""
