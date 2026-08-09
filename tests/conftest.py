"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from physiorag.ingestion.base import WaveformEpoch


@pytest.fixture
def synthetic_epoch() -> WaveformEpoch:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(1, 1250)).astype(np.float32)  # 10 s @ 125 Hz
    return WaveformEpoch(
        record_id="demo-001",
        modality="ventilator",
        start_time_s=0.0,
        sample_rate_hz=125.0,
        signal=signal,
        metadata={"diagnosis": "ARDS"},
    )
