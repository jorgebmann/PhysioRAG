"""Offline test of the real WFDB processor using a locally written record."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")

from physiorag.ingestion.wfdb_processor import WfdbWaveformProcessor


def _write_record(root: Path) -> None:
    rec_dir = root / "waves" / "p100" / "p10039708" / "81739927"
    rec_dir.mkdir(parents=True, exist_ok=True)
    fs = 250
    seconds = 30
    n = fs * seconds
    t = np.arange(n) / fs
    paw = (10 + 5 * np.sin(2 * np.pi * 0.3 * t)).astype(np.float64)
    flow = (20 * np.sin(2 * np.pi * 0.3 * t)).astype(np.float64)
    sig = np.stack([paw, flow], axis=1)
    wfdb.wrsamp(
        "81739927",
        fs=fs,
        units=["mmHg", "L/min"],
        sig_name=["Paw", "Flow"],
        p_signal=sig,
        fmt=["16", "16"],
        write_dir=str(rec_dir),
    )


def test_wfdb_processor_streams_epochs(tmp_path: Path) -> None:
    _write_record(tmp_path)
    proc = WfdbWaveformProcessor(
        modality="ventilator",
        window_seconds=10.0,
        target_sample_rate_hz=125.0,
        max_epochs_per_record=2,
    )

    found = proc.find_records(tmp_path)
    assert len(found) == 1

    epochs = list(proc.iter_epochs(tmp_path))
    assert len(epochs) == 2  # 30s / 10s = 3 windows, capped at 2
    first = epochs[0]
    assert first.signal.shape == (2, 1250)  # 2 channels, 10s @ 125Hz
    assert first.sample_rate_hz == 125.0
    assert "Paw" in (first.metadata.get("text") or "")
    # z-normalized -> near-zero mean per channel
    assert np.allclose(first.signal.mean(axis=1), 0.0, atol=1e-4)
