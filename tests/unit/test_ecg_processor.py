"""Offline test of the PTB-XL ECG processor using a locally written record."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")

from physiorag.ingestion.ecg_processor import STANDARD_LEADS, PtbxlEcgProcessor

# PTB-XL headers use uppercase AVR/AVL/AVF; the processor must normalize order.
_LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _write_record(root: Path, rel: str = "records500/00000/00001_hr") -> None:
    rec_path = root / rel
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    fs = 500
    seconds = 10
    n = fs * seconds
    t = np.arange(n) / fs
    # 12 distinct leads so we can verify ordering was preserved.
    sig = np.stack(
        [np.sin(2 * np.pi * (1.0 + k) * t) * (k + 1) for k in range(12)], axis=1
    ).astype(np.float64)
    wfdb.wrsamp(
        Path(rel).name,
        fs=fs,
        units=["mV"] * 12,
        sig_name=_LEAD_NAMES,
        p_signal=sig,
        fmt=["16"] * 12,
        write_dir=str(rec_path.parent),
    )


def _write_csvs(root: Path) -> None:
    (root / "ptbxl_database.csv").write_text(
        "ecg_id,patient_id,scp_codes,report,filename_lr,filename_hr\n"
        '1,101,"{\'NORM\': 100.0}",sinus rhythm normal ecg,'
        "records100/00000/00001_lr,records500/00000/00001_hr\n",
        encoding="utf-8",
    )
    (root / "scp_statements.csv").write_text(
        "code,description\nNORM,normal ECG\n",
        encoding="utf-8",
    )


def test_ecg_processor_csv_driven(tmp_path: Path) -> None:
    _write_record(tmp_path)
    _write_csvs(tmp_path)
    proc = PtbxlEcgProcessor(window_seconds=10.0, target_sample_rate_hz=500.0)

    epochs = list(proc.iter_epochs(tmp_path))
    assert len(epochs) == 1
    epoch = epochs[0]
    assert epoch.modality == "ecg"
    assert epoch.signal.shape == (12, 5000)
    assert epoch.sample_rate_hz == 500.0
    assert epoch.metadata["patient_id"] == "101"
    assert epoch.metadata["channels"] == list(STANDARD_LEADS)
    # Caption merges the report and the SCP description.
    text = epoch.metadata["text"].lower()
    assert "sinus rhythm" in text
    assert "normal ecg" in text


def test_ecg_processor_header_fallback(tmp_path: Path) -> None:
    _write_record(tmp_path)
    proc = PtbxlEcgProcessor(window_seconds=10.0, target_sample_rate_hz=500.0)
    epochs = list(proc.iter_epochs(tmp_path))
    assert len(epochs) == 1
    assert epochs[0].signal.shape == (12, 5000)


def test_ecg_processor_respects_max_records(tmp_path: Path) -> None:
    _write_record(tmp_path, "records500/00000/00001_hr")
    _write_record(tmp_path, "records500/00000/00002_hr")
    proc = PtbxlEcgProcessor(max_records=1)
    epochs = list(proc.iter_epochs(tmp_path))
    assert len(epochs) == 1


def test_process_epoch_pads_short_record_without_stretching() -> None:
    from physiorag.ingestion.base import WaveformEpoch

    proc = PtbxlEcgProcessor(window_seconds=10.0, target_sample_rate_hz=500.0)
    n = 4000  # 8 s @ 500 Hz
    sig = np.zeros((12, n), dtype=np.float32)
    sig[0, -1] = 1.0
    out = proc.process_epoch(
        WaveformEpoch(
            record_id="short",
            modality="ecg",
            start_time_s=0.0,
            sample_rate_hz=500.0,
            signal=sig,
        )
    )
    assert out.signal.shape == (12, 5000)
    assert out.signal[0, 3999] == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(out.signal[0, 4000:], 0.0)


def test_process_epoch_upsamples_rate_not_duration() -> None:
    from physiorag.ingestion.base import WaveformEpoch

    proc = PtbxlEcgProcessor(window_seconds=10.0, target_sample_rate_hz=500.0)
    sig = np.zeros((12, 1000), dtype=np.float32)  # 10 s @ 100 Hz
    sig[0, 0] = 1.0
    sig[0, -1] = 2.0
    out = proc.process_epoch(
        WaveformEpoch(
            record_id="lr",
            modality="ecg",
            start_time_s=0.0,
            sample_rate_hz=100.0,
            signal=sig,
        )
    )
    assert out.signal.shape == (12, 5000)
    assert out.sample_rate_hz == 500.0
    assert out.signal[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert out.signal[0, -1] == pytest.approx(2.0, abs=1e-3)
