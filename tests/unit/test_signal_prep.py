"""Rate conversion pads/truncates in time; it does not stretch duration."""

from __future__ import annotations

import numpy as np
import pytest

from physiorag.ingestion.signal_prep import (
    fill_non_finite,
    minmax_01,
    pad_or_truncate,
    prepare_window,
    resample_to_rate,
)


def test_resample_100hz_10s_to_500hz_keeps_duration() -> None:
    n = 1000  # 10 s @ 100 Hz
    sig = np.zeros((1, n), dtype=np.float32)
    sig[0, 0] = 1.0
    sig[0, -1] = 2.0
    out = resample_to_rate(sig, 100.0, 500.0)
    assert out.shape == (1, 5000)
    assert out[0, 0] == pytest.approx(1.0, abs=1e-5)
    # Last 100 Hz sample is at t=9.99 s; 500 Hz keeps it at the end of the window.
    assert out[0, -1] == pytest.approx(2.0, abs=1e-3)


def test_pad_short_record_does_not_stretch() -> None:
    n = 4000  # 8 s @ 500 Hz
    sig = np.zeros((2, n), dtype=np.float32)
    sig[0, -1] = 1.0
    out = pad_or_truncate(sig, n_samples=5000)
    assert out.shape == (2, 5000)
    assert out[0, 3999] == pytest.approx(1.0)
    assert np.allclose(out[0, 4000:], 0.0)


def test_truncate_long_record_keeps_the_start() -> None:
    sig = np.arange(6000, dtype=np.float32)[None, :]
    out = pad_or_truncate(sig, n_samples=5000)
    assert out.shape == (1, 5000)
    assert np.array_equal(out[0], np.arange(5000, dtype=np.float32))


def test_prepare_window_fill_resample_pad() -> None:
    sig = np.ones((2, 1000), dtype=np.float32)
    sig[0, 10] = np.nan
    out = prepare_window(
        sig, n_leads=12, n_samples=5000, native_fs=100.0, target_fs=500.0
    )
    assert out.shape == (12, 5000)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[2:], 0.0)


def test_fill_non_finite_all_nan_channel() -> None:
    sig = np.full((1, 8), np.nan, dtype=np.float32)
    out = fill_non_finite(sig)
    assert np.allclose(out, 0.0)


def test_minmax_01_scales_global_range() -> None:
    sig = np.asarray([[2.0, 4.0], [2.0, 2.0]], dtype=np.float32)
    out = minmax_01(sig)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(1.0)
