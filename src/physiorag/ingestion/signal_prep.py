"""Shared waveform cleaning: NaN fill, rate conversion, pad/truncate.

Used by the PTB-XL ECG processor and the MERL encoder so both agree on how a
window becomes ``(n_leads, n_samples)`` at a target sample rate. Duration is
preserved: short records are zero-padded, long records are truncated — never
time-stretched to fit the window.
"""

from __future__ import annotations

import numpy as np


def as_channels_first(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D signal, got shape {arr.shape}")
    return arr


def fill_non_finite(signal: np.ndarray) -> np.ndarray:
    """Neighbor-fill NaN/Inf per channel; all-invalid channels become zeros."""
    out = np.array(as_channels_first(signal), dtype=np.float32, copy=True)
    for ch in range(out.shape[0]):
        row = out[ch]
        mask = ~np.isfinite(row)
        if mask.all():
            out[ch] = 0.0
        elif mask.any():
            idx = np.arange(row.size)
            row[mask] = np.interp(idx[mask], idx[~mask], row[~mask])
            out[ch] = row
    return out


def resample_to_rate(
    signal: np.ndarray, native_fs: float, target_fs: float
) -> np.ndarray:
    """Change sample rate without changing duration.

    ``native_fs <= 0`` or a missing/empty signal is returned as float32 unchanged.
    """
    arr = as_channels_first(signal)
    n_ch, n = arr.shape
    native_fs = float(native_fs)
    target_fs = float(target_fs)
    if n == 0 or native_fs <= 0 or target_fs <= 0:
        return arr.astype(np.float32, copy=False)
    if abs(native_fs - target_fs) < 1e-9:
        return arr.astype(np.float32, copy=False)
    n_new = max(1, int(round(n * target_fs / native_fs)))
    src_t = np.arange(n, dtype=np.float64) / native_fs
    dst_t = np.arange(n_new, dtype=np.float64) / target_fs
    out = np.empty((n_ch, n_new), dtype=np.float32)
    for ch in range(n_ch):
        out[ch] = np.interp(dst_t, src_t, arr[ch]).astype(np.float32)
    return out


def pad_or_truncate(
    signal: np.ndarray,
    *,
    n_leads: int | None = None,
    n_samples: int | None = None,
) -> np.ndarray:
    """Zero-pad or crop leads and/or time. Does not interpolate."""
    arr = as_channels_first(signal)
    if n_leads is not None:
        leads, length = arr.shape
        if leads < n_leads:
            arr = np.concatenate(
                [arr, np.zeros((n_leads - leads, length), dtype=np.float32)], axis=0
            )
        elif leads > n_leads:
            arr = arr[:n_leads]
    if n_samples is not None:
        leads, length = arr.shape
        if length < n_samples:
            arr = np.concatenate(
                [arr, np.zeros((leads, n_samples - length), dtype=np.float32)], axis=1
            )
        elif length > n_samples:
            arr = arr[:, :n_samples]
    return arr.astype(np.float32, copy=False)


def minmax_01(signal: np.ndarray) -> np.ndarray:
    """Scale an epoch to ``[0, 1]`` with MERL's global min-max (not per-lead)."""
    arr = as_channels_first(signal)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def prepare_window(
    signal: np.ndarray,
    *,
    n_leads: int,
    n_samples: int,
    native_fs: float | None = None,
    target_fs: float | None = None,
) -> np.ndarray:
    """Fill → canonical lead count → rate convert → pad/truncate in time."""
    arr = fill_non_finite(signal)
    arr = pad_or_truncate(arr, n_leads=n_leads)
    if native_fs is not None and target_fs is not None:
        arr = resample_to_rate(arr, native_fs, target_fs)
    return pad_or_truncate(arr, n_samples=n_samples)
