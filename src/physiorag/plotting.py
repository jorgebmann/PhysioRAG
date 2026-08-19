"""Plottable evidence: PNG waveforms for search hits.

Ventilator / generic traces stay stacked. 12-lead ECG uses a conventional
3×4 grid (I/II/III, aVR/aVL/aVF, V1–V3, V4–V6) so the demo PNG fits a screen
instead of 12 full-width strips.
"""

from __future__ import annotations

import io
import struct
from typing import Sequence

import numpy as np

# Conventional 12-lead page: 3 rows × 4 columns.
ECG_LEAD_GRID: tuple[tuple[str, ...], ...] = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)
_LEAD_ALIASES: dict[str, str] = {
    "I": "I",
    "II": "II",
    "III": "III",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}


def png_wh(png: bytes) -> tuple[int, int]:
    """Width, height from a PNG IHDR (no extra decoder)."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    width, height = struct.unpack(">II", png[16:24])
    return int(width), int(height)


def normalize_lead_name(name: str) -> str:
    key = (name or "").strip().replace(" ", "").upper()
    return _LEAD_ALIASES.get(key, (name or "").strip())


def twelve_lead_index(channels: Sequence[str]) -> dict[str, int] | None:
    """Map canonical lead names to row indices, or None if not a 12-lead set."""
    by_name = {normalize_lead_name(c): i for i, c in enumerate(channels)}
    needed = [lead for row in ECG_LEAD_GRID for lead in row]
    if all(n in by_name for n in needed):
        return by_name
    return None


def render_waveform_png(
    signal: np.ndarray,
    *,
    channels: Sequence[str],
    sample_rate_hz: float,
    title: str,
    dpi: int = 110,
) -> bytes:
    """Render ``signal`` (n_channels, n_samples) to a PNG byte string."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.atleast_2d(np.asarray(signal, dtype=np.float32))
    names = [str(c) for c in channels]
    if len(names) != arr.shape[0]:
        names = [f"ch{i}" for i in range(arr.shape[0])]
    fs = float(sample_rate_hz) if sample_rate_hz else 1.0
    t = np.arange(arr.shape[1], dtype=np.float32) / fs

    lead_map = twelve_lead_index(names)
    if lead_map is None and arr.shape[0] == 12:
        names = [lead for row in ECG_LEAD_GRID for lead in row]
        lead_map = twelve_lead_index(names)

    if lead_map is not None:
        fig, axes = plt.subplots(
            3,
            4,
            figsize=(12.0, 7.2),
            sharex=True,
            constrained_layout=True,
        )
        ymax = float(np.nanpercentile(np.abs(arr), 99.0) * 1.15) or 1.0
        for r, row in enumerate(ECG_LEAD_GRID):
            for c, lead in enumerate(row):
                ax = axes[r][c]
                idx = lead_map[lead]
                ax.plot(t, arr[idx], color="#1a1a1a", linewidth=0.7)
                ax.set_ylim(-ymax, ymax)
                ax.set_title(lead, loc="left", fontsize=9, pad=2)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if r < 2:
                    ax.set_xticklabels([])
                if c > 0:
                    ax.set_yticklabels([])
        for ax in axes[-1]:
            ax.set_xlabel("time (s)", fontsize=8)
        axes[1][0].set_ylabel("mV", fontsize=8)
    else:
        n_ch = arr.shape[0]
        fig, axes = plt.subplots(
            n_ch,
            1,
            figsize=(9.0, max(2.4, 2.2 * n_ch)),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        for i in range(n_ch):
            ax = axes[i][0]
            ax.plot(t, arr[i], linewidth=0.9)
            ax.set_ylabel(names[i])
            ax.grid(True, alpha=0.3)
        axes[-1][0].set_xlabel("time (s)")

    fig.suptitle(title, fontsize=10)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
