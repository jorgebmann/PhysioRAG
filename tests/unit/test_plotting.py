"""12-lead grid vs stacked PNG plots."""

from __future__ import annotations

import numpy as np

from physiorag.plotting import (
    ECG_LEAD_GRID,
    png_wh,
    render_waveform_png,
    twelve_lead_index,
)


def test_twelve_lead_index_accepts_avr_aliases() -> None:
    channels = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    idx = twelve_lead_index(channels)
    assert idx is not None
    assert idx["aVR"] == 3
    assert idx["V6"] == 11


def test_twelve_lead_index_rejects_vent() -> None:
    assert twelve_lead_index(["Paw", "Flow"]) is None


def test_stacked_png_is_valid() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(2, 1250)).astype(np.float32)
    png = render_waveform_png(
        signal,
        channels=["Paw", "Flow"],
        sample_rate_hz=125.0,
        title="vent",
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = png_wh(png)
    assert width > 400
    assert height > 200


def test_twelve_lead_png_is_landscape_not_a_skyscraper() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    t = np.arange(n) / 500.0
    signal = np.stack(
        [np.sin(2 * np.pi * (1.0 + k * 0.1) * t) for k in range(12)],
        axis=0,
    ).astype(np.float32)
    channels = [lead for row in ECG_LEAD_GRID for lead in row]
    png = render_waveform_png(
        signal,
        channels=channels,
        sample_rate_hz=500.0,
        title="ecg-001 — ecg",
    )
    width, height = png_wh(png)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # Old stacked layout was ~9×26.4 in @ 110 dpi → height ~2900 px.
    assert height < 1400, f"12-lead PNG too tall: {width}x{height}"
    assert width > height, f"expected landscape 3×4 grid, got {width}x{height}"
