"""Tests for synthetic demo waveform generation."""

from __future__ import annotations

from pathlib import Path

from physiorag.ingestion.demo_processor import DemoWaveformProcessor


def test_demo_processor_yields_ventilator_epochs(tmp_path: Path) -> None:
    processor = DemoWaveformProcessor(modality="ventilator", window_seconds=10.0)
    epochs = list(processor.iter_epochs(tmp_path))
    assert len(epochs) == 4
    assert all(e.modality == "ventilator" for e in epochs)
    assert epochs[0].signal.ndim == 2
    assert epochs[0].signal.shape[1] == 1250
