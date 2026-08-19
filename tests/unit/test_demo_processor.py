"""Tests for synthetic demo waveform generation."""

from __future__ import annotations

from pathlib import Path

from physiorag.ingestion.demo_processor import DemoWaveformProcessor


def test_demo_processor_yields_ventilator_epochs(tmp_path: Path) -> None:
    processor = DemoWaveformProcessor(modality="ventilator", window_seconds=10.0)
    epochs = list(processor.iter_epochs(tmp_path))
    assert len(epochs) == 8  # one epoch per vent scenario in the catalog
    assert all(e.modality == "ventilator" for e in epochs)
    assert epochs[0].signal.ndim == 2
    assert epochs[0].signal.shape[1] == 1250
    # Frozen id + Paw/Flow channels + structured asynchrony metadata.
    assert epochs[0].record_id == "demo-ards-001"
    assert epochs[0].metadata["channels"] == ["Paw", "Flow"]
    assert epochs[0].metadata["pairing_tier"] == "medium"
    types = {e.metadata.get("asynchrony_type") for e in epochs}
    assert {"double_triggering", "ineffective_effort", "flow_starvation"} <= types


def test_demo_processor_variants_add_distinct_epochs(tmp_path: Path) -> None:
    processor = DemoWaveformProcessor(modality="ventilator", window_seconds=10.0, variants=3)
    epochs = list(processor.iter_epochs(tmp_path))
    assert len(epochs) == 24  # 8 scenarios x 3 variants
    ids = [e.record_id for e in epochs]
    assert len(set(ids)) == len(ids)  # unique record ids
    assert "demo-ards-001" in ids  # variant 0 keeps the frozen id


def test_demo_processor_captions_are_bilingual(tmp_path: Path) -> None:
    processor = DemoWaveformProcessor(modality="ventilator")
    epochs = list(processor.iter_epochs(tmp_path))
    spike = next(e for e in epochs if e.record_id == "demo-ards-001")
    text = spike.metadata["text"].lower()
    assert "ventilator" in text  # English half
    assert "beatmungsgerät" in text  # German half (umlaut, matches UI chips)
