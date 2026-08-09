"""Demo waveform processor with synthetic ICU ventilator-like epochs."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterator

import numpy as np

from physiorag.ingestion.base import WaveformEpoch, WaveformProcessor

# Hand-authored scenarios so BM25 / hybrid search has something useful to match.
DEMO_SCENARIOS: list[dict] = [
    {
        "record_id": "demo-ards-001",
        "modality": "ventilator",
        "label": "ards_asynchrony_pressure_spike",
        "text": (
            "12-second ventilator pressure/flow window: ARDS patient breathing "
            "spontaneously against the ventilator causing a clear pressure spike "
            "and flow reversal suggestive of patient-ventilator asynchrony."
        ),
        "metadata": {
            "diagnosis": "ARDS",
            "event": "patient_ventilator_asynchrony",
            "finding": "pressure_spike",
        },
        "pattern": "spike",
    },
    {
        "record_id": "demo-copd-002",
        "modality": "ventilator",
        "label": "copd_air_trapping",
        "text": (
            "Ventilator flow curve with incomplete expiration and rising end-expiratory "
            "pressure consistent with air trapping in a COPD exacerbation."
        ),
        "metadata": {
            "diagnosis": "COPD",
            "event": "air_trapping",
            "finding": "auto_peep",
        },
        "pattern": "trap",
    },
    {
        "record_id": "demo-normal-003",
        "modality": "ventilator",
        "label": "normal_controlled_breaths",
        "text": (
            "Stable volume-controlled ventilation with regular pressure and flow "
            "waveforms and no obvious asynchrony."
        ),
        "metadata": {
            "diagnosis": "post_op",
            "event": "controlled_ventilation",
            "finding": "normal",
        },
        "pattern": "normal",
    },
    {
        "record_id": "demo-ards-004",
        "modality": "ventilator",
        "label": "ards_low_compliance",
        "text": (
            "ARDS low-compliance pressure curve with elevated peak pressures during "
            "controlled breaths and reduced tidal excursion."
        ),
        "metadata": {
            "diagnosis": "ARDS",
            "event": "low_compliance",
            "finding": "high_peak_pressure",
        },
        "pattern": "stiff",
    },
    {
        "record_id": "demo-spo2-005",
        "modality": "spo2",
        "label": "desaturation_event",
        "text": (
            "Photoplethysmogram window around an SpO2 desaturation episode with "
            "reduced pulse amplitude."
        ),
        "metadata": {
            "diagnosis": "hypoxemia",
            "event": "desaturation",
            "finding": "low_spo2",
        },
        "pattern": "desat",
    },
]


class DemoWaveformProcessor(WaveformProcessor):
    """Generate or load small demo epochs for the PhysioRAG quickstart."""

    def __init__(
        self,
        *,
        modality: str = "ventilator",
        window_seconds: float = 10.0,
        sample_rate_hz: float = 125.0,
        seed: int = 42,
    ) -> None:
        self.modality = modality
        self.window_seconds = window_seconds
        self.sample_rate_hz = sample_rate_hz
        self.n_samples = int(window_seconds * sample_rate_hz)
        self.rng = np.random.default_rng(seed)

    def iter_epochs(self, source: str | PathLike[str]) -> Iterator[WaveformEpoch]:
        root = Path(source)
        npy_files = sorted(root.glob("*.npy")) if root.is_dir() else []
        if npy_files:
            yield from self._iter_from_npy(npy_files)
            return
        yield from self._iter_synthetic()

    def process_epoch(self, epoch: WaveformEpoch) -> WaveformEpoch:
        signal = np.asarray(epoch.signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[None, :]
        signal = np.clip(signal, -10.0, 10.0)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True) + 1e-6
        signal = (signal - mean) / std
        return WaveformEpoch(
            record_id=epoch.record_id,
            modality=epoch.modality,
            start_time_s=epoch.start_time_s,
            sample_rate_hz=epoch.sample_rate_hz,
            signal=signal.astype(np.float32),
            metadata=dict(epoch.metadata),
        )

    def _iter_from_npy(self, files: list[Path]) -> Iterator[WaveformEpoch]:
        for idx, path in enumerate(files):
            signal = np.load(path).astype(np.float32)
            epoch = WaveformEpoch(
                record_id=path.stem,
                modality=self.modality,
                start_time_s=float(idx * self.window_seconds),
                sample_rate_hz=self.sample_rate_hz,
                signal=signal,
                metadata={"source_file": str(path), "text": f"Loaded demo waveform {path.name}"},
            )
            yield self.process_epoch(epoch)

    def _iter_synthetic(self) -> Iterator[WaveformEpoch]:
        matched = [s for s in DEMO_SCENARIOS if s["modality"] == self.modality]
        for idx, scenario in enumerate(matched):
            signal = self._synthesize(scenario["pattern"], modality=scenario["modality"])
            metadata = {
                **scenario["metadata"],
                "label": scenario["label"],
                "text": scenario["text"],
                "start_time_s": float(idx * self.window_seconds),
            }
            epoch = WaveformEpoch(
                record_id=scenario["record_id"],
                modality=scenario["modality"],
                start_time_s=float(idx * self.window_seconds),
                sample_rate_hz=self.sample_rate_hz,
                signal=signal,
                metadata=metadata,
            )
            yield self.process_epoch(epoch)

    def _synthesize(self, pattern: str, *, modality: str) -> np.ndarray:
        t = np.arange(self.n_samples, dtype=np.float32) / self.sample_rate_hz
        if modality == "spo2":
            base = 0.8 + 0.2 * np.sin(2 * np.pi * 1.2 * t)
            if pattern == "desat":
                base = base * np.linspace(1.0, 0.55, self.n_samples, dtype=np.float32)
            noise = 0.02 * self.rng.normal(size=self.n_samples).astype(np.float32)
            return (base + noise).astype(np.float32)[None, :]

        breath_hz = 0.35
        pressure = 8 + 6 * (0.5 + 0.5 * np.sin(2 * np.pi * breath_hz * t))
        flow = 20 * np.sin(2 * np.pi * breath_hz * t)
        if pattern == "spike":
            center = self.n_samples // 2
            width = int(0.4 * self.sample_rate_hz)
            pressure[center : center + width] += 18
            flow[center : center + width] -= 25
        elif pattern == "trap":
            flow = np.where(flow < 0, flow * 0.4, flow)
            pressure = pressure + np.linspace(0, 4, self.n_samples, dtype=np.float32)
        elif pattern == "stiff":
            pressure = 12 + 14 * (0.5 + 0.5 * np.sin(2 * np.pi * breath_hz * t))
            flow = flow * 0.55
        noise_p = 0.3 * self.rng.normal(size=self.n_samples)
        noise_f = 0.8 * self.rng.normal(size=self.n_samples)
        return np.stack(
            [
                (pressure + noise_p).astype(np.float32),
                (flow + noise_f).astype(np.float32),
            ],
            axis=0,
        )
