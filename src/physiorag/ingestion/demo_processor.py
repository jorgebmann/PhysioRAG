"""Demo waveform processor with synthetic ICU ventilator-like epochs.

Ventilator scenarios (asynchrony types, captions, structured metadata) live in
:mod:`physiorag.ingestion.vent_captions`. Each scenario can be emitted as one or
more epochs (``variants``) with independent noise so a retrieval eval has
distractors without drowning out the curated smoke demo.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterator

import numpy as np

from physiorag.ingestion.base import WaveformEpoch, WaveformProcessor
from physiorag.ingestion.vent_captions import (
    SPO2_SCENARIO,
    VENT_CATALOG,
    build_metadata,
)


def _demo_scenarios() -> list[dict]:
    """Flatten the vent catalog + SpO2 scenario into demo scenario dicts."""
    scenarios: list[dict] = []
    for entry in VENT_CATALOG:
        scenarios.append({**entry, "modality": "ventilator"})
    scenarios.append({**SPO2_SCENARIO, "modality": SPO2_SCENARIO.get("modality", "spo2")})
    return scenarios


# Back-compat: a flat list of scenarios (used by tests / introspection).
DEMO_SCENARIOS: list[dict] = _demo_scenarios()


class DemoWaveformProcessor(WaveformProcessor):
    """Generate or load small demo epochs for the PhysioRAG quickstart."""

    def __init__(
        self,
        *,
        modality: str = "ventilator",
        window_seconds: float = 10.0,
        sample_rate_hz: float = 125.0,
        seed: int = 42,
        variants: int = 1,
    ) -> None:
        self.modality = modality
        self.window_seconds = window_seconds
        self.sample_rate_hz = sample_rate_hz
        self.n_samples = int(window_seconds * sample_rate_hz)
        self.variants = max(1, int(variants))
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
        for scenario in matched:
            base_id = scenario["record_id"]
            for variant in range(self.variants):
                # Variant 0 keeps the frozen record_id + t=0 so epoch ids like
                # ``demo-ards-001_0`` stay stable for smoke / API tests.
                record_id = base_id if variant == 0 else f"{base_id}-v{variant + 1}"
                start_time_s = float(variant * self.window_seconds)
                signal = self._synthesize(scenario["pattern"], modality=scenario["modality"])
                metadata = {
                    **build_metadata(scenario, modality=scenario["modality"]),
                    "start_time_s": start_time_s,
                }
                epoch = WaveformEpoch(
                    record_id=record_id,
                    modality=scenario["modality"],
                    start_time_s=start_time_s,
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

        pressure, flow = self._vent_breath_train(t, pattern)
        noise_p = 0.25 * self.rng.normal(size=self.n_samples)
        noise_f = 0.7 * self.rng.normal(size=self.n_samples)
        return np.stack(
            [
                (pressure + noise_p).astype(np.float32),
                (flow + noise_f).astype(np.float32),
            ],
            axis=0,
        )

    def _vent_breath_train(self, t: np.ndarray, pattern: str) -> tuple[np.ndarray, np.ndarray]:
        """Textbook-ish Paw/Flow scalar shapes for each asynchrony scenario.

        A controlled breath is a train of period ``T``: pressure ramps to a
        plateau over the inspiratory fraction ``ti`` then falls back to PEEP;
        flow is decelerating on inspiration and passive-decaying on expiration.
        Each ``pattern`` perturbs that baseline into the classic scalar shape.
        Absolute units are illustrative; epochs are z-normalized downstream so
        only morphology is retained.
        """
        period_s = 3.0
        peep = 5.0
        pip = 22.0
        # Delayed cycling = inspiration that runs long into neural expiration.
        ti = 0.55 if pattern == "delayed" else 0.34

        ph = np.mod(t, period_s) / period_s  # 0..1 within a breath
        insp = ph < ti
        insp_prog = np.clip(ph / ti, 0.0, 1.0)  # 0..1 across inspiration
        exp_prog = np.clip((ph - ti) / (1.0 - ti), 0.0, 1.0)  # 0..1 across expiration

        # Baseline controlled breath (VCV/PSV-like).
        rise = np.clip(insp_prog / 0.35, 0.0, 1.0)  # ramp to plateau
        pressure = np.where(insp, peep + (pip - peep) * rise, peep).astype(np.float32)
        flow = np.where(
            insp,
            30.0 * np.exp(-3.0 * insp_prog),  # decelerating inspiratory flow
            -30.0 * np.exp(-4.0 * exp_prog),  # passive expiratory flow
        ).astype(np.float32)

        if pattern == "spike":
            # Double triggering: a second stacked inspiration after a brief,
            # incomplete expiration.
            retrig = (~insp) & (exp_prog < 0.20)
            pressure = np.where(retrig, peep + (pip - peep) * 0.95, pressure)
            flow = np.where(retrig, 26.0 * np.exp(-5.0 * exp_prog), flow)
        elif pattern == "trap":
            # Air trapping / auto-PEEP: expiratory flow never returns to zero
            # and the baseline pressure creeps up breath over breath.
            flow = np.where(insp, flow, -16.0 * np.exp(-1.1 * exp_prog) - 4.0)
            pressure = pressure + np.linspace(0.0, 5.0, t.size, dtype=np.float32)
        elif pattern == "stiff":
            # Low compliance: high peak pressure reached fast, small flow.
            rise_stiff = np.clip(insp_prog / 0.25, 0.0, 1.0)
            pressure = np.where(insp, peep + (34.0 - peep) * rise_stiff, peep + 2.0).astype(
                np.float32
            )
            flow = flow * 0.5
        elif pattern == "ineffective":
            # Ineffective effort: a small mid-expiratory Paw dip + flow blip that
            # never crosses the trigger threshold.
            effort = (~insp) & (np.abs(exp_prog - 0.5) < 0.07)
            pressure = pressure - np.where(effort, 2.5, 0.0)
            flow = flow + np.where(effort, 6.0, 0.0)
        elif pattern == "flow_starv":
            # Flow starvation: scooped, concave inspiratory pressure as demand
            # outstrips the set flow.
            scoop = np.where(insp, 7.0 * np.sin(np.pi * insp_prog) ** 2, 0.0)
            pressure = pressure - scoop
        elif pattern == "reverse":
            # Reverse triggering: a late inspiratory effort entrained by the
            # mechanical breath adds a second-half deflection.
            frac = np.clip((insp_prog - 0.6) / 0.4, 0.0, 1.0)
            bump = np.where(insp, np.sin(np.pi * frac), 0.0)
            pressure = pressure + 4.0 * bump
            flow = flow + 9.0 * bump
        # "normal" and "delayed" already captured by the baseline + ti.
        return pressure, flow
