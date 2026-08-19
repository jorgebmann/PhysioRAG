"""WFDB waveform processor for MIMIC-IV waveform records.

Streams fixed-length epochs directly from a local WFDB mirror using
``wfdb.rdrecord`` with ``sampfrom``/``sampto`` so that multi-hour ICU stays are
never fully loaded into RAM.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from physiorag.ingestion.base import WaveformEpoch, WaveformProcessor

# Signal-name substrings (case-insensitive) preferred per modality. Ventilator
# is restricted to true airway pressure / flow channels; arterial (abp/art),
# impedance respiration (resp), and generic "pressure" are intentionally absent
# so we never index a blood-pressure trace as if it were a ventilator waveform.
MODALITY_CHANNELS: dict[str, list[str]] = {
    "ventilator": ["paw", "awp", "airway", "flow"],
    "spo2": ["pleth", "spo2", "ppg"],
    "ecg": ["ecg", " ii", "ii", "iii", " i", "v1", "v2", "v5", "avr", "avl", "avf", "mcl"],
}
# Fallbacks used when no modality-specific channel is present in a record.
# Ventilator deliberately does NOT fall back (see _select_channels).
FALLBACK_CHANNELS = ["pleth", "abp", "art", "ii"]


class WfdbWaveformProcessor(WaveformProcessor):
    """Load real WFDB records and yield cleaned, fixed-window epochs."""

    def __init__(
        self,
        *,
        modality: str = "ventilator",
        window_seconds: float = 10.0,
        target_sample_rate_hz: float = 125.0,
        max_epochs_per_record: int = 20,
        max_channels: int = 2,
    ) -> None:
        self.modality = modality
        self.window_seconds = window_seconds
        self.target_sample_rate_hz = float(target_sample_rate_hz)
        self.max_epochs_per_record = max_epochs_per_record
        self.max_channels = max_channels
        self.target_len = int(round(window_seconds * self.target_sample_rate_hz))

    # -- discovery ---------------------------------------------------------
    @staticmethod
    def find_records(root: Path) -> list[str]:
        """Return record paths (without .hea) for top-level (multi-segment) records.

        A top-level record header lives in a directory named after the record,
        e.g. ``.../83411188/83411188.hea``.
        """
        records: list[str] = []
        for hea in sorted(root.rglob("*.hea")):
            if hea.stem == hea.parent.name:
                records.append(str(hea.with_suffix("")))
        return records

    # -- iteration ---------------------------------------------------------
    def iter_epochs(self, source: str | PathLike[str]) -> Iterator[WaveformEpoch]:
        import wfdb

        root = Path(source)
        record_paths = self.find_records(root)
        for record_path in record_paths:
            try:
                header = wfdb.rdheader(record_path)
            except Exception:
                continue
            fs = float(getattr(header, "fs", 0) or 0)
            total = int(getattr(header, "sig_len", 0) or 0)
            sig_name = list(getattr(header, "sig_name", None) or [])
            if not sig_name:
                sig_name = self._peek_sig_names(record_path)
            if fs <= 0 or total <= 0 or not sig_name:
                continue

            channels, fallback = self._select_channels(sig_name)
            if not channels:
                continue

            win = int(round(self.window_seconds * fs))
            if win <= 0:
                continue
            n_windows = min(total // win, self.max_epochs_per_record)
            record_id = Path(record_path).name
            subject = Path(record_path).parent.name

            for w in range(n_windows):
                sampfrom = w * win
                sampto = sampfrom + win
                try:
                    rec = wfdb.rdrecord(
                        record_path,
                        sampfrom=sampfrom,
                        sampto=sampto,
                        channels=channels,
                    )
                except Exception:
                    continue
                signal = np.asarray(rec.p_signal, dtype=np.float32)  # (win, n_ch)
                if signal.size == 0:
                    continue
                signal = signal.T  # (n_ch, win)
                chan_names = list(getattr(rec, "sig_name", None) or [sig_name[i] for i in channels])
                display_channels = self._display_channels(chan_names)
                start_time_s = float(sampfrom / fs)
                text = self._describe(
                    record_id=record_id,
                    subject=subject,
                    channels=display_channels,
                    start_time_s=start_time_s,
                    fallback=fallback,
                )
                epoch = WaveformEpoch(
                    record_id=record_id,
                    modality=self.modality,
                    start_time_s=start_time_s,
                    sample_rate_hz=fs,
                    signal=signal,
                    metadata={
                        "subject": subject,
                        "channels": display_channels,
                        "source_channels": chan_names,
                        "native_fs_hz": fs,
                        "database": "mimic4wdb",
                        "channel_fallback": fallback,
                        # Continuous ICU waveform + generated prose is a WEAK pair
                        # (settings/description, not a morphology caption). Never
                        # promoted into contrastive training. See docs/VENT_RETRIEVAL.md.
                        "pairing_tier": "weak",
                        "text": text,
                    },
                )
                yield self.process_epoch(epoch)

    def process_epoch(self, epoch: WaveformEpoch) -> WaveformEpoch:
        signal = np.asarray(epoch.signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[None, :]
        # Fill gaps (NaN) then resample to the target rate.
        signal = self._fill_nan(signal)
        signal = self._resample(signal, epoch.sample_rate_hz)
        # Artifact clip + per-channel z-normalization.
        signal = np.clip(signal, -1e4, 1e4)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True) + 1e-6
        signal = ((signal - mean) / std).astype(np.float32)
        meta = dict(epoch.metadata)
        return WaveformEpoch(
            record_id=epoch.record_id,
            modality=epoch.modality,
            start_time_s=epoch.start_time_s,
            sample_rate_hz=self.target_sample_rate_hz,
            signal=signal,
            metadata={**meta, "sample_rate_hz": self.target_sample_rate_hz},
        )

    # -- helpers -----------------------------------------------------------
    def _peek_sig_names(self, record_path: str) -> list[str]:
        import wfdb

        try:
            rec = wfdb.rdrecord(record_path, sampfrom=0, sampto=2)
            return list(getattr(rec, "sig_name", None) or [])
        except Exception:
            return []

    def _select_channels(self, sig_name: Sequence[str]) -> tuple[list[int], bool]:
        lowered = [s.lower() for s in sig_name]
        wanted = MODALITY_CHANNELS.get(self.modality, [])
        indices = self._match(lowered, wanted)
        fallback = False
        # Ventilator retrieval must be real airway pressure/flow. A record with
        # no Paw/Flow-like channel is skipped rather than indexing an arterial
        # or impedance trace under the ventilator modality.
        if not indices and self.modality == "ventilator":
            return [], False
        if not indices:
            indices = self._match(lowered, FALLBACK_CHANNELS)
            fallback = True
        if not indices:
            indices = list(range(min(self.max_channels, len(sig_name))))
            fallback = True
        return indices[: self.max_channels], fallback

    def _display_channels(self, channels: Sequence[str]) -> list[str]:
        """Relabel recognized ventilator pressure/flow channels to Paw/Flow.

        Only fires for the ventilator modality and only on names that clearly
        match a pressure or flow keyword; anything else keeps its native name so
        clinical channels are never silently mislabeled.
        """
        if self.modality != "ventilator":
            return list(channels)
        out: list[str] = []
        for name in channels:
            low = name.lower()
            if any(k in low for k in ("paw", "awp", "airway", "pressure")):
                out.append("Paw")
            elif "flow" in low:
                out.append("Flow")
            else:
                out.append(name)
        return out

    @staticmethod
    def _match(lowered: Sequence[str], wanted: Sequence[str]) -> list[int]:
        hits: list[int] = []
        for idx, name in enumerate(lowered):
            if any(w.strip() and w.strip() in name for w in wanted):
                hits.append(idx)
        return hits

    @staticmethod
    def _fill_nan(signal: np.ndarray) -> np.ndarray:
        out = np.array(signal, dtype=np.float32, copy=True)
        for ch in range(out.shape[0]):
            row = out[ch]
            mask = np.isnan(row)
            if mask.all():
                out[ch] = 0.0
            elif mask.any():
                idx = np.arange(row.size)
                row[mask] = np.interp(idx[mask], idx[~mask], row[~mask])
                out[ch] = row
        return out

    def _resample(self, signal: np.ndarray, native_fs: float) -> np.ndarray:
        n_ch, n = signal.shape
        target = self.target_len
        if n == target:
            return signal
        src = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float64)
        dst = np.linspace(0.0, 1.0, num=target, endpoint=False, dtype=np.float64)
        out = np.empty((n_ch, target), dtype=np.float32)
        for ch in range(n_ch):
            out[ch] = np.interp(dst, src, signal[ch]).astype(np.float32)
        return out

    def _describe(
        self,
        *,
        record_id: str,
        subject: str,
        channels: Sequence[str],
        start_time_s: float,
        fallback: bool,
    ) -> str:
        chan_str = ", ".join(channels) if channels else "unknown"
        note = " (fallback channels)" if fallback else ""
        return (
            f"MIMIC-IV waveform record {record_id} (subject {subject}), "
            f"{self.window_seconds:.0f}s {self.modality} window starting at "
            f"t={start_time_s:.1f}s. Channels: {chan_str}{note}; "
            f"resampled to {self.target_sample_rate_hz:.0f} Hz."
        )
