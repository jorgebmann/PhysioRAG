"""PTB-XL 12-lead ECG processor (Phase B, study-level CLIP pairs).

Unlike :class:`WfdbWaveformProcessor` (continuous ICU waveforms, 2 channels,
125 Hz), this reads **study-level** 12-lead ECG records where one ~10 s
recording maps to one epoch with its own diagnostic report — the strong-tier
pairing the reused ECG dual-encoder (MERL) expects.

Source layout ([PTB-XL](https://physionet.org/content/ptb-xl/)):
  ptbxl/
    ptbxl_database.csv     # ecg_id, patient_id, scp_codes, report, filename_hr, ...
    scp_statements.csv     # SCP code -> human-readable description
    records500/<bin>/<ecg_id>_hr.hea|.dat   # 500 Hz, 12-lead

When the database CSV is present it drives iteration (captions + patient ids for
patient-level splits). Otherwise the processor falls back to globbing 500 Hz
record headers so it stays usable on ad-hoc WFDB fixtures.
"""

from __future__ import annotations

import csv
from os import PathLike
from pathlib import Path
from typing import Iterator

import numpy as np

from physiorag.ingestion.base import WaveformEpoch, WaveformProcessor
from physiorag.ingestion.signal_prep import fill_non_finite, pad_or_truncate, resample_to_rate

# Canonical 12-lead order (matches physiorag.embeddings.merl.STANDARD_LEADS).
STANDARD_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)
_DB_CSV = "ptbxl_database.csv"
_SCP_CSV = "scp_statements.csv"


class PtbxlEcgProcessor(WaveformProcessor):
    """Yield 12-lead ECG epochs (one per study) with diagnostic captions."""

    def __init__(
        self,
        *,
        modality: str = "ecg",
        window_seconds: float = 10.0,
        target_sample_rate_hz: float = 500.0,
        max_records: int = 200,
        prefer_high_rate: bool = True,
    ) -> None:
        self.modality = modality
        self.window_seconds = window_seconds
        self.target_sample_rate_hz = float(target_sample_rate_hz)
        self.max_records = max_records
        self.prefer_high_rate = prefer_high_rate
        self.target_len = int(round(window_seconds * self.target_sample_rate_hz))

    # -- iteration ---------------------------------------------------------
    def iter_epochs(self, source: str | PathLike[str]) -> Iterator[WaveformEpoch]:
        root = Path(source)
        db_csv = root / _DB_CSV
        if db_csv.exists():
            yield from self._iter_from_csv(root, db_csv)
        else:
            yield from self._iter_from_headers(root)

    def _iter_from_csv(self, root: Path, db_csv: Path) -> Iterator[WaveformEpoch]:
        scp_map = self._load_scp_map(root / _SCP_CSV)
        count = 0
        with db_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if count >= self.max_records:
                    break
                rel = row.get("filename_hr") if self.prefer_high_rate else row.get("filename_lr")
                rel = rel or row.get("filename_lr") or row.get("filename_hr")
                if not rel:
                    continue
                record_path = root / rel
                epoch = self._read_record(
                    record_path,
                    ecg_id=str(row.get("ecg_id", record_path.name)),
                    patient_id=str(row.get("patient_id", "")),
                    report=str(row.get("report", "") or ""),
                    scp_codes=str(row.get("scp_codes", "") or ""),
                    scp_map=scp_map,
                )
                if epoch is None:
                    continue
                count += 1
                yield self.process_epoch(epoch)

    def _iter_from_headers(self, root: Path) -> Iterator[WaveformEpoch]:
        search_root = root / "records500" if (root / "records500").is_dir() else root
        headers = sorted(search_root.rglob("*.hea"))
        count = 0
        for hea in headers:
            if count >= self.max_records:
                break
            epoch = self._read_record(
                hea.with_suffix(""),
                ecg_id=hea.stem,
                patient_id="",
                report="",
                scp_codes="",
                scp_map={},
            )
            if epoch is None:
                continue
            count += 1
            yield self.process_epoch(epoch)

    def _read_record(
        self,
        record_path: Path,
        *,
        ecg_id: str,
        patient_id: str,
        report: str,
        scp_codes: str,
        scp_map: dict[str, str],
    ) -> WaveformEpoch | None:
        import wfdb

        try:
            rec = wfdb.rdrecord(str(record_path))
        except Exception:
            return None
        signal = np.asarray(rec.p_signal, dtype=np.float32)  # (n_samples, n_leads)
        if signal.size == 0:
            return None
        sig_name = list(getattr(rec, "sig_name", None) or [])
        fs = float(getattr(rec, "fs", 0) or 0) or self.target_sample_rate_hz
        ordered = self._reorder_leads(signal.T, sig_name)  # (12, n_samples)

        caption = self._caption(report=report, scp_codes=scp_codes, scp_map=scp_map)
        return WaveformEpoch(
            record_id=ecg_id,
            modality=self.modality,
            start_time_s=0.0,
            sample_rate_hz=fs,
            signal=ordered,
            metadata={
                "patient_id": patient_id,
                "ecg_id": ecg_id,
                "channels": list(STANDARD_LEADS),
                "native_fs_hz": fs,
                "database": "ptb-xl",
                "report": report,
                "scp_codes": scp_codes,
                "text": caption,
            },
        )

    def process_epoch(self, epoch: WaveformEpoch) -> WaveformEpoch:
        signal = fill_non_finite(epoch.signal)
        signal = resample_to_rate(signal, epoch.sample_rate_hz, self.target_sample_rate_hz)
        signal = pad_or_truncate(signal, n_samples=self.target_len)
        meta = dict(epoch.metadata)
        return WaveformEpoch(
            record_id=epoch.record_id,
            modality=epoch.modality,
            start_time_s=epoch.start_time_s,
            sample_rate_hz=self.target_sample_rate_hz,
            signal=signal.astype(np.float32),
            metadata={**meta, "sample_rate_hz": self.target_sample_rate_hz},
        )

    # -- helpers -----------------------------------------------------------
    def _reorder_leads(self, signal: np.ndarray, sig_name: list[str]) -> np.ndarray:
        """Place leads into canonical 12-lead order; zero-fill missing leads."""
        n_leads, length = signal.shape
        out = np.zeros((len(STANDARD_LEADS), length), dtype=np.float32)
        lookup = {name.strip().lower(): i for i, name in enumerate(sig_name)}
        for target_idx, lead in enumerate(STANDARD_LEADS):
            src = lookup.get(lead.lower())
            if src is not None and src < n_leads:
                out[target_idx] = signal[src]
        return out

    @staticmethod
    def _caption(*, report: str, scp_codes: str, scp_map: dict[str, str]) -> str:
        report = (report or "").strip()
        descriptions: list[str] = []
        for code in _parse_scp_codes(scp_codes):
            desc = scp_map.get(code)
            if desc and desc not in descriptions:
                descriptions.append(desc)
        parts = [p for p in (report, "; ".join(descriptions)) if p]
        text = ". ".join(parts).strip()
        return text or "12-lead ECG (no report available)."

    @staticmethod
    def _load_scp_map(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        mapping: dict[str, str] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            key = reader.fieldnames[0] if reader.fieldnames else ""
            for row in reader:
                code = (row.get(key) or "").strip()
                desc = (row.get("description") or "").strip()
                if code:
                    mapping[code] = desc or code
        return mapping


def _parse_scp_codes(scp_codes: str) -> list[str]:
    """PTB-XL stores scp_codes as a dict-literal string, e.g. "{'NORM': 100.0}"."""
    text = (scp_codes or "").strip()
    if not text:
        return []
    codes: list[str] = []
    for chunk in text.strip("{}").split(","):
        key = chunk.split(":", 1)[0].strip().strip("'\"")
        if key:
            codes.append(key)
    return codes
