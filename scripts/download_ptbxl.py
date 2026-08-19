#!/usr/bin/env python3
"""Download a bounded subset of PTB-XL (12-lead ECG) from PhysioNet.

PTB-XL is an **open-access** database (no MIMIC DUA), but you should still
review its terms on PhysioNet. This grabs the metadata CSVs plus the 500 Hz
records for the first N studies so PhysioRAG's ECG (MERL) track has a small,
bounded corpus for offline retrieval + Recall@k eval.

Usage:
    python scripts/download_ptbxl.py --max-records 200
    # then ingest:
    python scripts/ingest_waveforms.py --config configs/ecg_merl.yaml \
        --dataset ptbxl --modality ecg --reset-collection
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Database slug only. wfdb.dl_files() appends get_version(db) itself;
# passing "ptb-xl/1.0.3" 404s as /content/ptb-xl/1.0.3/1.0.3/.
PTBXL_DB = "ptb-xl"
DB_CSV = "ptbxl_database.csv"
SCP_CSV = "scp_statements.csv"


def _dl_files(files: list[str], dl_dir: Path) -> None:
    import wfdb

    wfdb.dl_files(PTBXL_DB, str(dl_dir), files)


def _record_files(dl_dir: Path, max_records: int) -> list[str]:
    """Read the metadata CSV and return the WFDB files for the first N studies."""
    db_path = dl_dir / DB_CSV
    files: list[str] = []
    with db_path.open(encoding="utf-8", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i >= max_records:
                break
            rel = row.get("filename_hr") or row.get("filename_lr")
            if not rel:
                continue
            files.append(f"{rel}.hea")
            files.append(f"{rel}.dat")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a bounded PTB-XL subset")
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument(
        "--dl-dir",
        default=str(ROOT / "data" / "raw" / "ptbxl"),
        help="Local mirror directory (matches configs/ecg_merl.yaml mirror_subdir).",
    )
    args = parser.parse_args(argv)

    dl_dir = Path(args.dl_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ptbxl] downloading metadata CSVs to {dl_dir} ...")
    _dl_files([DB_CSV, SCP_CSV], dl_dir)

    record_files = _record_files(dl_dir, args.max_records)
    if not record_files:
        print("[ptbxl] no records found in metadata CSV; aborting.", file=sys.stderr)
        return 1

    print(f"[ptbxl] downloading {len(record_files) // 2} records (500 Hz) ...")
    _dl_files(record_files, dl_dir)

    print(
        f"[ptbxl] done. {len(record_files) // 2} studies under {dl_dir}. "
        "Ingest with configs/ecg_merl.yaml (--dataset ptbxl --modality ecg)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
