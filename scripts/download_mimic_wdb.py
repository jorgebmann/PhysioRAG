#!/usr/bin/env python3
"""Download a bounded subset of the MIMIC-IV Waveform Database from PhysioNet.

Credentialed access only. You must have a PhysioNet account that has signed the
MIMIC-IV Waveform Database DUA. Credentials are read from the environment
(PHYSIONET_USERNAME / PHYSIONET_PASSWORD) or a local .env file — never from YAML.

Usage:
    export PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=...
    python scripts/download_mimic_wdb.py --max-records 3
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Does not overwrite existing env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@contextmanager
def physionet_netrc(username: str, password: str) -> Iterator[None]:
    """Temporarily install ~/.netrc so ``requests`` (used by wfdb) authenticates.

    Any existing ~/.netrc is backed up and restored afterwards.
    """
    netrc_path = Path.home() / ".netrc"
    backup: bytes | None = None
    if netrc_path.exists():
        backup = netrc_path.read_bytes()
    try:
        netrc_path.write_text(
            f"machine physionet.org login {username} password {password}\n",
            encoding="utf-8",
        )
        netrc_path.chmod(0o600)
        yield
    finally:
        if backup is not None:
            netrc_path.write_bytes(backup)
            netrc_path.chmod(0o600)
        else:
            netrc_path.unlink(missing_ok=True)


def _record_names(records: list[str]) -> list[str]:
    """Normalize to MIMIC record **directory** paths under waves/.

    Example: ``p100/p10014354/81739927`` ->
    ``waves/p100/p10014354/81739927`` (directory containing ``81739927.hea``).
    """
    normalized = []
    for rec in records:
        rec = rec.strip().strip("/")
        if not rec:
            continue
        path = rec if rec.startswith("waves/") else f"waves/{rec}"
        parts = path.split("/")
        if len(parts) >= 2 and parts[-1] == parts[-2]:
            path = "/".join(parts[:-1])
        normalized.append(path)
    return normalized


def _list_remote_files(record_dir: str, *, db_dir: str, session) -> list[str]:
    """List downloadable WFDB files in a MIMIC record directory via PhysioNet index."""
    import re

    index_url = f"https://physionet.org/files/{db_dir}/{record_dir}/"
    resp = session.get(index_url, timeout=60)
    resp.raise_for_status()
    names = re.findall(r'href="([^"?/][^"]*)"', resp.text)
    allowed = (".hea", ".dat", ".csv.gz", ".n.csv.gz")
    files = []
    for name in names:
        if name.startswith("../") or name.endswith("/"):
            continue
        if name.endswith(allowed):
            files.append(f"{record_dir}/{name}")
    return sorted(set(files))


def download(
    *,
    database: str,
    out_dir: Path,
    records: list[str],
    username: str,
    password: str,
) -> dict:
    import posixpath
    import requests
    from wfdb.io import download as wfdb_download

    out_dir.mkdir(parents=True, exist_ok=True)
    names = _record_names(records)

    db_dir = posixpath.join(database, wfdb_download.get_version(database))
    session = requests.Session()
    session.auth = (username, password)

    all_files: list[str] = []
    for record_dir in names:
        files = _list_remote_files(record_dir, db_dir=db_dir, session=session)
        if not files:
            raise RuntimeError(f"No WFDB files found for record directory: {record_dir}")
        all_files.extend(files)

    with physionet_netrc(username, password):
        wfdb_download.dl_files(
            database,
            dl_dir=str(out_dir),
            files=all_files,
            keep_subdirs=True,
            overwrite=False,
        )
    return {
        "database": database,
        "out_dir": str(out_dir),
        "records": names,
        "files_downloaded": len(all_files),
    }


def main(argv: list[str] | None = None) -> None:
    from physiorag.config import load_config

    parser = argparse.ArgumentParser(description="Download a MIMIC-IV WFDB subset")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--database", default=None, help="Override PhysioNet database slug")
    parser.add_argument("--out-dir", default=None, help="Override local mirror directory")
    parser.add_argument("--max-records", type=int, default=None, help="Cap number of records")
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Explicit record paths (relative to db root, e.g. p100/p10039708/83411188)",
    )
    args = parser.parse_args(argv)

    _load_dotenv(ROOT / ".env")
    username = os.environ.get("PHYSIONET_USERNAME", "")
    password = os.environ.get("PHYSIONET_PASSWORD", "")
    if not username or not password:
        parser.error(
            "Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD (env or .env). "
            "See .env.example."
        )

    cfg = load_config(args.config)
    physio = cfg.get("physionet", {})
    data_cfg = cfg.get("data", {})

    database = args.database or physio.get("database", "mimic4wdb")
    out_dir = Path(
        args.out_dir
        or (Path(data_cfg.get("raw_dir", "data/raw")) / physio.get("mirror_subdir", "mimic4wdb"))
    )
    records = args.records if args.records is not None else list(physio.get("default_records", []))
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        parser.error("No records specified (config.physionet.default_records is empty)")

    result = download(
        database=database,
        out_dir=out_dir,
        records=records,
        username=username,
        password=password,
    )
    print(result)


if __name__ == "__main__":
    main()
