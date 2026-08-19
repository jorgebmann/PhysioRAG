"""Guards for the PTB-XL download helper (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_ptbxl.py"
_spec = importlib.util.spec_from_file_location("download_ptbxl", _SCRIPT)
assert _spec and _spec.loader
_dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dl)


def test_ptbxl_db_slug_has_no_embedded_version() -> None:
    # wfdb.dl_files joins db + get_version(db); a version in the slug 404s.
    assert "/" not in _dl.PTBXL_DB
    assert _dl.PTBXL_DB == "ptb-xl"
