#!/usr/bin/env python3
"""CLI wrapper matching the README quickstart.

Usage:
    python scripts/ingest_waveforms.py --dataset mimic_demo --modality ventilator
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without an editable install during early scaffolding.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physiorag.pipeline.ingest import main

if __name__ == "__main__":
    main()
