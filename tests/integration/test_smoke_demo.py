"""Optional end-to-end demo smoke test.

Runs the real Phase A acceptance path (ingest -> Weaviate -> /health -> /search
-> plot -> Ollama). Requires Weaviate + Ollama + a cached MiniLM + pyturboquant,
so it is skipped unless PHYSIORAG_SMOKE=1 is set.

Set PHYSIORAG_SMOKE_DATASET=mimic_wdb to exercise the bounded WFDB path
(requires a local mirror from scripts/download_mimic_wdb.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.environ.get("PHYSIORAG_SMOKE") != "1",
    reason="set PHYSIORAG_SMOKE=1 (with Weaviate + Ollama running) to run the demo smoke test",
)
def test_smoke_demo_script_passes() -> None:
    dataset = os.environ.get("PHYSIORAG_SMOKE_DATASET", "mimic_demo")
    cmd = [sys.executable, str(ROOT / "scripts" / "smoke_demo.py"), "--dataset", dataset]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        },
    )
    assert result.returncode == 0, f"smoke failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
