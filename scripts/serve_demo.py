#!/usr/bin/env python3
"""One-command public demo: clone -> install -> search.

Serves PhysioRAG with the fully synthetic ventilator dataset in a single
process. No Docker, no PhysioNet credentials, no Ollama required:

    python scripts/serve_demo.py

Then open http://127.0.0.1:8000/ and try a query (or click an example chip).
Search results and waveform plots always work offline; if Ollama is running
with llama3.1, you also get a grounded, citation-backed answer.

Under the hood this points the API at configs/demo.yaml, which uses an in-memory
store and auto-ingests the synthetic scenarios on startup. The full local stack
(Weaviate + Ollama + real pyturboquant) still lives in scripts/smoke_demo.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEMO_CONFIG = ROOT / "configs" / "demo.yaml"

EXAMPLE_QUERIES = [
    "ARDS patient breathing spontaneously against the ventilator causing a pressure spike",
    "COPD-Patient mit Air Trapping und steigendem endexspiratorischem Druck",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default=None, help="Override host (default from config)")
    parser.add_argument("--port", type=int, default=None, help="Override port (default from config)")
    args = parser.parse_args(argv)

    os.environ["PHYSIORAG_CONFIG"] = str(DEMO_CONFIG)

    from physiorag.config import load_config

    config = load_config(DEMO_CONFIG)
    api_cfg = config.get("api", {})
    host = args.host or str(api_cfg.get("host", "127.0.0.1"))
    port = int(args.port if args.port is not None else api_cfg.get("port", 8000))

    ui_url = f"http://{host}:{port}/"
    print("=" * 70)
    print("PhysioRAG demo — synthetic ICU ventilator search (offline)")
    print("=" * 70)
    print(f"UI:        {ui_url}")
    print(f"API docs:  http://{host}:{port}/docs")
    print("Try a query in the UI, e.g.:")
    for query in EXAMPLE_QUERIES:
        print(f"  - {query}")
    print("(Ollama optional: search + plots work without it.)")
    print("=" * 70, flush=True)

    import uvicorn

    # No --reload: reload would re-run startup (and re-ingest) on every file
    # touch. The demo should come up once, deterministically.
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
