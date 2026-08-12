#!/usr/bin/env python3
"""Phase A acceptance smoke test for the PhysioRAG demo path.

Proves the full product path works end-to-end and air-gapped:

    ingest -> Weaviate -> /health -> /search -> plot -> Ollama answer

Supports ``mimic_demo`` (default) and bounded ``mimic_wdb`` when a local mirror
exists. Runs in strict mode by default, so a broken offline setup (missing
MiniLM cache, Weaviate down, no pyturboquant, Ollama unavailable) fails loudly
instead of silently degrading. Exits non-zero with an actionable message on any
failure.

Usage:
    # Requires: Weaviate on :8080, Ollama running with the configured model,
    # MiniLM cached locally, and `pip install ".[quant]"`.
    python scripts/smoke_demo.py
    python scripts/smoke_demo.py --dataset mimic_wdb
    python scripts/smoke_demo.py --no-strict          # allow soft degrade
    python scripts/smoke_demo.py --query "air trapping in COPD"

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

DEMO_COLLECTION = "WaveformEpochDemo"
OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


class SmokeError(RuntimeError):
    """Raised when a smoke-test step fails its assertion."""


def _step(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def _ensure_offline_flags(*, strict: bool) -> None:
    missing = [name for name in OFFLINE_ENV_VARS if not os.environ.get(name)]
    if not missing:
        return
    if strict:
        raise SmokeError(
            "strict smoke requires air-gap env vars: "
            + ", ".join(f"{n}=1" for n in missing)
            + ". Export them (see README) or re-run with --no-strict."
        )
    for name in missing:
        os.environ[name] = "1"
        _step(f"set {name}=1 for soft-offline smoke")


def _prepare_config(config: dict, *, dataset: str) -> None:
    """Isolate demo vs WFDB collections so indexes never collide."""
    storage = config.setdefault("storage", {})
    if dataset == "mimic_demo":
        storage["collection"] = DEMO_COLLECTION
    # mimic_wdb keeps configs/default.yaml collection (WaveformEpochV2).


def _require_wdb_source(config: dict) -> Path:
    from physiorag.pipeline.ingest import _resolve_source

    source = _resolve_source("mimic_wdb", config)
    if not source.exists():
        raise SmokeError(
            f"mimic_wdb source not found at {source}. "
            "Run: python scripts/download_mimic_wdb.py --max-records 3"
        )
    return source


def _run(args: argparse.Namespace) -> None:
    strict = None if args.strict is None else bool(args.strict)
    # Default to strict so the acceptance run cannot pass while degraded.
    if strict is None:
        os.environ.setdefault("PHYSIORAG_STRICT", "1")
        strict = True

    _ensure_offline_flags(strict=strict)

    from fastapi.testclient import TestClient

    from physiorag.config import load_config
    from physiorag.pipeline.ingest import _format_summary, run_ingest

    config = load_config(args.config)
    _prepare_config(config, dataset=args.dataset)
    if args.dataset in {"mimic_wdb", "mimic4wdb"}:
        _require_wdb_source(config)

    backend = config.get("storage", {}).get("backend", "memory")
    collection = config.get("storage", {}).get("collection")
    _step(
        f"strict={strict} dataset={args.dataset} "
        f"storage_backend={backend} collection={collection}"
    )

    # 1. Ingest (reset the collection for a clean run).
    _step(f"ingesting dataset={args.dataset} modality={args.modality} (reset_collection=True)")
    result = run_ingest(
        dataset=args.dataset,
        modality=args.modality,
        config=config,
        strict=strict,
        reset_collection=True,
    )
    print(_format_summary(result))
    if result["epochs_written"] < 1:
        raise SmokeError("ingest wrote 0 epochs; expected at least one epoch")
    if strict and result["quant_codec"] != "pyturboquant":
        raise SmokeError(
            f"quant codec is {result['quant_codec']!r}, expected 'pyturboquant' "
            "under strict mode (pip install '.[quant]')"
        )
    stats = result.get("quant_stats") or {}
    if strict and stats:
        if float(stats.get("compression_ratio", 0)) < 1.0:
            raise SmokeError(f"expected compression_ratio >= 1, got {stats}")
        _step(
            f"quant_stats ratio={stats.get('compression_ratio')} "
            f"mean_cosine={stats.get('mean_cosine')} "
            f"compressed_bytes={stats.get('compressed_bytes')}"
        )
    if args.dataset == "mimic_demo" and result["text_vectors_written"] < 1:
        raise SmokeError("no text vectors written; cross-modal search would be keyword-only")

    # 2. Bring up the API against the same store and validate health.
    import api.main as api_main

    # Ensure the TestClient lifespan loads the same collection override.
    def _load_smoke_config(*_a, **_k):
        return config

    import physiorag.config as config_mod

    # Patch both the API import site and the config module used at lifespan.
    api_main.load_config = _load_smoke_config  # type: ignore[assignment]
    config_mod.load_config = _load_smoke_config  # type: ignore[assignment]

    with TestClient(api_main.app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise SmokeError(f"/health returned {health.status_code}")
        hbody = health.json()
        _step(f"/health -> {hbody}")
        if not hbody.get("store_ok"):
            raise SmokeError("vector store not healthy")
        if not hbody.get("text_encoder"):
            raise SmokeError("text encoder not active; cross-modal search unavailable")
        if not hbody.get("llm"):
            raise SmokeError("LLM (Ollama) not healthy; grounded synthesis unavailable")
        if strict and not hbody.get("quant_available"):
            raise SmokeError("pyturboquant.core not available under strict smoke")

        # 3. Cross-modal search should return plottable hits + a grounded answer.
        _step(f"POST /search query={args.query!r}")
        resp = client.post("/search", json={"query": args.query, "top_k": 3})
        if resp.status_code != 200:
            raise SmokeError(f"/search returned {resp.status_code}: {resp.text}")
        body = resp.json()
        hits = body.get("hits") or []
        if not hits:
            raise SmokeError("search returned no hits for the demo query")
        top = hits[0]
        _step(f"top hit: epoch_id={top['epoch_id']} score={top.get('score')} text={top.get('text')!r}")

        # 4. The plottable evidence must be fetchable as a PNG.
        plot = client.get(f"/waveforms/{top['epoch_id']}", params={"format": "png"})
        if plot.status_code != 200 or plot.headers.get("content-type") != "image/png":
            raise SmokeError(f"plot fetch failed: status={plot.status_code}")

        # 5. The answer must be grounded and cite epoch ids in the answer text.
        answer = body.get("answer")
        sources = body.get("sources") or []
        if not answer:
            raise SmokeError("no synthesized answer returned (Ollama down or synthesis disabled)")
        hit_ids = {h["epoch_id"] for h in hits}
        cited_in_answer = [eid for eid in hit_ids if eid in answer]
        if not cited_in_answer:
            raise SmokeError(
                f"answer does not cite any retrieved epoch id; answer={answer!r} hits={hit_ids}"
            )
        if not (set(sources) & hit_ids):
            raise SmokeError(
                f"answer sources {sources} do not cite any retrieved epoch id "
                "(sources are filtered to ids present in the answer text)"
            )
        _step(f"answer cites: {sources}")
        print("\n--- synthesized answer ---")
        print(answer)
        print("--------------------------")

    print(f"\n[smoke] PASS: {args.dataset} path works end-to-end.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument(
        "--dataset",
        default="mimic_demo",
        choices=["mimic_demo", "mimic_wdb", "mimic4wdb"],
        help="mimic_demo (synthetic) or bounded mimic_wdb (requires local mirror).",
    )
    parser.add_argument("--modality", default="ventilator", choices=["ventilator", "spo2", "ecg"])
    parser.add_argument(
        "--query",
        default="ARDS patient breathing spontaneously against the ventilator causing a pressure spike",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require text encoder / Weaviate / pyturboquant / Ollama / offline flags (default: strict).",
    )
    args = parser.parse_args(argv)

    try:
        _run(args)
    except SmokeError as exc:
        print(f"\n[smoke] FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - environment/setup failures
        print(f"\n[smoke] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
