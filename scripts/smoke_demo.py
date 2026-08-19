#!/usr/bin/env python3
"""End-to-end smoke test for the PhysioRAG demo path.

Proves the full product path works end-to-end and air-gapped:

    ingest -> Weaviate -> /health -> /search -> plot -> Ollama answer

Phase A (ventilator):
    python scripts/smoke_demo.py
    python scripts/smoke_demo.py --dataset mimic_wdb

Phase B (ECG / MERL, isolated collection):
    python scripts/smoke_demo.py --dataset ptbxl --max-records 20

Requires Weaviate + Ollama. Vent also needs MiniLM + pyturboquant; ECG needs
the MERL checkpoint, Med-CPT cache, PTB-XL files, and pyturboquant.
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
ECG_DATASETS = {"ptbxl", "ptb-xl", "ptb_xl"}
DEFAULT_VENT_QUERY = (
    "ARDS patient breathing spontaneously against the ventilator causing a pressure spike"
)
DEFAULT_ECG_QUERY = "sinus rhythm"
DEFAULT_ECG_CONFIG = ROOT / "configs" / "ecg_merl.yaml"
DEFAULT_ECG_PLOT = ROOT / "data" / "processed" / "ecg_smoke.png"


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


def _prepare_config(config: dict, *, dataset: str, max_records: int | None) -> None:
    """Isolate demo vs WFDB collections so indexes never collide."""
    storage = config.setdefault("storage", {})
    if dataset == "mimic_demo":
        storage["collection"] = DEMO_COLLECTION
    if max_records is not None:
        config.setdefault("ingestion", {})["max_records"] = int(max_records)


def _require_wdb_source(config: dict) -> Path:
    from physiorag.pipeline.ingest import _resolve_source

    source = _resolve_source("mimic_wdb", config)
    if not source.exists():
        raise SmokeError(
            f"mimic_wdb source not found at {source}. "
            "Run: python scripts/download_mimic_wdb.py --max-records 3"
        )
    return source


def _require_ptbxl_source(config: dict) -> Path:
    from physiorag.pipeline.ingest import _resolve_source

    source = _resolve_source("ptbxl", config)
    if not (source / "ptbxl_database.csv").is_file():
        raise SmokeError(
            f"PTB-XL source not found at {source}. "
            "Run: python scripts/download_ptbxl.py --max-records 200"
        )
    return source


def _require_merl_checkpoint(config: dict) -> Path:
    rel = str(config.get("embeddings", {}).get("merl", {}).get("checkpoint") or "")
    ckpt = Path(rel)
    if not ckpt.is_file():
        raise SmokeError(
            f"MERL checkpoint missing at {ckpt or '(unset)'}. "
            "Download the full *_ckpt.pth (see README Phase B) and set "
            "embeddings.merl.checkpoint in configs/ecg_merl.yaml."
        )
    return ckpt


def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill ECG config/modality/query when ``--dataset ptbxl`` is used."""
    if args.dataset in ECG_DATASETS:
        if not args.config:
            args.config = str(DEFAULT_ECG_CONFIG)
        if args.modality == "ventilator":
            args.modality = "ecg"
        if args.query == DEFAULT_VENT_QUERY:
            args.query = DEFAULT_ECG_QUERY
        if args.plot_out is None:
            args.plot_out = str(DEFAULT_ECG_PLOT)
    return args


def _search_payload(args: argparse.Namespace, query: str, *, synthesize: bool) -> dict:
    body: dict = {"query": query, "top_k": 3, "synthesize": synthesize}
    if args.modality == "ecg":
        body["modality"] = "ecg"
    return body


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
    _prepare_config(config, dataset=args.dataset, max_records=args.max_records)
    if args.dataset in {"mimic_wdb", "mimic4wdb"}:
        _require_wdb_source(config)
    if args.dataset in ECG_DATASETS:
        _require_ptbxl_source(config)
        _require_merl_checkpoint(config)

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
        resp = client.post("/search", json=_search_payload(args, args.query, synthesize=True))
        if resp.status_code != 200:
            raise SmokeError(f"/search returned {resp.status_code}: {resp.text}")
        body = resp.json()
        hits = body.get("hits") or []
        if not hits:
            raise SmokeError("search returned no hits for the demo query")
        top = hits[0]
        _step(f"top hit: epoch_id={top['epoch_id']} score={top.get('score')} text={top.get('text')!r}")

        # 4. The plottable evidence must be fetchable as a PNG.
        meta = client.get(f"/waveforms/{top['epoch_id']}", params={"format": "json"})
        if meta.status_code != 200:
            raise SmokeError(f"waveform JSON failed: status={meta.status_code}")
        n_ch = len((meta.json() or {}).get("channels") or [])
        plot = client.get(f"/waveforms/{top['epoch_id']}", params={"format": "png"})
        if plot.status_code != 200 or plot.headers.get("content-type") != "image/png":
            raise SmokeError(f"plot fetch failed: status={plot.status_code}")
        from physiorag.plotting import png_wh

        width, height = png_wh(plot.content)
        _step(f"plot PNG {width}x{height}px channels={n_ch}")
        if n_ch == 12:
            if height >= 1400 or width <= height:
                raise SmokeError(
                    f"12-lead PNG should be a landscape 3×4 grid, got {width}x{height}"
                )
        if args.plot_out:
            out = Path(args.plot_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(plot.content)
            _step(f"wrote {out}")

        if args.dataset in ECG_DATASETS:
            _step("POST /search query='vorhofflimmern' (glossary, no synthesis)")
            de = client.post(
                "/search",
                json=_search_payload(args, "vorhofflimmern", synthesize=False),
            )
            if de.status_code != 200 or not (de.json().get("hits") or []):
                raise SmokeError(
                    f"German glossary query failed: status={de.status_code} body={de.text}"
                )

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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument(
        "--dataset",
        default="mimic_demo",
        choices=["mimic_demo", "mimic_wdb", "mimic4wdb", "ptbxl"],
        help="mimic_demo (synthetic), bounded mimic_wdb, or ptbxl (ECG / MERL).",
    )
    parser.add_argument("--modality", default="ventilator", choices=["ventilator", "spo2", "ecg"])
    parser.add_argument("--query", default=DEFAULT_VENT_QUERY)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap ingest studies (ECG smoke: 20 is enough to prove the path).",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help="Write the top-hit PNG here (ECG default: data/processed/ecg_smoke.png).",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require text encoder / Weaviate / pyturboquant / Ollama / offline flags (default: strict).",
    )
    args = apply_dataset_defaults(parser.parse_args(argv))

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
