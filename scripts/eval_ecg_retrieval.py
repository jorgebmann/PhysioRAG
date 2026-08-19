#!/usr/bin/env python3
"""Text -> ECG Recall@k evaluation (Phase B, corpus RAG).

Measures whether a natural-language query (the ECG's **raw diagnostic report**)
retrieves *that ECG's signal* from the index — the product metric from the brief
(text → waveform Recall@k), not zero-shot classification AUC.

Methods on the **same** frozen, patient-level query list (corpus = all loaded
epochs so the gold ECG is in the index; ``--eval-fraction`` only selects which
patients contribute queries):

* ``merl`` report → ECG — product metric: raw diagnostic report vs signal index.
* ``merl`` glossary DE→EN report → ECG — same reports after longest-match
  PTB-XL phrase replace (no new encoder). Isolates German/Swedish vs English.
* ``merl`` quantized — same (raw) report queries on ingest-quantized index vectors.
* ``merl`` SCP caption → ECG — English SCP descriptions (closer to MERL zeroshot
  prompts) vs the same signal index. Diagnoses preprocessing vs language mismatch.
* ``baseline`` MiniLM — text→text only: query = report, index = SCP caption.
* chance — ``k / corpus_size``.

Run (offline, in-memory; Weaviate not required):
    python scripts/eval_ecg_retrieval.py --config configs/ecg_merl.yaml \
        --dataset ptbxl --max-records 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def recall_at_k(
    sims: np.ndarray,
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    ks: Sequence[int],
) -> dict[int, float]:
    """Recall@k for self-retrieval.

    ``sims`` is ``(n_queries, n_corpus)`` similarity. A query is a hit@k if its
    own id (``query_ids[i]``) appears within the top-k corpus ids by similarity.
    """
    corpus_index = {cid: j for j, cid in enumerate(corpus_ids)}
    max_k = max(ks)
    order = np.argsort(-sims, axis=1)[:, :max_k]
    hits = {k: 0 for k in ks}
    counted = 0
    for i, qid in enumerate(query_ids):
        gold = corpus_index.get(qid)
        if gold is None:
            continue
        counted += 1
        ranked = order[i]
        for k in ks:
            if gold in ranked[:k]:
                hits[k] += 1
    return {k: (hits[k] / counted if counted else 0.0) for k in ks}


def _cosine_sims(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    return q @ c.T


def _report_text(epoch: Any) -> str:
    return str((epoch.metadata or {}).get("report", "") or "").strip()


def _filter_queries_with_reports(epochs: list[Any], query_idx: list[int]) -> list[int]:
    """Keep only query epochs that have a raw report (no SCP-caption fallback)."""
    return [i for i in query_idx if _report_text(epochs[i])]


def _collect_epochs(dataset: str, modality: str, cfg: dict[str, Any]) -> list[Any]:
    from physiorag.pipeline.ingest import _build_processor, _resolve_source

    source = _resolve_source(dataset, cfg)
    processor = _build_processor(dataset, modality, cfg)
    return list(processor.iter_epochs(source))


def _patient_query_mask(epochs: list[Any], *, eval_fraction: float, seed: int) -> list[int]:
    """Return indices of epochs whose patient falls in the frozen eval split.

    The corpus is not reduced: gold ECGs stay in the index. ``eval_fraction``
    only chooses which patients issue queries (patient-level, no subject leak
    into a train split — there is no training here).
    """
    patients = sorted({str(e.metadata.get("patient_id", e.record_id)) for e in epochs})
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_eval = max(1, int(round(len(patients) * eval_fraction)))
    eval_patients = set(patients[:n_eval])
    return [
        i
        for i, e in enumerate(epochs)
        if str(e.metadata.get("patient_id", e.record_id)) in eval_patients
    ]


def _encode_signals(encoder: Any, epochs: list[Any], batch_size: int) -> np.ndarray:
    vecs: list[np.ndarray] = []
    for start in range(0, len(epochs), batch_size):
        vecs.append(encoder.encode(epochs[start : start + batch_size]))
    return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, encoder.embedding_dim))


def _quantize_index(vectors: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Apply the ingest quantization path; return reconstructed index + codec info."""
    quant_cfg = cfg.get("quantization", {})
    if not bool(quant_cfg.get("enabled", True)):
        return None
    from physiorag.embeddings.quantization import quantize_embeddings

    bits = int(quant_cfg.get("bits", 8))
    quantized = quantize_embeddings(vectors, bits=bits)
    data = np.asarray(quantized.data, dtype=np.float32)
    meta = {"codec": quantized.codec, "bits": quantized.bits}
    if quantized.metadata:
        meta.update(
            {
                k: quantized.metadata[k]
                for k in ("mean_cosine", "mse", "stored_as", "compression_ratio")
                if k in quantized.metadata
            }
        )
    return data, meta


def _eval_merl(
    cfg: dict[str, Any],
    epochs: list[Any],
    query_idx: list[int],
    ks: list[int],
    batch_size: int,
) -> dict[str, Any] | None:
    from physiorag.embeddings.merl import build_merl_encoder, build_merl_text_encoder
    from physiorag.ingestion.ptbxl_glossary import translate_ptbxl_report

    try:
        sig_encoder = build_merl_encoder(cfg)
        txt_encoder = build_merl_text_encoder(cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[eval] MERL unavailable, skipping ({type(exc).__name__}: {exc})")
        return None

    corpus_vecs = _encode_signals(sig_encoder, epochs, batch_size)
    query_ids = [epochs[i].record_id for i in query_idx]
    corpus_ids = [e.record_id for e in epochs]
    report_texts = [_report_text(epochs[i]) for i in query_idx]
    report_en_texts = [translate_ptbxl_report(t) for t in report_texts]
    scp_texts = [_scp_caption(epochs[i]) for i in query_idx]
    report_vecs = txt_encoder.encode(report_texts)
    float32 = recall_at_k(_cosine_sims(report_vecs, corpus_vecs), query_ids, corpus_ids, ks)
    report_en = recall_at_k(
        _cosine_sims(txt_encoder.encode(report_en_texts), corpus_vecs),
        query_ids,
        corpus_ids,
        ks,
    )
    scp = recall_at_k(
        _cosine_sims(txt_encoder.encode(scp_texts), corpus_vecs),
        query_ids,
        corpus_ids,
        ks,
    )

    quantized_recalls = None
    quant_meta = None
    quantized = _quantize_index(corpus_vecs, cfg)
    if quantized is not None:
        q_vecs, quant_meta = quantized
        quantized_recalls = recall_at_k(
            _cosine_sims(report_vecs, q_vecs), query_ids, corpus_ids, ks
        )

    return {
        "query_field": "report",
        "float32": float32,
        "report_en": report_en,
        "quantized": quantized_recalls,
        "scp_caption": scp,
        "quantization": quant_meta,
    }


def _eval_baseline(
    cfg: dict[str, Any],
    epochs: list[Any],
    query_idx: list[int],
    ks: list[int],
) -> dict[str, Any]:
    corpus_ids = [e.record_id for e in epochs]
    query_ids = [epochs[i].record_id for i in query_idx]
    n_corpus = len(corpus_ids)
    chance = {k: (min(k, n_corpus) / n_corpus if n_corpus else 0.0) for k in ks}

    # MiniLM text->text reference: index = SCP caption, query = free-text report.
    try:
        from physiorag.embeddings.text_encoder import TextEncoder

        text_encoder = TextEncoder(
            model_name=str(
                cfg.get("embeddings", {}).get(
                    "baseline_text_encoder", "sentence-transformers/all-MiniLM-L6-v2"
                )
            )
        )
        index_texts = [_scp_caption(e) for e in epochs]
        query_texts = [_report_text(epochs[i]) for i in query_idx]
        idx_vecs = text_encoder.encode(index_texts)
        q_vecs = text_encoder.encode(query_texts)
        sims = _cosine_sims(q_vecs, idx_vecs)
        minilm = recall_at_k(sims, query_ids, corpus_ids, ks)
    except Exception as exc:  # pragma: no cover - model cache dependent
        print(f"[eval] MiniLM baseline unavailable ({type(exc).__name__}: {exc})")
        minilm = None

    return {"chance": chance, "minilm_text_to_text": minilm}


def _scp_caption(epoch: Any) -> str:
    text = str(epoch.metadata.get("text", "")).strip()
    report = str(epoch.metadata.get("report", "")).strip()
    # Prefer the SCP portion (after the report) when both are present.
    if report and text.startswith(report):
        remainder = text[len(report) :].strip(". ").strip()
        if remainder:
            return remainder
    return text or report or "12-lead ECG"


def _apply_cli_overrides(cfg: dict[str, Any], *, max_records: int | None) -> dict[str, Any]:
    if max_records is not None:
        cfg.setdefault("ingestion", {})["max_records"] = int(max_records)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Text->ECG Recall@k evaluation")
    parser.add_argument("--config", default=str(ROOT / "configs" / "ecg_merl.yaml"))
    parser.add_argument("--dataset", default="ptbxl")
    parser.add_argument("--modality", default="ecg")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap PTB-XL studies (overrides ingestion.max_records in the config).",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=1.0,
        help="Fraction of patients whose reports are used as queries. "
        "Corpus stays the full loaded set (gold ECG remains in the index). "
        "Default 1.0 = all patients.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default=str(ROOT / "data" / "processed" / "ecg_recall.json"))
    args = parser.parse_args(argv)

    from physiorag.config import load_config

    cfg = _apply_cli_overrides(load_config(args.config), max_records=args.max_records)
    batch_size = int(cfg.get("embeddings", {}).get("batch_size", 16))

    print(f"[eval] loading {args.dataset} epochs ...")
    epochs = _collect_epochs(args.dataset, args.modality, cfg)
    if not epochs:
        print("[eval] no epochs found; download a PTB-XL subset first.", file=sys.stderr)
        return 1

    query_idx = _patient_query_mask(epochs, eval_fraction=args.eval_fraction, seed=args.seed)
    query_idx = _filter_queries_with_reports(epochs, query_idx)
    if not query_idx:
        print(
            "[eval] no queries with a raw report; need ptbxl_database.csv captions.",
            file=sys.stderr,
        )
        return 1

    ks = sorted(set(args.k))
    print(
        f"[eval] corpus={len(epochs)} queries={len(query_idx)} "
        f"(report + glossary DE→EN + SCP-caption queries, "
        f"eval_fraction={args.eval_fraction}) k={ks}"
    )

    merl = _eval_merl(cfg, epochs, query_idx, ks, batch_size)
    baseline = _eval_baseline(cfg, epochs, query_idx, ks)

    report = {
        "dataset": args.dataset,
        "corpus_size": len(epochs),
        "num_queries": len(query_idx),
        "query_field": "report",
        "eval_fraction": args.eval_fraction,
        "max_records": cfg.get("ingestion", {}).get("max_records"),
        "k": ks,
        "merl_text_to_ecg": (merl or {}).get("float32") if merl else None,
        "merl_report_en_to_ecg": (merl or {}).get("report_en") if merl else None,
        "merl_text_to_ecg_quantized": (merl or {}).get("quantized") if merl else None,
        "merl_scp_to_ecg": (merl or {}).get("scp_caption") if merl else None,
        "quantization": (merl or {}).get("quantization") if merl else None,
        "baseline": baseline,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(_format_report(report))
    print(f"[eval] wrote {out_path}")
    return 0


def _format_recall_lines(label: str, recalls: dict[Any, Any] | None, ks: list[int]) -> list[str]:
    if not recalls:
        return []
    lines = [label]
    for k in ks:
        value = recalls.get(k, recalls.get(str(k)))
        if value is None:
            continue
        lines.append(f"    Recall@{k}: {float(value):.3f}")
    return lines


def _format_report(report: dict[str, Any]) -> str:
    lines = ["", "Text -> ECG Recall@k", "-" * 32]
    ks = report["k"]
    merl = report.get("merl_text_to_ecg")
    merl_en = report.get("merl_report_en_to_ecg")
    merl_q = report.get("merl_text_to_ecg_quantized")
    merl_scp = report.get("merl_scp_to_ecg")
    base = report.get("baseline", {})
    quant = report.get("quantization") or {}
    if merl:
        lines.extend(
            _format_recall_lines("MERL (report -> ECG signal index, float32):", merl, ks)
        )
    else:
        lines.append("MERL: unavailable (no checkpoint). See README Phase B.")
    if merl_en:
        lines.extend(
            _format_recall_lines(
                "MERL (glossary DE→EN report -> ECG signal index):",
                merl_en,
                ks,
            )
        )
    if merl_q:
        codec = quant.get("codec", "quantized")
        stored = quant.get("stored_as", "")
        suffix = f", {codec}" + (f" stored as {stored}" if stored else "")
        lines.extend(
            _format_recall_lines(
                f"MERL (same report queries, ingest-quantized index{suffix}):",
                merl_q,
                ks,
            )
        )
    if merl_scp:
        lines.extend(
            _format_recall_lines(
                "MERL (English SCP caption -> ECG signal index):",
                merl_scp,
                ks,
            )
        )
    if base.get("minilm_text_to_text"):
        lines.extend(
            _format_recall_lines(
                "Baseline MiniLM (report -> SCP-caption index; NOT text->signal):",
                base["minilm_text_to_text"],
                ks,
            )
        )
    lines.extend(
        _format_recall_lines(
            "Chance (k / corpus_size; baseline_cnn cannot align text->signal):",
            base.get("chance"),
            ks,
        )
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
