#!/usr/bin/env python3
"""Text -> ventilator epoch Recall@k evaluation (Phase C, product metric).

This is **labeled-query retrieval**, not the self-retrieval used for ECG. A
frozen set of German/English Dräger-style queries is scored against the
synthetic vent demo corpus; a query is a hit@k when any epoch of its gold
``asynchrony_type`` appears in the top-k.

Honesty note: the ventilator path is ``hybrid_text`` — MiniLM over the bilingual
captions + keyword/BM25 over caption and metadata. It is **not** CLIP-style
text->signal retrieval (there is no matched vent text tower yet; ``baseline_cnn``
waveform vectors are random-init and only stored for a future ``signal_aligned``
swap). We therefore also report keyword-only and MiniLM-only so the hybrid
contribution is visible, plus chance.

Run (offline, in-memory; Weaviate not required; MiniLM must be cached):
    python scripts/eval_vent_retrieval.py --variants 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Frozen query set. ``gold_type`` must be an ``asynchrony_type`` present in the
# demo corpus. Each query has a ``subset``:
#   caption    -> phrased with the same vocabulary as the templated caption
#                 (keyword-only tends to do well; this is the "easy" set).
#   paraphrase -> describes the phenomenon WITHOUT copying caption keywords, so
#                 dense (MiniLM) matching has to carry the retrieval. This is the
#                 honest robustness signal.
# Mix of English and German (incl. the UI example chips) exercises the
# query-side glossary and the bilingual captions.
CAPTION_QUERIES: tuple[dict[str, str], ...] = (
    {
        "query": "ARDS patient breathing spontaneously against the ventilator causing a pressure spike",
        "lang": "en",
        "gold_type": "double_triggering",
    },
    {
        "query": "Patienten mit ARDS, die spontan gegen das Beatmungsgerät atmen und dadurch einen Druckanstieg verursachen",
        "lang": "de",
        "gold_type": "double_triggering",
    },
    {
        "query": "double triggering during pressure support",
        "lang": "en",
        "gold_type": "double_triggering",
    },
    {
        "query": "COPD patient with air trapping and rising end-expiratory pressure",
        "lang": "en",
        "gold_type": "air_trapping",
    },
    {
        "query": "COPD-Patient mit Air Trapping und steigendem endexspiratorischem Druck",
        "lang": "de",
        "gold_type": "air_trapping",
    },
    {
        "query": "ineffective triggering with a missed trigger",
        "lang": "en",
        "gold_type": "ineffective_effort",
    },
    {
        "query": "Frustrane Triggerung mit verpasstem Trigger",
        "lang": "de",
        "gold_type": "ineffective_effort",
    },
    {
        "query": "flow starvation and air hunger during a volume controlled breath",
        "lang": "en",
        "gold_type": "flow_starvation",
    },
    {
        "query": "Flussmangel und Lufthunger bei ARDS",
        "lang": "de",
        "gold_type": "flow_starvation",
    },
    {
        "query": "delayed cycling with prolonged inspiration",
        "lang": "en",
        "gold_type": "delayed_cycling",
    },
    {
        "query": "Verzögertes Umschalten mit verlängerter Inspiration",
        "lang": "de",
        "gold_type": "delayed_cycling",
    },
    {
        "query": "reverse triggering entrained by the mechanical breath",
        "lang": "en",
        "gold_type": "reverse_triggering",
    },
    {
        "query": "Reverse Triggering durch den maschinellen Atemzug ausgelöst",
        "lang": "de",
        "gold_type": "reverse_triggering",
    },
    {
        "query": "low compliance ARDS lungs with high peak pressures",
        "lang": "en",
        "gold_type": "low_compliance",
    },
    {
        "query": "stable controlled ventilation with no asynchrony",
        "lang": "en",
        "gold_type": "none",
    },
)

# Paraphrases that deliberately avoid the caption's own terminology.
PARAPHRASE_QUERIES: tuple[dict[str, str], ...] = (
    {
        "query": "the machine delivers two stacked inspirations from a single patient effort",
        "lang": "en",
        "gold_type": "double_triggering",
    },
    {
        "query": "gas is left behind in the lungs because exhalation never fully completes",
        "lang": "en",
        "gold_type": "air_trapping",
    },
    {
        "query": "a weak patient breath during exhalation that never opens the ventilator valve",
        "lang": "en",
        "gold_type": "ineffective_effort",
    },
    {
        "query": "the patient demands more gas than the preset delivery rate can supply",
        "lang": "en",
        "gold_type": "flow_starvation",
    },
    {
        "query": "the machine keeps insufflating long after the patient wants to breathe out",
        "lang": "en",
        "gold_type": "delayed_cycling",
    },
    {
        "query": "the mechanical breath drags a passive diaphragm into a late contraction",
        "lang": "en",
        "gold_type": "reverse_triggering",
    },
    {
        "query": "stiff lungs that need very high airway pressures for a small delivered volume",
        "lang": "en",
        "gold_type": "low_compliance",
    },
)


def _tag(queries: tuple[dict[str, str], ...], subset: str) -> tuple[dict[str, str], ...]:
    return tuple({**q, "subset": subset} for q in queries)


# Backward-compatible flat set (used by tests / introspection) with subset tags.
FROZEN_QUERIES: tuple[dict[str, str], ...] = _tag(CAPTION_QUERIES, "caption") + _tag(
    PARAPHRASE_QUERIES, "paraphrase"
)


def recall_at_k(
    rankings: Sequence[Sequence[str]],
    gold_sets: Sequence[set[str]],
    ks: Sequence[int],
) -> dict[int, float]:
    """Recall@k for labeled retrieval.

    ``rankings[i]`` is the ordered list of retrieved ids for query ``i``;
    ``gold_sets[i]`` is the set of acceptable ids. A query is a hit@k when the
    top-k of its ranking intersects its gold set. Queries with an empty gold set
    are skipped.
    """
    hits = {k: 0 for k in ks}
    counted = 0
    for ranked, gold in zip(rankings, gold_sets):
        if not gold:
            continue
        counted += 1
        for k in ks:
            if any(rid in gold for rid in ranked[:k]):
                hits[k] += 1
    return {k: (hits[k] / counted if counted else 0.0) for k in ks}


def chance_at_k(gold_sizes: Sequence[int], corpus_size: int, ks: Sequence[int]) -> dict[int, float]:
    """Expected Recall@k of a random ranking (mean over queries).

    For a query with ``g`` gold items in a corpus of ``N``, the chance of at
    least one gold in ``k`` random draws is ``1 - C(N-g, k) / C(N, k)``.
    """
    result: dict[int, float] = {}
    n = corpus_size
    for k in ks:
        if n <= 0:
            result[k] = 0.0
            continue
        probs = []
        kk = min(k, n)
        for g in gold_sizes:
            miss = 1.0
            if g <= 0:
                miss = 1.0
            elif n - g < kk:
                miss = 0.0
            else:
                miss = math.comb(n - g, kk) / math.comb(n, kk)
            probs.append(1.0 - miss)
        result[k] = float(np.mean(probs)) if probs else 0.0
    return result


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b @ a


def _keyword_ranking(query: str, haystacks: Sequence[str], ids: Sequence[str]) -> list[str]:
    tokens = {t.lower() for t in query.split() if t}
    scored = []
    for rid, hay in zip(ids, haystacks):
        score = sum(1 for tok in tokens if tok in hay)
        scored.append((score, rid))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [rid for _score, rid in scored]


# Unlabeled "controlled ventilation" distractors. They share no gold
# asynchrony_type, so they only add lexical competition (and lower chance).
# Signal noise alone does not stress caption/keyword retrieval; extra captions do.
_DISTRACTOR_CAPTIONS: tuple[str, ...] = (
    "Routine controlled ventilation with stable airway pressures and no asynchrony.",
    "Weaning trial on pressure support with comfortable spontaneous breaths.",
    "Post-operative patient on volume control, unremarkable pressure and flow.",
    "Stable SIMV with regular mandatory breaths and no triggering problems.",
)


def _distractor_epochs(variants: int) -> list[Any]:
    from types import SimpleNamespace

    out: list[Any] = []
    for v in range(max(1, variants)):
        for i, caption in enumerate(_DISTRACTOR_CAPTIONS):
            out.append(
                SimpleNamespace(
                    record_id=f"distractor-{v + 1}-{i + 1}",
                    metadata={
                        "text": caption,
                        "asynchrony_type": "distractor",
                        "diagnosis": "n/a",
                        "finding": "none",
                        "label": "distractor",
                    },
                )
            )
    return out


def build_corpus(cfg: dict[str, Any], *, variants: int, distractors: bool = True) -> list[Any]:
    from physiorag.ingestion.demo_processor import DemoWaveformProcessor

    ing = cfg.get("ingestion", {})
    processor = DemoWaveformProcessor(
        modality="ventilator",
        window_seconds=float(ing.get("window_seconds", 10.0)),
        sample_rate_hz=float(ing.get("target_sample_rate_hz", 125.0)),
        variants=variants,
    )
    # DemoWaveformProcessor synthesizes when the source dir has no .npy files.
    epochs = list(processor.iter_epochs(cfg.get("data", {}).get("demo_dir", "data/demo")))
    if distractors:
        epochs.extend(_distractor_epochs(variants))
    return epochs


def gold_sets_for(queries: Sequence[dict[str, str]], id_to_type: dict[str, str]) -> list[set[str]]:
    out: list[set[str]] = []
    for q in queries:
        gold_type = q["gold_type"]
        out.append({rid for rid, t in id_to_type.items() if t == gold_type})
    return out


def _subset_recall(
    rankings: Sequence[Sequence[str]],
    gold_sets: Sequence[set[str]],
    subsets: Sequence[str],
    want: str,
    ks: Sequence[int],
) -> dict[int, float]:
    idx = [i for i, s in enumerate(subsets) if s == want]
    return recall_at_k([rankings[i] for i in idx], [gold_sets[i] for i in idx], ks)


def _evaluate(
    epochs: list[Any],
    text_encoder: Any | None,
    ks: list[int],
    *,
    backend: str = "memory",
) -> dict[str, Any]:
    from physiorag.ingestion.vent_captions import apply_vent_glossary

    ids = [e.record_id for e in epochs]
    captions = [str(e.metadata.get("text", "")) for e in epochs]
    id_to_type = {e.record_id: str(e.metadata.get("asynchrony_type", "none")) for e in epochs}
    haystacks = [f"{cap} {e.metadata}".lower() for cap, e in zip(captions, epochs)]

    queries = list(FROZEN_QUERIES)
    subsets = [str(q.get("subset", "caption")) for q in queries]
    gold_sets = gold_sets_for(queries, id_to_type)
    gold_sizes = [len(g) for g in gold_sets]
    corpus_size = len(ids)

    # Keyword-only ranking (glossary applied so German maps onto English too).
    keyword_rankings = [
        _keyword_ranking(apply_vent_glossary(q["query"]), haystacks, ids) for q in queries
    ]

    minilm_rankings: list[list[str]] | None = None
    hybrid_rankings: list[list[str]] | None = None
    if text_encoder is not None:
        caption_vecs = text_encoder.encode(captions)
        minilm_rankings = []
        for q in queries:
            qvec = text_encoder.encode_one(apply_vent_glossary(q["query"]))
            sims = _cosine(np.asarray(qvec, dtype=np.float32), np.asarray(caption_vecs, dtype=np.float32))
            order = np.argsort(-sims)
            minilm_rankings.append([ids[j] for j in order])

        if backend == "weaviate":
            hybrid_rankings = _hybrid_rankings_weaviate(epochs, queries, text_encoder, corpus_size)
        else:
            hybrid_rankings = _hybrid_rankings(epochs, queries, text_encoder, corpus_size)

    chance = chance_at_k(gold_sizes, corpus_size, ks)

    def _method_block(rankings: list[list[str]] | None) -> dict[str, Any] | None:
        if rankings is None:
            return None
        return {
            "overall": recall_at_k(rankings, gold_sets, ks),
            "caption": _subset_recall(rankings, gold_sets, subsets, "caption", ks),
            "paraphrase": _subset_recall(rankings, gold_sets, subsets, "paraphrase", ks),
        }

    return {
        "corpus_size": corpus_size,
        "num_queries": len(queries),
        "k": ks,
        "backend": backend,
        "subset_counts": {
            "caption": subsets.count("caption"),
            "paraphrase": subsets.count("paraphrase"),
        },
        # Flat overall recall per method (quick read / back-compat).
        "hybrid": recall_at_k(hybrid_rankings, gold_sets, ks) if hybrid_rankings is not None else None,
        "minilm_only": recall_at_k(minilm_rankings, gold_sets, ks) if minilm_rankings is not None else None,
        "keyword_only": recall_at_k(keyword_rankings, gold_sets, ks),
        "chance": chance,
        # Recall split by query subset so the "easy" caption-copy queries do not
        # hide whether dense matching actually helps on paraphrases.
        "by_subset": {
            "hybrid": _method_block(hybrid_rankings),
            "minilm_only": _method_block(minilm_rankings),
            "keyword_only": _method_block(keyword_rankings),
        },
        "per_query": [
            {
                "query": q["query"],
                "lang": q["lang"],
                "subset": q.get("subset"),
                "gold_type": q["gold_type"],
                "gold_count": len(g),
            }
            for q, g in zip(queries, gold_sets)
        ],
    }


def _corpus_records(epochs: list[Any], text_encoder: Any, *, waveform_dim: int) -> list[Any]:
    from physiorag.storage.base import StoredRecord

    records = []
    for e in epochs:
        caption = str(e.metadata.get("text", ""))
        records.append(
            StoredRecord(
                record_id=e.record_id,
                epoch_id=e.record_id,
                modality="ventilator",
                embedding=np.zeros(waveform_dim, dtype=np.float32),
                array_ref="",
                metadata=e.metadata,
                text=caption,
                text_embedding=np.asarray(text_encoder.encode_one(caption), dtype=np.float32),
            )
        )
    return records


def _hybrid_rankings(
    epochs: list[Any], queries: Sequence[dict[str, str]], text_encoder: Any, top_k: int
) -> list[list[str]]:
    """Rank via the real product path: InMemoryVectorStore + Retriever (RRF)."""
    from physiorag.retrieval.search import Retriever
    from physiorag.storage.memory_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    store.upsert(_corpus_records(epochs, text_encoder, waveform_dim=1))
    retriever = Retriever(store, text_encoder=text_encoder, mode="hybrid_text")
    rankings: list[list[str]] = []
    for q in queries:
        hits = retriever.search_text(q["query"], top_k=top_k)
        rankings.append([h.record_id for h in hits])
    return rankings


def _hybrid_rankings_weaviate(
    epochs: list[Any], queries: Sequence[dict[str, str]], text_encoder: Any, top_k: int
) -> list[list[str]]:
    """Rank against a live Weaviate (same hybrid fusion the API uses).

    Uses a throwaway collection so the demo/default indexes are untouched.
    """
    from physiorag.retrieval.search import Retriever
    from physiorag.storage.weaviate_store import WeaviateVectorStore

    waveform_dim = 8
    text_dim = int(getattr(text_encoder, "embedding_dim", 384))
    store = WeaviateVectorStore(
        collection="WaveformEpochVentEval",
        waveform_dim=waveform_dim,
        text_dim=text_dim,
    )
    try:
        store.reset_schema()
        store.upsert(_corpus_records(epochs, text_encoder, waveform_dim=waveform_dim))
        retriever = Retriever(store, text_encoder=text_encoder, mode="hybrid_text")
        rankings: list[list[str]] = []
        for q in queries:
            hits = retriever.search_text(q["query"], top_k=top_k, filters={"modality": "ventilator"})
            rankings.append([h.record_id for h in hits])
        return rankings
    finally:
        try:
            if store._client.collections.exists(store.collection_name):
                store._client.collections.delete(store.collection_name)
        finally:
            store.close()


def _build_text_encoder(cfg: dict[str, Any]) -> Any | None:
    try:
        from physiorag.embeddings.text_factory import build_query_encoder

        enc = build_query_encoder(cfg)
        enc.encode_one("warmup")
        return enc
    except Exception as exc:  # pragma: no cover - model cache dependent
        print(f"[eval] MiniLM unavailable, keyword-only ({type(exc).__name__}: {exc})")
        return None


def _weaviate_available(cfg: dict[str, Any]) -> bool:  # pragma: no cover - needs live DB
    try:
        import weaviate
        from urllib.parse import urlparse

        url = cfg.get("storage", {}).get("weaviate_url", "http://127.0.0.1:8080")
        grpc = int(cfg.get("storage", {}).get("weaviate_grpc_port", 50051))
        parsed = urlparse(url)
        client = weaviate.connect_to_local(
            host=parsed.hostname or "127.0.0.1", port=parsed.port or 8080, grpc_port=grpc
        )
        ready = client.is_ready()
        client.close()
        return bool(ready)
    except Exception as exc:
        print(f"[eval] Weaviate not reachable ({type(exc).__name__}: {exc})")
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Text -> ventilator epoch Recall@k")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Synthetic epochs per scenario (more = more gold copies + distractors).",
    )
    parser.add_argument(
        "--backend",
        choices=["memory", "weaviate"],
        default="memory",
        help="Hybrid ranking backend. 'weaviate' needs a live instance (falls back to memory).",
    )
    parser.add_argument(
        "--no-distractors",
        action="store_true",
        help="Skip the unlabeled controlled-ventilation distractor epochs.",
    )
    parser.add_argument("--out", default=str(ROOT / "data" / "processed" / "vent_recall.json"))
    args = parser.parse_args(argv)

    from physiorag.config import load_config

    cfg = load_config(args.config)
    ks = sorted(set(args.k))

    print(f"[eval] building synthetic vent corpus (variants={args.variants}) ...")
    epochs = build_corpus(cfg, variants=args.variants, distractors=not args.no_distractors)
    if not epochs:
        print("[eval] no vent epochs generated.", file=sys.stderr)
        return 1

    text_encoder = _build_text_encoder(cfg)

    backend = args.backend
    if backend == "weaviate":
        if text_encoder is None:
            print("[eval] weaviate backend needs MiniLM; falling back to memory.")
            backend = "memory"
        elif not _weaviate_available(cfg):
            print("[eval] weaviate unavailable; falling back to memory backend.")
            backend = "memory"

    report = _evaluate(epochs, text_encoder, ks, backend=backend)
    report["config"] = args.config
    report["variants"] = args.variants

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(_format_report(report))
    print(f"[eval] wrote {out_path}")
    return 0


def _format_recall_lines(label: str, recalls: dict[Any, Any] | None, ks: list[int]) -> list[str]:
    if not recalls:
        return [f"{label} unavailable"]
    lines = [label]
    for k in ks:
        value = recalls.get(k, recalls.get(str(k)))
        if value is None:
            continue
        lines.append(f"    Recall@{k}: {float(value):.3f}")
    return lines


def _format_report(report: dict[str, Any]) -> str:
    ks = report["k"]
    backend = report.get("backend", "memory")
    lines = [
        "",
        f"Text -> ventilator epoch Recall@k (labeled asynchrony_type) [backend={backend}]",
        "-" * 56,
        f"corpus={report['corpus_size']} queries={report['num_queries']}",
    ]
    lines.extend(_format_recall_lines("Hybrid (MiniLM caption + keyword/BM25, RRF):", report.get("hybrid"), ks))
    lines.extend(_format_recall_lines("MiniLM-only (caption text vector):", report.get("minilm_only"), ks))
    lines.extend(_format_recall_lines("Keyword-only (caption + metadata):", report.get("keyword_only"), ks))
    lines.extend(
        _format_recall_lines(
            "Chance (random ranking; baseline_cnn cannot align text->signal):",
            report.get("chance"),
            ks,
        )
    )

    by_subset = report.get("by_subset")
    if by_subset:
        counts = report.get("subset_counts", {})
        lines.append("")
        lines.append(
            f"Paraphrase subset (queries avoid caption keywords; "
            f"n={counts.get('paraphrase', '?')}) -- the honest dense-vs-keyword signal:"
        )
        for label, key in (
            ("Hybrid", "hybrid"),
            ("MiniLM-only", "minilm_only"),
            ("Keyword-only", "keyword_only"),
        ):
            block = by_subset.get(key)
            para = block.get("paraphrase") if isinstance(block, dict) else None
            lines.extend(_format_recall_lines(f"  {label}:", para, ks))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
