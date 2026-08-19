"""Unit tests for the vent retrieval eval helpers (no model / Weaviate needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval_vent_retrieval.py"
_spec = importlib.util.spec_from_file_location("eval_vent_retrieval", _SCRIPT)
assert _spec and _spec.loader
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)


def test_recall_at_k_labeled_membership() -> None:
    rankings = [["x", "z", "a"], ["b", "y", "z"]]
    gold_sets = [{"a"}, {"y", "z"}]
    result = _eval.recall_at_k(rankings, gold_sets, ks=[1, 2, 3])
    assert result[1] == 0.0  # neither gold is rank 1
    assert result[2] == 0.5  # only query 2 hits within top-2 (y)
    assert result[3] == 1.0  # query 1's gold "a" appears within top-3


def test_recall_at_k_skips_empty_gold() -> None:
    rankings = [["a"], ["b"]]
    gold_sets = [set(), {"b"}]
    result = _eval.recall_at_k(rankings, gold_sets, ks=[1])
    assert result[1] == 1.0  # only the second (non-empty) query counts


def test_chance_increases_with_k() -> None:
    chance = _eval.chance_at_k([1, 1], corpus_size=10, ks=[1, 5, 10])
    assert chance[1] < chance[5] < chance[10]
    assert chance[10] == 1.0  # drawing the whole corpus always includes the gold


def test_gold_sets_group_by_asynchrony_type() -> None:
    id_to_type = {
        "r1": "double_triggering",
        "r2": "double_triggering",
        "r3": "air_trapping",
    }
    queries = [
        {"query": "double triggering", "gold_type": "double_triggering"},
        {"query": "air trapping", "gold_type": "air_trapping"},
    ]
    golds = _eval.gold_sets_for(queries, id_to_type)
    assert golds[0] == {"r1", "r2"}
    assert golds[1] == {"r3"}


def test_keyword_ranking_prefers_more_overlap() -> None:
    ids = ["a", "b", "c"]
    haystacks = [
        "double triggering pressure spike",
        "air trapping auto peep",
        "normal controlled ventilation",
    ]
    ranked = _eval._keyword_ranking("double triggering", haystacks, ids)
    assert ranked[0] == "a"


def test_frozen_queries_have_corpus_backed_gold_types() -> None:
    from physiorag.ingestion.vent_captions import VENT_CATALOG

    catalog_types = {s["asynchrony_type"] for s in VENT_CATALOG}
    for q in _eval.FROZEN_QUERIES:
        assert q["gold_type"] in catalog_types


def test_frozen_queries_include_tagged_paraphrase_subset() -> None:
    from physiorag.ingestion.vent_captions import VENT_CATALOG

    subsets = {q.get("subset") for q in _eval.FROZEN_QUERIES}
    assert subsets == {"caption", "paraphrase"}
    catalog_types = {s["asynchrony_type"] for s in VENT_CATALOG}
    # Paraphrase queries must still map to a real asynchrony_type.
    assert _eval.PARAPHRASE_QUERIES
    for q in _eval.PARAPHRASE_QUERIES:
        assert q["gold_type"] in catalog_types


def test_distractor_epochs_are_never_gold() -> None:
    from physiorag.ingestion.vent_captions import VENT_CATALOG

    catalog_types = {s["asynchrony_type"] for s in VENT_CATALOG}
    epochs = _eval._distractor_epochs(variants=2)
    assert epochs
    for e in epochs:
        assert e.metadata["asynchrony_type"] not in catalog_types


def test_subset_recall_filters_by_subset() -> None:
    rankings = [["a"], ["b"], ["c"]]
    gold_sets = [{"a"}, {"x"}, {"c"}]
    subsets = ["caption", "paraphrase", "paraphrase"]
    # Only the two paraphrase rows count: query 1 misses ("x"), query 2 hits.
    result = _eval._subset_recall(rankings, gold_sets, subsets, "paraphrase", ks=[1])
    assert result[1] == 0.5


def test_format_report_labels_methods() -> None:
    text = _eval._format_report(
        {
            "k": [1, 3],
            "corpus_size": 16,
            "num_queries": 11,
            "hybrid": {1: 0.8, 3: 0.9},
            "minilm_only": {1: 0.6, 3: 0.8},
            "keyword_only": {1: 0.7, 3: 0.85},
            "chance": {1: 0.06, 3: 0.18},
        }
    )
    assert "Hybrid" in text
    assert "MiniLM-only" in text
    assert "Keyword-only" in text
    assert "Chance" in text
