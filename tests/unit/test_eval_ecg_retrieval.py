"""Unit test for the Recall@k metric used by the ECG eval script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval_ecg_retrieval.py"
_spec = importlib.util.spec_from_file_location("eval_ecg_retrieval", _SCRIPT)
assert _spec and _spec.loader
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)


def test_recall_at_k_perfect_diagonal() -> None:
    sims = np.eye(3, dtype=np.float32)
    ids = ["a", "b", "c"]
    result = _eval.recall_at_k(sims, ids, ids, ks=[1, 2])
    assert result[1] == 1.0
    assert result[2] == 1.0


def test_recall_at_k_counts_topk_membership() -> None:
    # Query 0's gold ("a") is the 2nd best, query 1's gold ("b") is best.
    sims = np.asarray([[0.2, 0.9, 0.1], [0.1, 0.8, 0.0]], dtype=np.float32)
    query_ids = ["a", "b"]
    corpus_ids = ["a", "b", "c"]
    result = _eval.recall_at_k(sims, query_ids, corpus_ids, ks=[1, 2])
    assert result[1] == 0.5  # only query "b" hits at k=1
    assert result[2] == 1.0  # both hit within top-2


def test_cosine_sims_shape() -> None:
    q = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32)
    c = np.random.default_rng(1).normal(size=(5, 4)).astype(np.float32)
    sims = _eval._cosine_sims(q, c)
    assert sims.shape == (2, 5)


class _Epoch:
    def __init__(self, record_id: str, report: str, text: str = "", patient_id: str = "p") -> None:
        self.record_id = record_id
        self.metadata = {"report": report, "text": text, "patient_id": patient_id}


def test_filter_queries_requires_raw_report() -> None:
    epochs = [
        _Epoch("a", "sinus rhythm", text="sinus rhythm. normal ECG"),
        _Epoch("b", "", text="normal ECG"),
        _Epoch("c", "atrial fibrillation", text="atrial fibrillation. AF"),
    ]
    kept = _eval._filter_queries_with_reports(epochs, [0, 1, 2])
    assert kept == [0, 2]
    assert _eval._report_text(epochs[0]) == "sinus rhythm"


def test_apply_max_records_override() -> None:
    cfg = {"ingestion": {"max_records": 200}}
    _eval._apply_cli_overrides(cfg, max_records=50)
    assert cfg["ingestion"]["max_records"] == 50
    _eval._apply_cli_overrides(cfg, max_records=None)
    assert cfg["ingestion"]["max_records"] == 50


def test_main_max_records_reaches_config(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_collect(dataset, modality, used_cfg):
        captured["max_records"] = used_cfg.get("ingestion", {}).get("max_records")
        return []

    monkeypatch.setattr(_eval, "_collect_epochs", fake_collect)
    rc = _eval.main(
        [
            "--config",
            "configs/ecg_merl.yaml",
            "--max-records",
            "25",
            "--out",
            str(tmp_path / "ecg_recall.json"),
        ]
    )
    assert rc == 1
    assert captured["max_records"] == 25


def test_format_report_includes_quantized_and_labels_minilm() -> None:
    text = _eval._format_report(
        {
            "k": [1, 5],
            "merl_text_to_ecg": {1: 0.5, 5: 0.8},
            "merl_report_en_to_ecg": {1: 0.55, 5: 0.85},
            "merl_text_to_ecg_quantized": {1: 0.4, 5: 0.7},
            "merl_scp_to_ecg": {1: 0.6, 5: 0.9},
            "quantization": {"codec": "float16_stub", "stored_as": "float16_stub"},
            "baseline": {
                "chance": {1: 0.1, 5: 0.5},
                "minilm_text_to_text": {1: 0.2, 5: 0.6},
            },
        }
    )
    assert "report -> ECG signal index, float32" in text
    assert "glossary DE→EN report -> ECG" in text
    assert "ingest-quantized index" in text
    assert "SCP caption -> ECG" in text
    assert "NOT text->signal" in text
    assert "Chance" in text


def test_quantize_index_disabled() -> None:
    vecs = np.eye(3, dtype=np.float32)
    assert _eval._quantize_index(vecs, {"quantization": {"enabled": False}}) is None


def test_quantize_index_stub_or_real() -> None:
    vecs = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    result = _eval._quantize_index(vecs, {"quantization": {"enabled": True, "bits": 8}})
    assert result is not None
    data, meta = result
    assert data.shape == vecs.shape
    assert meta.get("codec") in {"pyturboquant", "float16_stub"}
