"""Citation filtering for grounded synthesis answers."""

from __future__ import annotations

from physiorag.synthesis.ollama import cited_epoch_ids


def test_cited_epoch_ids_filters_to_answer_mentions() -> None:
    answer = "See demo-ards-001_0 for the pressure spike; ignore others."
    cited = cited_epoch_ids(answer, ["demo-ards-001_0", "demo-copd-002_0", "missing"])
    assert cited == ["demo-ards-001_0"]


def test_cited_epoch_ids_empty_when_none_mentioned() -> None:
    assert cited_epoch_ids("No evidence found.", ["demo-ards-001_0"]) == []
    assert cited_epoch_ids("", ["demo-ards-001_0"]) == []
    assert cited_epoch_ids("demo-ards-001_0", []) == []
