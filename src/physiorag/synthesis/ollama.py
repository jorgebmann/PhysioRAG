"""Local LLM synthesis via Ollama (no cloud APIs)."""

from __future__ import annotations

import re
from typing import Any, Sequence

import httpx

from physiorag.storage.base import StoredRecord
from physiorag.synthesis.llm import LocalLLM, SynthesisResult

SYSTEM_PROMPT = (
    "You are a clinical time-series retrieval assistant for a MedTech R&D tool. "
    "Answer ONLY from the provided retrieved waveform epochs (their text and "
    "metadata). Cite the epoch ids you used. If the context is empty or does not "
    "support an answer, say you found no matching evidence. Do not invent data. "
    "This is a research prototype, not for clinical decision-making. "
    "Always reply in the same language as the user's question, regardless of "
    "the language used in the retrieved epoch text or metadata."
)


def cited_epoch_ids(answer: str, candidates: Sequence[str]) -> list[str]:
    """Return candidate epoch ids that appear as substrings in ``answer`` (stable order)."""
    if not answer or not candidates:
        return []
    cited: list[str] = []
    for epoch_id in candidates:
        if epoch_id and re.search(re.escape(epoch_id), answer):
            cited.append(epoch_id)
    return cited


class OllamaLLM(LocalLLM):
    """Talk to a local Ollama server over HTTP."""

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s

    def health(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def _context(self, records: Sequence[StoredRecord]) -> str:
        lines: list[str] = []
        for rec in records:
            meta = {
                k: v
                for k, v in (rec.metadata or {}).items()
                if k not in {"_score"} and not k.startswith("_")
            }
            lines.append(
                f"- epoch_id={rec.epoch_id} | modality={rec.modality} | "
                f"text={rec.text or ''} | metadata={meta}"
            )
        return "\n".join(lines) if lines else "(no retrieved epochs)"

    def synthesize(
        self,
        query: str,
        records: Sequence[StoredRecord],
        *,
        temperature: float | None = None,
    ) -> SynthesisResult:
        candidates = [r.epoch_id for r in records]
        if not records:
            return SynthesisResult(
                answer="No matching waveform evidence was found for this query.",
                sources=[],
            )
        prompt = (
            f"Question: {query}\n\n"
            f"Retrieved waveform epochs:\n{self._context(records)}\n\n"
            "Write a concise answer grounded in the epochs above and cite epoch ids. "
            "Respond in the same language as the question above."
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature
            },
        }
        resp = httpx.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s
        )
        resp.raise_for_status()
        data = resp.json()
        answer = str(data.get("response", "")).strip()
        sources = cited_epoch_ids(answer, candidates)
        return SynthesisResult(
            answer=answer,
            sources=sources,
            raw=data,
        )


def build_llm(config: dict[str, Any]) -> LocalLLM | None:
    """Build a LocalLLM from config, or None if synthesis is disabled."""
    syn = config.get("synthesis", {})
    if not syn.get("enabled", True):
        return None
    backend = syn.get("backend", "ollama")
    if backend != "ollama":
        raise ValueError(f"Unsupported synthesis backend: {backend}")
    return OllamaLLM(
        model=str(syn.get("model", "llama3.1")),
        base_url=str(syn.get("base_url", "http://127.0.0.1:11434")),
        temperature=float(syn.get("temperature", 0.2)),
    )
