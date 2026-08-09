"""Local LLM client interface (Ollama / llama.cpp — never cloud APIs)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from physiorag.storage.base import StoredRecord


@dataclass(slots=True)
class SynthesisResult:
    answer: str
    sources: list[str]
    raw: dict[str, Any] | None = None


class LocalLLM(ABC):
    """Generate a natural-language answer from retrieved waveform context."""

    @abstractmethod
    def synthesize(
        self,
        query: str,
        records: Sequence[StoredRecord],
        *,
        temperature: float | None = None,
    ) -> SynthesisResult:
        """Produce an answer grounded in ``records`` (metadata/text only for MVP)."""
