"""Retrieval orchestration over a local vector store."""

from __future__ import annotations

from typing import Any

import numpy as np

from physiorag.storage.base import StoredRecord, VectorStore

# Retrieval modes:
#   hybrid_text    -> Phase A: encode query with the local text tower (MiniLM),
#                     search the aligned ``text`` named vector + BM25.
#   signal_aligned -> Phase B: encode query with a *matched* dual-encoder text
#                     tower (e.g. MERL) and nearest-neighbor the ``waveform``
#                     vectors directly (CLIP-style text -> signal retrieval).
HYBRID_TEXT = "hybrid_text"
SIGNAL_ALIGNED = "signal_aligned"


class Retriever:
    """Thin search facade; query encoding and retrieval mode stay pluggable."""

    def __init__(
        self,
        store: VectorStore,
        *,
        text_encoder: Any | None = None,
        mode: str = HYBRID_TEXT,
    ) -> None:
        self._store = store
        self._text_encoder = text_encoder
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[StoredRecord]:
        return self._store.search(query_embedding, top_k=top_k, filters=filters)

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[StoredRecord]:
        """Natural-language search.

        In ``signal_aligned`` mode the query is embedded with the matched text
        tower and compared against the waveform vectors directly. PTB-XL-style
        German/Swedish report tokens are rewritten to English first (same
        glossary as eval); already-English queries are left unchanged.
        Otherwise the text encoder feeds a cross-modal (text-vector) hybrid
        search, falling back to keyword search when no encoder is available.
        """
        if self._mode == SIGNAL_ALIGNED:
            if self._text_encoder is None:
                raise NotImplementedError(
                    "signal_aligned retrieval requires a matched query text encoder "
                    "(embeddings.text_encoder). None is configured."
                )
            from physiorag.ingestion.ptbxl_glossary import apply_ptbxl_glossary

            query_embedding = self._text_encoder.encode_one(apply_ptbxl_glossary(query))
            return self._store.search(query_embedding, top_k=top_k, filters=filters)

        from physiorag.ingestion.vent_captions import apply_vent_glossary

        # German UI queries (e.g. "Beatmungsgerät", "Druckanstieg") are rewritten
        # to English so both the MiniLM text vector and BM25 line up with the
        # English half of the bilingual captions/metadata. English queries and
        # non-vent vocabulary (incl. PTB-XL ECG German) pass through unchanged.
        # The vent glossary is only applied for ventilator (or unfiltered "all
        # modalities") queries so SpO2/ECG hybrid searches are never rewritten.
        modality = (filters or {}).get("modality")
        search_query = (
            apply_vent_glossary(query)
            if modality in (None, "", "ventilator")
            else query
        )
        query_embedding = None
        if self._text_encoder is not None:
            query_embedding = self._text_encoder.encode_one(search_query)
        if not hasattr(self._store, "search_text"):
            raise NotImplementedError("Store does not support text search")
        return self._store.search_text(  # type: ignore[attr-defined]
            search_query,
            top_k=top_k,
            filters=filters,
            query_embedding=query_embedding,
        )
