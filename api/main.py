"""PhysioRAG retrieval API (local-only)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from physiorag import __version__
from physiorag.config import load_config
from physiorag.embeddings.quantization import has_pyturboquant_core
from physiorag.retrieval.search import Retriever
from physiorag.runtime import is_strict
from physiorag.storage.array_store import ArrayStore
from physiorag.storage.factory import build_vector_store

# Module-level singletons populated in the lifespan handler.
_state: dict[str, Any] = {}


def _build_store(config: dict[str, Any]) -> Any:
    try:
        return build_vector_store(config)
    except Exception as exc:
        raise RuntimeError(
            f"[api] failed to initialize vector store ({type(exc).__name__}: {exc}). "
            "Is the backend (e.g. Weaviate) running and reachable?"
        ) from exc


def _build_text_encoder(config: dict[str, Any], *, strict: bool) -> Any | None:
    emb = config.get("embeddings", {})
    if not emb.get("text_encoder_enabled", True):
        return None
    try:
        from physiorag.embeddings.text_factory import build_query_encoder

        encoder = build_query_encoder(config)
        encoder.encode_one("warmup")  # force model load now so /search is fast
        return encoder
    except Exception as exc:  # pragma: no cover - depends on local model cache
        detail = f"{type(exc).__name__}: {exc}"
        if strict:
            raise RuntimeError(
                f"[api] strict mode: text encoder failed to load ({detail}). "
                "Pre-cache the model (see README air-gapped install) or set "
                "embeddings.text_encoder_enabled: false."
            ) from exc
        print(f"[api] text encoder disabled ({detail}); keyword search")
        return None


def _build_llm(config: dict[str, Any], *, strict: bool) -> Any | None:
    syn = config.get("synthesis", {})
    enabled = bool(syn.get("enabled", True))
    try:
        from physiorag.synthesis.ollama import build_llm

        llm = build_llm(config)
    except Exception as exc:  # pragma: no cover
        detail = f"{type(exc).__name__}: {exc}"
        if strict and enabled:
            raise RuntimeError(f"[api] strict mode: synthesis failed to initialize ({detail}).") from exc
        print(f"[api] synthesis disabled ({detail})")
        return None
    if strict and enabled and (llm is None or not llm.health()):
        base_url = syn.get("base_url", "http://127.0.0.1:11434")
        raise RuntimeError(
            f"[api] strict mode: Ollama is not healthy at {base_url}. "
            "Start Ollama and pull the model, or set synthesis.enabled: false."
        )
    return llm


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    config = load_config()
    strict = is_strict(config)
    _state["config"] = config
    _state["strict"] = strict
    store = _build_store(config)
    _state["store"] = store
    _state["arrays"] = ArrayStore(config.get("storage", {}).get("array_dir", "data/processed/arrays"))
    text_encoder = _build_text_encoder(config, strict=strict)
    _state["text_encoder"] = text_encoder
    _state["llm"] = _build_llm(config, strict=strict)
    mode = str(config.get("retrieval", {}).get("mode", "hybrid_text"))
    _state["retriever"] = Retriever(store, text_encoder=text_encoder, mode=mode)
    try:
        yield
    finally:
        store = _state.get("store")
        if store is not None and hasattr(store, "close"):
            store.close()
        _state.clear()


app = FastAPI(
    title="PhysioRAG",
    description="Offline multi-modal RAG for physiological time-series",
    version=__version__,
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language clinical query")
    top_k: int = Field(5, ge=1, le=50)
    modality: str | None = Field(None, description="Optional filter: ventilator | spo2 | ecg")
    synthesize: bool = Field(True, description="Generate a grounded LLM answer if available")


class SearchHit(BaseModel):
    epoch_id: str
    record_id: str
    modality: str
    score: float | None = None
    array_ref: str | None = None
    text: str | None = None
    plot_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    answer: str | None = None
    sources: list[str] = Field(default_factory=list)
    status: str = "ok"


class WaveformResponse(BaseModel):
    epoch_id: str
    record_id: str
    modality: str
    sample_rate_hz: float
    channels: list[str]
    signal: list[list[float]]


@app.get("/health")
def health() -> dict[str, Any]:
    config = _state.get("config", {})
    emb = config.get("embeddings", {})
    syn = config.get("synthesis", {})
    storage = config.get("storage", {})
    quant = config.get("quantization", {})

    llm = _state.get("llm")
    store = _state.get("store")
    text_ok = _state.get("text_encoder") is not None
    llm_ok = bool(llm and llm.health()) if llm else False
    store_ok = bool(store and store.health()) if store is not None else False
    quant_ok = has_pyturboquant_core()

    text_enabled = bool(emb.get("text_encoder_enabled", True))
    syn_enabled = bool(syn.get("enabled", True))
    quant_enabled = bool(quant.get("enabled", True))
    strict = bool(_state.get("strict", False))
    # "degraded" when an enabled runtime component is missing. Quant absence
    # only degrades under strict mode (non-strict may use float16_stub).
    degraded = (
        (text_enabled and not text_ok)
        or (syn_enabled and not llm_ok)
        or (not store_ok)
        or (strict and quant_enabled and not quant_ok)
    )

    return {
        "status": "degraded" if degraded else "ok",
        "version": __version__,
        "text_encoder": text_ok,
        "llm": llm_ok,
        "store": storage.get("backend", "memory"),
        "store_ok": store_ok,
        "collection": storage.get("collection"),
        "quant_available": quant_ok,
        "strict": strict,
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Cross-modal search: NL query -> text vector (hybrid) over waveform epochs."""
    retriever = _state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    filters = {"modality": request.modality} if request.modality else None
    try:
        records = retriever.search_text(
            request.query,
            top_k=request.top_k,
            filters=filters,
        )
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Store does not support text search")

    hits = [
        SearchHit(
            epoch_id=r.epoch_id,
            record_id=r.record_id,
            modality=r.modality,
            score=(r.metadata or {}).get("_score"),
            array_ref=r.array_ref,
            text=r.text,
            plot_url=f"/waveforms/{r.epoch_id}?format=png",
        )
        for r in records
    ]

    answer: str | None = None
    sources: list[str] = []
    llm = _state.get("llm")
    strict = bool(_state.get("strict", False))
    if request.synthesize:
        if llm is None or not llm.health():
            if strict:
                raise HTTPException(
                    status_code=503,
                    detail="Strict mode: synthesis requested but Ollama is unavailable.",
                )
        else:
            try:
                result = llm.synthesize(request.query, records)
                answer, sources = result.answer, result.sources
            except Exception as exc:  # pragma: no cover
                detail = f"{type(exc).__name__}: {exc}"
                if strict:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Strict mode: synthesis failed ({detail}).",
                    ) from exc
                print(f"[api] synthesis failed ({detail})")

    return SearchResponse(query=request.query, hits=hits, answer=answer, sources=sources)


def _load_epoch(epoch_id: str) -> tuple[Any, np.ndarray]:
    store = _state.get("store")
    arrays: ArrayStore | None = _state.get("arrays")
    if store is None or arrays is None:
        raise HTTPException(status_code=503, detail="Store not initialized")
    record = store.get_by_epoch_id(epoch_id)
    if record is None or not record.array_ref:
        raise HTTPException(status_code=404, detail=f"Unknown epoch_id: {epoch_id}")
    try:
        signal = arrays.load(record.array_ref)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Array file missing")
    signal = np.atleast_2d(np.asarray(signal, dtype=np.float32))
    return record, signal


def _channel_names(record: Any, n_channels: int) -> list[str]:
    names = (record.metadata or {}).get("channels")
    if isinstance(names, list) and len(names) == n_channels:
        return [str(n) for n in names]
    return [f"ch{i}" for i in range(n_channels)]


@app.get("/waveforms/{epoch_id}")
def get_waveform(
    epoch_id: str,
    format: str = Query("json", pattern="^(json|png)$"),
) -> Any:
    """Return the plottable array for an epoch as JSON samples or a PNG plot."""
    record, signal = _load_epoch(epoch_id)
    fs = float((record.metadata or {}).get("sample_rate_hz", 0.0)) or 1.0
    channels = _channel_names(record, signal.shape[0])

    if format == "json":
        return WaveformResponse(
            epoch_id=record.epoch_id,
            record_id=record.record_id,
            modality=record.modality,
            sample_rate_hz=fs,
            channels=channels,
            signal=signal.tolist(),
        )

    png = _render_png(signal, channels, fs, record)
    return Response(content=png, media_type="image/png")


def _render_png(signal: np.ndarray, channels: list[str], fs: float, record: Any) -> bytes:
    from physiorag.plotting import render_waveform_png

    return render_waveform_png(
        signal,
        channels=channels,
        sample_rate_hz=fs,
        title=f"{record.epoch_id} — {record.modality}",
    )


# Demo search widget (static HTML/JS). Mounted last so it never shadows the
# API routes defined above (Starlette matches routes in registration order).
_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
