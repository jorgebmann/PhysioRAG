"""PhysioRAG retrieval API (local-only)."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from physiorag import __version__
from physiorag.config import load_config
from physiorag.storage.array_store import ArrayStore
from physiorag.storage.factory import build_vector_store

# Module-level singletons populated in the lifespan handler.
_state: dict[str, Any] = {}


def _build_text_encoder(config: dict[str, Any]) -> Any | None:
    emb = config.get("embeddings", {})
    if not emb.get("text_encoder_enabled", True):
        return None
    try:
        from physiorag.embeddings.text_encoder import TextEncoder

        encoder = TextEncoder(
            model_name=str(emb.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2")),
            device=str(emb.get("device", "cpu")),
            embedding_dim=int(emb.get("text_embedding_dim", 384)),
        )
        encoder.encode_one("warmup")  # force model load now so /search is fast
        return encoder
    except Exception as exc:  # pragma: no cover - depends on local model cache
        print(f"[api] text encoder disabled ({type(exc).__name__}: {exc}); keyword search")
        return None


def _build_llm(config: dict[str, Any]) -> Any | None:
    try:
        from physiorag.synthesis.ollama import build_llm

        return build_llm(config)
    except Exception as exc:  # pragma: no cover
        print(f"[api] synthesis disabled ({type(exc).__name__}: {exc})")
        return None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    config = load_config()
    _state["config"] = config
    _state["store"] = build_vector_store(config)
    _state["arrays"] = ArrayStore(config.get("storage", {}).get("array_dir", "data/processed/arrays"))
    _state["text_encoder"] = _build_text_encoder(config)
    _state["llm"] = _build_llm(config)
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
    llm = _state.get("llm")
    return {
        "status": "ok",
        "version": __version__,
        "text_encoder": _state.get("text_encoder") is not None,
        "llm": bool(llm and llm.health()) if llm else False,
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Cross-modal search: NL query -> text vector (hybrid) over waveform epochs."""
    store = _state.get("store")
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    if not hasattr(store, "search_text"):
        raise HTTPException(status_code=501, detail="Store does not support text search")

    filters = {"modality": request.modality} if request.modality else None
    text_encoder = _state.get("text_encoder")
    query_embedding = text_encoder.encode_one(request.query) if text_encoder else None
    records = store.search_text(
        request.query,
        top_k=request.top_k,
        filters=filters,
        query_embedding=query_embedding,
    )

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
    if request.synthesize and llm is not None and llm.health():
        try:
            result = llm.synthesize(request.query, records)
            answer, sources = result.answer, result.sources
        except Exception as exc:  # pragma: no cover
            print(f"[api] synthesis failed ({type(exc).__name__}: {exc})")

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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ch = signal.shape[0]
    t = np.arange(signal.shape[1]) / fs
    fig, axes = plt.subplots(n_ch, 1, figsize=(9, 2.2 * n_ch), sharex=True, squeeze=False)
    for i in range(n_ch):
        ax = axes[i][0]
        ax.plot(t, signal[i], linewidth=0.9)
        ax.set_ylabel(channels[i])
        ax.grid(True, alpha=0.3)
    axes[-1][0].set_xlabel("time (s)")
    fig.suptitle(f"{record.epoch_id} — {record.modality}")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# Demo search widget (static HTML/JS). Mounted last so it never shadows the
# API routes defined above (Starlette matches routes in registration order).
_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
