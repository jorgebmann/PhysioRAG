"""Ingestion pipeline: epochs -> encode (waveform+text) -> quantize -> store."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from physiorag.embeddings.factory import build_encoder
from physiorag.embeddings.quantization import is_real_quant_codec, quantize_embeddings
from physiorag.ingestion.base import WaveformEpoch, WaveformProcessor
from physiorag.ingestion.demo_processor import DemoWaveformProcessor
from physiorag.ingestion.wfdb_processor import WfdbWaveformProcessor
from physiorag.runtime import is_strict
from physiorag.storage.array_store import ArrayStore
from physiorag.storage.base import StoredRecord, VectorStore
from physiorag.storage.factory import build_vector_store

DEMO_COLLECTION = "WaveformEpochDemo"


def _resolve_source(dataset: str, config: dict[str, Any]) -> Path:
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    if dataset == "mimic_demo":
        return Path(data_cfg.get("demo_dir", "data/demo"))
    if dataset in {"mimic_wdb", "mimic4wdb"}:
        subdir = config.get("physionet", {}).get("mirror_subdir", "mimic4wdb")
        return raw_dir / subdir
    return raw_dir / dataset


def _build_processor(dataset: str, modality: str, config: dict[str, Any]) -> WaveformProcessor:
    ing = config.get("ingestion", {})
    window_seconds = float(ing.get("window_seconds", 10.0))
    target_fs = float(ing.get("target_sample_rate_hz", 125.0))
    if dataset == "mimic_demo":
        return DemoWaveformProcessor(
            modality=modality,
            window_seconds=window_seconds,
            sample_rate_hz=target_fs,
        )
    return WfdbWaveformProcessor(
        modality=modality,
        window_seconds=window_seconds,
        target_sample_rate_hz=target_fs,
        max_epochs_per_record=int(ing.get("max_epochs_per_record", 20)),
    )


def _maybe_build_text_encoder(config: dict[str, Any], *, strict: bool) -> Any | None:
    emb = config.get("embeddings", {})
    if not emb.get("text_encoder_enabled", True):
        return None
    try:
        from physiorag.embeddings.text_encoder import TextEncoder

        return TextEncoder(
            model_name=str(emb.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2")),
            device=str(emb.get("device", "cpu")),
            embedding_dim=int(emb.get("text_embedding_dim", 384)),
        )
    except Exception as exc:  # pragma: no cover - depends on local model cache
        detail = f"{type(exc).__name__}: {exc}"
        if strict:
            raise RuntimeError(
                f"[ingest] strict mode: text encoder failed to load ({detail}). "
                "Pre-cache the model (see README air-gapped install) or set "
                "embeddings.text_encoder_enabled: false."
            ) from exc
        print(f"[ingest] text encoder disabled ({detail}); waveform-only")
        return None


def _batched(epochs: Iterator[WaveformEpoch], batch_size: int) -> Iterator[list[WaveformEpoch]]:
    batch: list[WaveformEpoch] = []
    for epoch in epochs:
        batch.append(epoch)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _epoch_text(epoch: WaveformEpoch) -> str | None:
    if not epoch.metadata:
        return None
    return epoch.metadata.get("text") or epoch.metadata.get("label")


def _accumulate_quant_stats(
    totals: dict[str, float],
    *,
    n_vectors: int,
    meta: dict[str, Any] | None,
) -> None:
    if not meta:
        return
    totals["batches"] += 1
    totals["vectors"] += n_vectors
    totals["original_bytes"] += float(meta.get("original_bytes", 0) or 0)
    totals["compressed_bytes"] += float(meta.get("compressed_bytes", 0) or 0)
    # Weighted by vectors in the batch for running means.
    totals["_cosine_sum"] += float(meta.get("mean_cosine", 0.0) or 0.0) * n_vectors
    totals["_mse_sum"] += float(meta.get("mse", 0.0) or 0.0) * n_vectors


def _finalize_quant_stats(totals: dict[str, float], codec: str | None) -> dict[str, Any] | None:
    if not totals["vectors"]:
        return None
    n = totals["vectors"]
    original = int(totals["original_bytes"])
    compressed = int(totals["compressed_bytes"])
    ratio = (original / compressed) if compressed else 0.0
    return {
        "codec": codec,
        "vectors": int(n),
        "original_bytes": original,
        "compressed_bytes": compressed,
        "compression_ratio": round(ratio, 3),
        "mean_cosine": round(totals["_cosine_sum"] / n, 6),
        "mse": totals["_mse_sum"] / n,
        "stored_as": "float32_reconstruction"
        if is_real_quant_codec(codec)
        else ("float16_stub" if codec == "float16_stub" else "float32"),
    }


def run_ingest(
    *,
    dataset: str = "mimic_wdb",
    modality: str = "ventilator",
    config: dict[str, Any] | None = None,
    store: VectorStore | None = None,
    text_encoder: Any | None = None,
    strict: bool | None = None,
    reset_collection: bool = False,
) -> dict[str, Any]:
    """Run waveform ingestion into local array + vector storage."""
    from physiorag.config import load_config

    cfg = config if config is not None else load_config()
    strict_mode = is_strict(cfg, override=strict)
    source = _resolve_source(dataset, cfg)
    processor = _build_processor(dataset, modality, cfg)
    encoder = build_encoder(cfg)
    if text_encoder is None:
        text_encoder = _maybe_build_text_encoder(cfg, strict=strict_mode)
    array_dir = Path(cfg.get("storage", {}).get("array_dir", "data/processed/arrays"))
    arrays = ArrayStore(array_dir)
    owns_store = store is None
    vector_store = store if store is not None else build_vector_store(cfg)
    if reset_collection and hasattr(vector_store, "reset_schema"):
        vector_store.reset_schema()  # type: ignore[attr-defined]

    batch_size = int(cfg.get("embeddings", {}).get("batch_size", 16))
    quant_cfg = cfg.get("quantization", {})
    quant_enabled = bool(quant_cfg.get("enabled", True))
    quant_bits = int(quant_cfg.get("bits", 8))

    epochs_written = 0
    codec: str | None = None
    text_vectors_written = 0
    quant_totals: dict[str, float] = {
        "batches": 0,
        "vectors": 0,
        "original_bytes": 0,
        "compressed_bytes": 0,
        "_cosine_sum": 0.0,
        "_mse_sum": 0.0,
    }
    try:
        for batch in _batched(processor.iter_epochs(source), batch_size):
            dense = encoder.encode(batch)
            if quant_enabled:
                quantized = quantize_embeddings(dense, bits=quant_bits)
                codec = quantized.codec
                if strict_mode and not is_real_quant_codec(codec):
                    raise RuntimeError(
                        f"[ingest] strict mode: quantization codec is '{codec}', not a real "
                        "TurboQuant backend. Install with "
                        'pip install ".[quant]" (pyturboquant>=0.1.1 provides pyturboquant.core) '
                        "or set quantization.enabled: false."
                    )
                _accumulate_quant_stats(
                    quant_totals, n_vectors=dense.shape[0], meta=quantized.metadata
                )
                vectors = np.asarray(quantized.data, dtype=np.float32)
            else:
                vectors = dense
                codec = "float32"

            texts = [_epoch_text(e) for e in batch]
            text_vecs: np.ndarray | None = None
            if text_encoder is not None:
                encodable = [t if t else "" for t in texts]
                text_vecs = text_encoder.encode(encodable)

            records: list[StoredRecord] = []
            for i, epoch in enumerate(batch):
                epoch_id = f"{epoch.record_id}_{int(epoch.start_time_s * 1000)}"
                array_ref = arrays.save(epoch_id, epoch.signal)
                text = texts[i]
                text_embedding = None
                if text_vecs is not None and text:
                    text_embedding = text_vecs[i]
                    text_vectors_written += 1
                records.append(
                    StoredRecord(
                        record_id=epoch.record_id,
                        epoch_id=epoch_id,
                        modality=epoch.modality,
                        embedding=vectors[i],
                        array_ref=array_ref,
                        metadata={
                            **epoch.metadata,
                            "start_time_s": epoch.start_time_s,
                            "sample_rate_hz": epoch.sample_rate_hz,
                            "quant_codec": codec,
                        },
                        text=str(text) if text else None,
                        text_embedding=text_embedding,
                    )
                )
            epochs_written += vector_store.upsert(records)
    finally:
        if owns_store and hasattr(vector_store, "close"):
            vector_store.close()  # type: ignore[attr-defined]

    storage_cfg = cfg.get("storage", {})
    quant_stats = _finalize_quant_stats(quant_totals, codec)
    return {
        "status": "ok",
        "dataset": dataset,
        "modality": modality,
        "source": str(source),
        "epochs_written": epochs_written,
        "text_vectors_written": text_vectors_written,
        "text_encoder": text_encoder is not None,
        "array_dir": str(array_dir),
        "quant_codec": codec,
        "quant_stats": quant_stats,
        "storage_backend": storage_cfg.get("backend", "memory"),
        "collection": storage_cfg.get("collection"),
        "strict": strict_mode,
    }


def _format_summary(result: dict[str, Any]) -> str:
    keys = [
        "dataset",
        "modality",
        "source",
        "epochs_written",
        "text_vectors_written",
        "text_encoder",
        "quant_codec",
        "storage_backend",
        "collection",
        "strict",
    ]
    lines = ["PhysioRAG ingest complete:"]
    lines.extend(f"  {key}: {result.get(key)}" for key in keys)
    stats = result.get("quant_stats")
    if isinstance(stats, dict):
        lines.append("  quant_stats:")
        for key in (
            "original_bytes",
            "compressed_bytes",
            "compression_ratio",
            "mean_cosine",
            "mse",
            "stored_as",
        ):
            if key in stats:
                lines.append(f"    {key}: {stats[key]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest physiological waveforms into PhysioRAG")
    parser.add_argument("--dataset", default="mimic_wdb")
    parser.add_argument(
        "--modality",
        default="ventilator",
        choices=["ventilator", "spo2", "ecg"],
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail loudly if text encoder / vector store / pyturboquant are unavailable "
        "(overrides runtime.strict and PHYSIORAG_STRICT).",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Drop and recreate the vector store collection before ingesting "
        "(use when embedding dims change).",
    )
    args = parser.parse_args(argv)

    from physiorag.config import load_config

    cfg = load_config(args.config)
    # Keep curated demo scenarios in their own collection so they never share
    # an index with the larger mimic_wdb corpus (default is WaveformEpochV2).
    if args.dataset == "mimic_demo":
        storage = cfg.setdefault("storage", {})
        if storage.get("collection") in {None, "", "WaveformEpochV2"}:
            storage["collection"] = DEMO_COLLECTION
    result = run_ingest(
        dataset=args.dataset,
        modality=args.modality,
        config=cfg,
        strict=args.strict,
        reset_collection=args.reset_collection,
    )
    print(_format_summary(result))


if __name__ == "__main__":
    main()
