"""Ingestion pipeline: epochs -> encode (waveform+text) -> quantize -> store."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from physiorag.embeddings.baseline_cnn import BaselineCNNEncoder
from physiorag.embeddings.quantization import quantize_embeddings
from physiorag.ingestion.base import WaveformEpoch, WaveformProcessor
from physiorag.ingestion.demo_processor import DemoWaveformProcessor
from physiorag.ingestion.wfdb_processor import WfdbWaveformProcessor
from physiorag.storage.array_store import ArrayStore
from physiorag.storage.base import StoredRecord, VectorStore
from physiorag.storage.factory import build_vector_store


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


def _build_encoder(config: dict[str, Any]) -> BaselineCNNEncoder:
    emb = config.get("embeddings", {})
    name = emb.get("encoder", "baseline_cnn")
    if name != "baseline_cnn":
        raise ValueError(f"Unsupported encoder '{name}' in baseline ingest path")
    return BaselineCNNEncoder(
        embedding_dim=int(emb.get("embedding_dim", 128)),
        device=str(emb.get("device", "cpu")),
    )


def _maybe_build_text_encoder(config: dict[str, Any]) -> Any | None:
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
        print(f"[ingest] text encoder disabled ({type(exc).__name__}: {exc}); waveform-only")
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


def run_ingest(
    *,
    dataset: str = "mimic_wdb",
    modality: str = "ventilator",
    config: dict[str, Any] | None = None,
    store: VectorStore | None = None,
    text_encoder: Any | None = None,
) -> dict[str, Any]:
    """Run waveform ingestion into local array + vector storage."""
    from physiorag.config import load_config

    cfg = config if config is not None else load_config()
    source = _resolve_source(dataset, cfg)
    processor = _build_processor(dataset, modality, cfg)
    encoder = _build_encoder(cfg)
    if text_encoder is None:
        text_encoder = _maybe_build_text_encoder(cfg)
    array_dir = Path(cfg.get("storage", {}).get("array_dir", "data/processed/arrays"))
    arrays = ArrayStore(array_dir)
    owns_store = store is None
    vector_store = store if store is not None else build_vector_store(cfg)

    batch_size = int(cfg.get("embeddings", {}).get("batch_size", 16))
    quant_cfg = cfg.get("quantization", {})
    quant_enabled = bool(quant_cfg.get("enabled", True))
    quant_bits = int(quant_cfg.get("bits", 8))

    epochs_written = 0
    codec: str | None = None
    text_vectors_written = 0
    try:
        for batch in _batched(processor.iter_epochs(source), batch_size):
            dense = encoder.encode(batch)
            if quant_enabled:
                quantized = quantize_embeddings(dense, bits=quant_bits)
                codec = quantized.codec
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

    return {
        "status": "ok",
        "dataset": dataset,
        "modality": modality,
        "source": str(source),
        "epochs_written": epochs_written,
        "text_vectors_written": text_vectors_written,
        "array_dir": str(array_dir),
        "quant_codec": codec,
        "storage_backend": cfg.get("storage", {}).get("backend", "memory"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest physiological waveforms into PhysioRAG")
    parser.add_argument("--dataset", default="mimic_wdb")
    parser.add_argument(
        "--modality",
        default="ventilator",
        choices=["ventilator", "spo2", "ecg"],
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args(argv)

    from physiorag.config import load_config

    cfg = load_config(args.config)
    result = run_ingest(dataset=args.dataset, modality=args.modality, config=cfg)
    print(result)


if __name__ == "__main__":
    main()
