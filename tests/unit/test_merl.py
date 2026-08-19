"""MERL wrapper: preprocessing, official key names, fail-loud loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from physiorag.embeddings.factory import available_encoders, build_encoder
from physiorag.embeddings.merl import (
    DEFAULT_PROJECTION_DIM,
    DEFAULT_TEXT_HIDDEN,
    _build_resnet18,
    _select_by_prefix,
    build_attention_pool,
    build_merl_encoder,
    build_merl_text_encoder,
    build_text_projection,
    prepare_ecg,
    prepare_query_text,
    resnet18_spatial_dim,
    swap_avl_avf,
)
from physiorag.ingestion.base import WaveformEpoch


def test_merl_is_registered() -> None:
    assert "merl" in available_encoders()


def test_merl_requires_checkpoint() -> None:
    with pytest.raises(ValueError) as exc:
        build_encoder({"embeddings": {"encoder": "merl"}})
    assert "checkpoint" in str(exc.value)


def test_prepare_ecg_pads_leads_and_length() -> None:
    signal = np.ones((2, 1250), dtype=np.float32)  # too few leads, too short
    out = prepare_ecg(signal, in_leads=12, target_len=5000)
    assert out.shape == (12, 5000)
    # Original leads preserved, extra leads zero-padded.
    assert np.allclose(out[0, :1250], 1.0)
    assert np.allclose(out[2], 0.0)
    assert np.allclose(out[0, 1250:], 0.0)


def test_prepare_ecg_truncates_oversized_input() -> None:
    signal = np.ones((15, 6000), dtype=np.float32)
    out = prepare_ecg(signal, in_leads=12, target_len=5000)
    assert out.shape == (12, 5000)


def test_prepare_ecg_fills_non_finite() -> None:
    signal = np.ones((1, 10), dtype=np.float32)
    signal[0, 5] = np.nan
    signal[0, 6] = np.inf
    out = prepare_ecg(signal, in_leads=1, target_len=10)
    assert np.all(np.isfinite(out))


def test_prepare_ecg_pads_short_duration_without_stretching() -> None:
    signal = np.zeros((12, 4000), dtype=np.float32)
    signal[0, -1] = 1.0
    out = prepare_ecg(
        signal, in_leads=12, target_len=5000, native_fs=500.0, target_fs=500.0
    )
    assert out.shape == (12, 5000)
    assert out[0, 3999] == pytest.approx(1.0)
    assert np.allclose(out[0, 4000:], 0.0)


def test_prepare_ecg_minmax_and_lead_swap() -> None:
    signal = np.zeros((12, 10), dtype=np.float32)
    signal[4] = 0.0  # aVL
    signal[5] = 1.0  # aVF
    out = prepare_ecg(signal, in_leads=12, target_len=10, minmax=True, swap_leads=True)
    # After swap, former aVF (1.0) is at index 4.
    assert out[4, 0] == pytest.approx(1.0)
    assert out[5, 0] == pytest.approx(0.0)


def test_swap_avl_avf_exchanges_rows() -> None:
    sig = np.zeros((12, 3), dtype=np.float32)
    sig[4] = 1.0
    sig[5] = 2.0
    out = swap_avl_avf(sig)
    assert np.allclose(out[4], 2.0)
    assert np.allclose(out[5], 1.0)


def test_prepare_query_text_lowercases_and_rewrites_ekg() -> None:
    assert prepare_query_text("Sinusrhythmus normales EKG") == "sinusrhythmus normales ecg"
    assert prepare_query_text("AF", lowercase=False) == "AF"


def test_resnet18_spatial_dim_matches_merl() -> None:
    assert resnet18_spatial_dim(5000) == 313


def test_select_by_prefix_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="empty checkpoint prefix"):
        _select_by_prefix({"ecg_encoder.conv1.weight": torch.zeros(1)}, ("",))


def _official_state_dict() -> dict[str, torch.Tensor]:
    """Random weights under the published ECGCLIP ResNet key names."""
    dim = DEFAULT_PROJECTION_DIM
    backbone = _build_resnet18(12)
    downconv = torch.nn.Conv1d(512, dim, kernel_size=1)
    att = build_attention_pool(spacial_dim=313, embed_dim=dim, num_heads=4, output_dim=dim)
    proj_t = build_text_projection(DEFAULT_TEXT_HIDDEN, dim, dim)
    state: dict[str, torch.Tensor] = {}
    state.update({f"ecg_encoder.{k}": v for k, v in backbone.state_dict().items()})
    state.update({f"downconv.{k}": v for k, v in downconv.state_dict().items()})
    state.update({f"att_pool_head.{k}": v for k, v in att.state_dict().items()})
    state.update({f"proj_t.{k}": v for k, v in proj_t.state_dict().items()})
    return state


def _write_ckpt(path: Path, state: dict[str, torch.Tensor]) -> Path:
    torch.save(state, path)
    return path


def _cfg(ckpt: Path) -> dict:
    return {
        "embeddings": {
            "encoder": "merl",
            "device": "cpu",
            "merl": {"checkpoint": str(ckpt), "device": "cpu"},
        }
    }


def test_merl_loads_official_key_names(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path / "res18_best_ckpt.pth", _official_state_dict())
    encoder = build_encoder(_cfg(ckpt))
    assert encoder.embedding_dim == 256

    epoch = WaveformEpoch(
        record_id="1",
        modality="ecg",
        start_time_s=0.0,
        sample_rate_hz=500.0,
        signal=np.zeros((12, 5000), dtype=np.float32),
    )
    vecs = encoder.encode([epoch])
    assert vecs.shape == (1, 256)
    assert np.all(np.isfinite(vecs))
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_merl_loads_module_prefixed_state_dict(tmp_path: Path) -> None:
    wrapped = {f"module.{k}": v for k, v in _official_state_dict().items()}
    ckpt = _write_ckpt(tmp_path / "ddp_ckpt.pth", wrapped)
    encoder = build_merl_encoder(_cfg(ckpt))
    assert encoder.embedding_dim == 256


def test_merl_rejects_encoder_only_checkpoint(tmp_path: Path) -> None:
    backbone = _build_resnet18(12)
    state = {f"ecg_encoder.{k}": v for k, v in backbone.state_dict().items()}
    ckpt = _write_ckpt(tmp_path / "res18_encoder.pth", state)
    with pytest.raises(RuntimeError, match="downconv / att_pool_head"):
        build_merl_encoder(_cfg(ckpt))


def test_merl_rejects_vit_checkpoint(tmp_path: Path) -> None:
    dim = DEFAULT_PROJECTION_DIM
    proj_e = build_text_projection(192, dim, dim)
    proj_t = build_text_projection(DEFAULT_TEXT_HIDDEN, dim, dim)
    state = {f"proj_e.{k}": v for k, v in proj_e.state_dict().items()}
    state.update({f"proj_t.{k}": v for k, v in proj_t.state_dict().items()})
    ckpt = _write_ckpt(tmp_path / "vit_tiny_ckpt.pth", state)
    with pytest.raises(RuntimeError, match="ViT"):
        build_merl_encoder(_cfg(ckpt))


def test_merl_rejects_missing_proj_t(tmp_path: Path) -> None:
    state = _official_state_dict()
    state = {k: v for k, v in state.items() if not k.startswith("proj_t.")}
    ckpt = _write_ckpt(tmp_path / "no_text.pth", state)
    with pytest.raises(RuntimeError, match="proj_t"):
        build_merl_text_encoder(_cfg(ckpt))


def test_empty_prefix_in_config_fails(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path / "res18_best_ckpt.pth", _official_state_dict())
    cfg = _cfg(ckpt)
    cfg["embeddings"]["merl"]["ecg_prefixes"] = [""]
    with pytest.raises(ValueError, match="empty checkpoint prefix"):
        build_merl_encoder(cfg)
