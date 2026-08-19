"""Reused ECG dual-encoder backend (MERL) for PhysioRAG (Phase B).

Wraps the open-source **MERL** ECG-language model
([Liu et al., ICML 2024](https://arxiv.org/abs/2403.06659),
`cheliu-computation/MERL-ICML2024`) so its two towers plug into the existing
PhysioRAG interfaces *without any training*.

The architecture here is a faithful copy of MERL's ResNet18 ``ECGCLIP`` path
(``utils/resnet1d.py`` + ``utils/utils_builder.py``), not a generic 1D-ResNet:

* :class:`MerlEcgEncoder` — ``ecg_encoder`` (ResNet18, no stem max-pool) →
  ``downconv`` (1×1, 512→256) → ``att_pool_head`` (AttentionPool2d, L=313 for
  10 s @ 500 Hz). That vector is the ``waveform`` index.
* :class:`MerlTextEncoder` — Med-CPT ``pooler_output`` → ``proj_t`` so queries
  land in the same 256-d space. Retrieval uses ``signal_aligned``.

Checkpoints must be the full ``*_ckpt.pth`` (both towers + CLIP heads).
``*_encoder.pth`` is ECG-backbone only and is rejected. ViT MERL checkpoints
(``proj_e``, no ``downconv``) are not supported by this wrapper.

Weights load with ``strict=True``. Missing keys fail loudly with a sample of
checkpoint key names. Confirm the MERL license before redistributing weights.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from physiorag.embeddings.base import TimeSeriesEncoder
from physiorag.ingestion.base import WaveformEpoch

logger = logging.getLogger(__name__)

# Conventional 12-lead order for plots / PTB-XL. The MERL encoder swaps aVL/aVF
# to MIMIC order (aVF before aVL) at encode time.
STANDARD_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)
DEFAULT_TEXT_MODEL = "ncbi/MedCPT-Query-Encoder"
DEFAULT_PROJECTION_DIM = 256
DEFAULT_IN_LEADS = 12
DEFAULT_SIGNAL_LENGTH = 5000  # 10 s @ 500 Hz
DEFAULT_SAMPLE_RATE_HZ = 500.0
DEFAULT_TEXT_HIDDEN = 768  # Med-CPT / MERL proj_t in_features
DEFAULT_ATT_HEADS = 4
# Conventional 12-lead: aVL=4, aVF=5. MIMIC/MERL stores aVF then aVL.
AVL_LEAD_INDEX = 4
AVF_LEAD_INDEX = 5

# Official ECGCLIP key prefixes (after stripping a leading ``module.``).
ECG_PREFIXES: tuple[str, ...] = ("ecg_encoder.",)
DOWNCONV_PREFIXES: tuple[str, ...] = ("downconv.",)
ATT_POOL_PREFIXES: tuple[str, ...] = ("att_pool_head.",)
TEXT_PROJ_PREFIXES: tuple[str, ...] = ("proj_t.",)


# --------------------------------------------------------------------------- #
# MERL ResNet18 (utils/resnet1d.py) — residual blocks named ``shortcut``
# --------------------------------------------------------------------------- #
def resnet18_spatial_dim(signal_length: int) -> int:
    """Spatial length after MERL ResNet18 for a 1-D input of ``signal_length``.

    conv1 stride 2, then layer2/3/4 stride 2 (no stem max-pool). Official
    5000-sample inputs yield 313, which is ``AttentionPool2d.spacial_dim``.
    """
    n = int(signal_length)
    n = (n + 2 * 3 - 7) // 2 + 1  # conv1: k=7, s=2, p=3
    for _ in range(3):  # layer2, layer3, layer4
        n = (n + 2 * 1 - 3) // 2 + 1
    return n


def _build_resnet18(in_channels: int = DEFAULT_IN_LEADS) -> Any:
    import torch
    import torch.nn as nn

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != self.expansion * out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        self.expansion * out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self.expansion * out_channels),
                )

        def forward(self, x: Any) -> Any:
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            return torch.relu(out)

    class ResNet(nn.Module):
        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.in_channels = 64
            self.out_dim = 512
            self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
            self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
            self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
            # Present in MERL ResNet so ``ecg_encoder.linear.*`` keys load; unused in CLIP forward.
            self.linear = nn.Linear(512, 10)
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        def _make_layer(self, block: Any, out_channels: int, num_blocks: int, stride: int):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for block_stride in strides:
                layers.append(block(self.in_channels, out_channels, block_stride))
                self.in_channels = out_channels * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            return out  # (B, 512, L) — MERL does not pool here

    return ResNet(in_channels)


def build_attention_pool(
    *,
    spacial_dim: int,
    embed_dim: int,
    num_heads: int = DEFAULT_ATT_HEADS,
    output_dim: int | None = None,
) -> Any:
    """MERL ``AttentionPool2d`` (parameter names must match ``att_pool_head.*``)."""
    import torch
    import torch.nn as nn

    class _AttentionPool2d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            out_dim = output_dim or embed_dim
            self.positional_embedding = nn.Parameter(
                torch.randn(1, spacial_dim + 1, embed_dim) / embed_dim
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.c_proj = nn.Linear(embed_dim, out_dim)
            self.num_heads = num_heads

        def forward(self, x: Any) -> tuple[Any, Any]:
            # x: (B, C, L) → (B, L, C)
            x = x.permute(0, 2, 1)
            cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
            cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.positional_embedding.to(dtype=x.dtype)
            x, att_map = self.mhsa(x[:, :1, :], x, x, average_attn_weights=True)
            x = self.c_proj(x)
            return x.squeeze(0), att_map[:, :, 1:]

    return _AttentionPool2d()


def build_text_projection(in_dim: int, hidden: int, out_dim: int) -> Any:
    """MERL ``proj_t``: Linear → GELU → Linear (Sequential indices 0 and 2)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _sample_keys(state: dict[str, Any], n: int = 16) -> str:
    keys = [k for k in state if isinstance(k, str)][:n]
    return ", ".join(keys) if keys else "(none)"


def _load_state_dict(checkpoint: str | Path) -> dict[str, Any]:
    import torch

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"MERL checkpoint not found at '{path}'. Download the full "
            "'*_ckpt.pth' (both towers) from the MERL release and set "
            "embeddings.merl.checkpoint to its local path (see README Phase B)."
        )
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"[merl] failed to load '{path}' with weights_only=True ({type(exc).__name__}: {exc}). "
            "The file must be a tensor state dict, not a pickled Python object."
        ) from exc
    if not isinstance(obj, dict):
        raise TypeError(
            f"[merl] checkpoint at '{path}' is a {type(obj).__name__}, not a state dict. "
            "Use the official '*_ckpt.pth' file."
        )
    for key in ("model", "state_dict", "model_state_dict"):
        inner = obj.get(key)
        if isinstance(inner, dict) and inner and _looks_like_state_dict(inner):
            obj = inner
            break
    tensor_state = {k: v for k, v in obj.items() if hasattr(v, "shape")}
    if not tensor_state:
        raise RuntimeError(
            f"[merl] no tensor weights in '{path}'. Sample keys: {_sample_keys(obj)}"
        )
    if all(k.startswith("module.") for k in tensor_state):
        tensor_state = {k[len("module.") :]: v for k, v in tensor_state.items()}
    return tensor_state


def _looks_like_state_dict(obj: dict[str, Any]) -> bool:
    return any(hasattr(v, "shape") for v in obj.values())


def _select_by_prefix(state: dict[str, Any], prefixes: Sequence[str]) -> dict[str, Any]:
    """Return keys under the first prefix that matches anything (prefix stripped)."""
    for prefix in prefixes:
        if not prefix:
            raise ValueError("[merl] empty checkpoint prefix is not allowed")
        out = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if out:
            return out
    return {}


def _load_into(module: Any, state: dict[str, Any], *, what: str, sample: str) -> None:
    if not state:
        raise RuntimeError(
            f"[merl] no weights matched the {what}. Need a full ResNet '*_ckpt.pth' "
            f"(keys like ecg_encoder.*, downconv.*, att_pool_head.*, proj_t.*). "
            f"Sample keys: {sample}"
        )
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"[merl] failed to load {what} with strict=True ({len(state)} matched keys). "
            f"{exc} Sample checkpoint keys: {sample}"
        ) from exc


def _require_resnet_ckpt(state: dict[str, Any]) -> None:
    keys = [k for k in state if isinstance(k, str)]
    has_downconv = any(k.startswith("downconv.") for k in keys)
    has_att = any(k.startswith("att_pool_head.") for k in keys)
    has_proj_e = any(k.startswith("proj_e.") for k in keys)
    has_proj_t = any(k.startswith("proj_t.") for k in keys)
    sample = _sample_keys(state)
    if has_proj_e and not has_downconv:
        raise RuntimeError(
            "[merl] checkpoint looks like a MERL ViT dual-encoder (proj_e present, "
            "no downconv). This wrapper implements the ResNet18 ECGCLIP path. "
            "Use res18_best_ckpt.pth (full '*_ckpt.pth'), not a vit_* checkpoint. "
            f"Sample keys: {sample}"
        )
    if not has_downconv or not has_att:
        raise RuntimeError(
            "[merl] checkpoint is missing downconv / att_pool_head (ResNet CLIP head). "
            "A '*_encoder.pth' is ECG-backbone only and cannot serve text→ECG queries. "
            f"Sample keys: {sample}"
        )
    if not has_proj_t:
        raise RuntimeError(
            "[merl] checkpoint is missing proj_t (text projection). "
            "Query and signal spaces cannot be aligned without it. "
            f"Sample keys: {sample}"
        )


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def swap_avl_avf(signal: np.ndarray) -> np.ndarray:
    """Swap aVL/aVF to MIMIC-ECG lead order (I, II, III, aVR, aVF, aVL, V1–V6)."""
    arr = np.array(signal, dtype=np.float32, copy=True)
    if arr.ndim != 2 or arr.shape[0] <= AVF_LEAD_INDEX:
        return arr
    arr[[AVL_LEAD_INDEX, AVF_LEAD_INDEX]] = arr[[AVF_LEAD_INDEX, AVL_LEAD_INDEX]]
    return arr


def prepare_query_text(text: str, *, lowercase: bool = True) -> str:
    """MERL text cleanup: EKG→ECG, strip junk, optional lowercase."""
    report = (text or "").replace("EKG", "ECG").replace("ekg", "ecg")
    report = report.strip(" *-=")
    if lowercase:
        report = report.lower()
    return report


def prepare_ecg(
    signal: np.ndarray,
    *,
    in_leads: int,
    target_len: int,
    native_fs: float | None = None,
    target_fs: float | None = None,
    minmax: bool = True,
    swap_leads: bool = True,
) -> np.ndarray:
    """Coerce an epoch signal to ``(in_leads, target_len)`` MERL input.

    Fills NaN/Inf, pads/truncates leads, converts sample rate when ``native_fs``
    and ``target_fs`` are given, then pads/truncates in time (no duration stretch).
    By default swaps aVL/aVF to MIMIC order and min-max scales to ``[0, 1]``.
    """
    from physiorag.ingestion.signal_prep import minmax_01, prepare_window

    arr = prepare_window(
        signal,
        n_leads=in_leads,
        n_samples=target_len,
        native_fs=native_fs,
        target_fs=target_fs,
    )
    if swap_leads:
        arr = swap_avl_avf(arr)
    if minmax:
        arr = minmax_01(arr)
    return arr


@dataclass(slots=True)
class _MerlConfig:
    checkpoint: str
    text_model: str
    projection_dim: int
    in_leads: int
    signal_length: int
    sample_rate_hz: float
    spatial_dim: int
    device: str
    local_files_only: bool
    minmax: bool
    swap_avl_avf: bool
    lowercase_text: bool
    ecg_prefixes: tuple[str, ...]
    downconv_prefixes: tuple[str, ...]
    att_pool_prefixes: tuple[str, ...]
    text_proj_prefixes: tuple[str, ...]


def _offline_hf() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _resolve_config(config: dict[str, Any]) -> _MerlConfig:
    emb = config.get("embeddings", {})
    merl = emb.get("merl", {}) or {}
    checkpoint = merl.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "MERL encoder requires embeddings.merl.checkpoint (local path to the "
            "MERL '*_ckpt.pth'). See README Phase B for offline setup."
        )
    signal_length = int(merl.get("signal_length", DEFAULT_SIGNAL_LENGTH))
    spatial_dim = int(merl.get("spatial_dim", resnet18_spatial_dim(signal_length)))
    local_files = merl.get("local_files_only")
    if local_files is None:
        local_files = _offline_hf()
    return _MerlConfig(
        checkpoint=str(checkpoint),
        text_model=str(merl.get("text_model", DEFAULT_TEXT_MODEL)),
        projection_dim=int(merl.get("projection_dim", DEFAULT_PROJECTION_DIM)),
        in_leads=int(merl.get("in_leads", DEFAULT_IN_LEADS)),
        signal_length=signal_length,
        sample_rate_hz=float(merl.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ)),
        spatial_dim=spatial_dim,
        device=str(merl.get("device", emb.get("device", "cpu"))),
        local_files_only=bool(local_files),
        minmax=bool(merl.get("minmax", True)),
        swap_avl_avf=bool(merl.get("swap_avl_avf", True)),
        lowercase_text=bool(merl.get("lowercase_text", True)),
        ecg_prefixes=tuple(merl.get("ecg_prefixes", ECG_PREFIXES)),
        downconv_prefixes=tuple(merl.get("downconv_prefixes", DOWNCONV_PREFIXES)),
        att_pool_prefixes=tuple(merl.get("att_pool_prefixes", ATT_POOL_PREFIXES)),
        text_proj_prefixes=tuple(merl.get("text_proj_prefixes", TEXT_PROJ_PREFIXES)),
    )


# --------------------------------------------------------------------------- #
# Signal tower encoder
# --------------------------------------------------------------------------- #
class MerlEcgEncoder(TimeSeriesEncoder):
    """MERL ECG signal tower as a PhysioRAG :class:`TimeSeriesEncoder`."""

    def __init__(self, cfg: _MerlConfig) -> None:
        import torch
        import torch.nn as nn

        self._cfg = cfg
        self._embedding_dim = cfg.projection_dim
        self._spatial_dim = cfg.spatial_dim
        self._device = torch.device(cfg.device)

        state = _load_state_dict(cfg.checkpoint)
        _require_resnet_ckpt(state)
        sample = _sample_keys(state)

        self._backbone = _build_resnet18(cfg.in_leads).to(self._device)
        _load_into(
            self._backbone,
            _select_by_prefix(state, cfg.ecg_prefixes),
            what="ECG backbone (ecg_encoder)",
            sample=sample,
        )
        self._downconv = nn.Conv1d(self._backbone.out_dim, cfg.projection_dim, kernel_size=1)
        self._downconv = self._downconv.to(self._device)
        _load_into(
            self._downconv,
            _select_by_prefix(state, cfg.downconv_prefixes),
            what="ECG downconv",
            sample=sample,
        )
        self._att_pool = build_attention_pool(
            spacial_dim=cfg.spatial_dim,
            embed_dim=cfg.projection_dim,
            num_heads=DEFAULT_ATT_HEADS,
            output_dim=cfg.projection_dim,
        ).to(self._device)
        _load_into(
            self._att_pool,
            _select_by_prefix(state, cfg.att_pool_prefixes),
            what="ECG att_pool_head",
            sample=sample,
        )
        self._backbone.eval()
        self._downconv.eval()
        self._att_pool.eval()
        logger.info(
            "Loaded MERL ResNet18 ECG tower from %s (dim=%d, spatial_dim=%d)",
            cfg.checkpoint,
            cfg.projection_dim,
            cfg.spatial_dim,
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, epochs: Sequence[WaveformEpoch]) -> np.ndarray:
        import torch

        if not epochs:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        batch = np.stack(
            [
                prepare_ecg(
                    e.signal,
                    in_leads=self._cfg.in_leads,
                    target_len=self._cfg.signal_length,
                    native_fs=float(e.sample_rate_hz) if e.sample_rate_hz else None,
                    target_fs=self._cfg.sample_rate_hz,
                    minmax=self._cfg.minmax,
                    swap_leads=self._cfg.swap_avl_avf,
                )
                for e in epochs
            ],
            axis=0,
        )
        x = torch.from_numpy(batch).to(self._device)
        with torch.no_grad():
            feats = self._backbone(x)
            if feats.shape[-1] != self._spatial_dim:
                raise RuntimeError(
                    f"[merl] ResNet spatial dim is {feats.shape[-1]}, expected "
                    f"{self._spatial_dim} (signal_length={self._cfg.signal_length}). "
                    "Official MERL ResNet18 is trained on 10 s @ 500 Hz (5000 samples)."
                )
            pooled, _ = self._att_pool(self._downconv(feats))
            vecs = pooled.reshape(pooled.shape[0], -1).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms


# --------------------------------------------------------------------------- #
# Text tower encoder (Med-CPT pooler + proj_t)
# --------------------------------------------------------------------------- #
class MerlTextEncoder:
    """MERL text tower: Med-CPT ``pooler_output`` + matched ``proj_t``.

    Produces vectors in the same space as :class:`MerlEcgEncoder` so NL queries
    can be nearest-neighbored against the ECG (waveform) index.
    """

    def __init__(self, cfg: _MerlConfig) -> None:
        import torch

        self._cfg = cfg
        self._embedding_dim = cfg.projection_dim
        self._device = torch.device(cfg.device)
        self._model = None
        self._tokenizer = None

        state = _load_state_dict(cfg.checkpoint)
        _require_resnet_ckpt(state)
        sample = _sample_keys(state)
        self._proj = build_text_projection(
            DEFAULT_TEXT_HIDDEN, cfg.projection_dim, cfg.projection_dim
        ).to(self._device)
        _load_into(
            self._proj,
            _select_by_prefix(state, cfg.text_proj_prefixes),
            what="text projection (proj_t)",
            sample=sample,
        )
        self._proj.eval()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        kwargs: dict[str, Any] = {"local_files_only": self._cfg.local_files_only}
        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.text_model, **kwargs)
        self._model = AutoModel.from_pretrained(self._cfg.text_model, **kwargs).to(self._device)
        self._model.eval()
        hidden = int(self._model.config.hidden_size)
        if hidden != DEFAULT_TEXT_HIDDEN:
            raise RuntimeError(
                f"[merl] text model hidden_size is {hidden}, expected {DEFAULT_TEXT_HIDDEN} "
                f"(MERL proj_t is Linear({DEFAULT_TEXT_HIDDEN}, {self._cfg.projection_dim})). "
                f"text_model={self._cfg.text_model}"
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None
        cleaned = [
            prepare_query_text(t, lowercase=self._cfg.lowercase_text) for t in texts
        ]
        tokens = self._tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self._model(**tokens)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                raise RuntimeError(
                    "[merl] text model did not return pooler_output. MERL's Med-CPT "
                    "tower projects pooler_output, not the raw CLS hidden state."
                )
            vecs = self._proj(pooled).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# --------------------------------------------------------------------------- #
# Builders (used by the encoder / text factories)
# --------------------------------------------------------------------------- #
def build_merl_encoder(config: dict[str, Any]) -> MerlEcgEncoder:
    return MerlEcgEncoder(_resolve_config(config))


def build_merl_text_encoder(config: dict[str, Any]) -> MerlTextEncoder:
    return MerlTextEncoder(_resolve_config(config))
