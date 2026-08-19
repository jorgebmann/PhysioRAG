"""CLI defaults for the demo / ECG smoke script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "smoke_demo.py"
_spec = importlib.util.spec_from_file_location("smoke_demo", _SCRIPT)
assert _spec and _spec.loader
_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoke)


def test_ptbxl_defaults_config_modality_query_and_plot() -> None:
    args = SimpleNamespace(
        dataset="ptbxl",
        config=None,
        modality="ventilator",
        query=_smoke.DEFAULT_VENT_QUERY,
        plot_out=None,
    )
    out = _smoke.apply_dataset_defaults(args)
    assert out.config.endswith("configs/ecg_merl.yaml")
    assert out.modality == "ecg"
    assert out.query == "sinus rhythm"
    assert out.plot_out.endswith("ecg_smoke.png")


def test_vent_defaults_unchanged() -> None:
    args = SimpleNamespace(
        dataset="mimic_demo",
        config=None,
        modality="ventilator",
        query=_smoke.DEFAULT_VENT_QUERY,
        plot_out=None,
    )
    out = _smoke.apply_dataset_defaults(args)
    assert out.config is None
    assert out.modality == "ventilator"
    assert out.query == _smoke.DEFAULT_VENT_QUERY
    assert out.plot_out is None
