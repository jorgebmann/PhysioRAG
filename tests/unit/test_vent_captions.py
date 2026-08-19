"""Vent caption catalog, templated captions, and the DE→EN query glossary."""

from __future__ import annotations

from physiorag.ingestion.vent_captions import (
    VENT_CATALOG,
    apply_vent_glossary,
    build_caption,
    build_metadata,
)


def test_glossary_translates_german_vent_terms() -> None:
    out = apply_vent_glossary("Druckanstieg am Beatmungsgeraet")
    assert "pressure spike" in out
    assert "ventilator" in out


def test_glossary_translates_umlaut_ui_chip() -> None:
    # The real UI chip uses umlauts; folding must still map it to English.
    chip = (
        "Patienten mit ARDS, die spontan gegen das Beatmungsgerät atmen "
        "und dadurch einen Druckanstieg verursachen"
    )
    out = apply_vent_glossary(chip)
    assert "ventilator" in out
    assert "pressure spike" in out
    assert "spontaneous" in out


def test_glossary_leaves_english_unchanged() -> None:
    query = "double triggering during pressure support"
    assert apply_vent_glossary(query) == query


def test_glossary_does_not_touch_ecg_german() -> None:
    # PTB-XL ECG German must not be rewritten by the vent glossary.
    assert apply_vent_glossary("sinusrhythmus") == "sinusrhythmus"


def test_glossary_no_longer_carries_spo2_or_noop_phrases() -> None:
    # SpO2-only terms and the patient->patient no-op were removed so the
    # glossary is strictly ventilator vocabulary.
    assert apply_vent_glossary("Entsättigung") == "Entsättigung"
    assert apply_vent_glossary("Patient") == "Patient"


def test_build_caption_is_bilingual() -> None:
    caption = build_caption(VENT_CATALOG[0])
    assert "ventilator" in caption.lower()
    assert "beatmungs" in caption.lower()


def test_build_metadata_marks_medium_pairing_and_channels() -> None:
    meta = build_metadata(VENT_CATALOG[0], modality="ventilator")
    assert meta["pairing_tier"] == "medium"
    assert meta["channels"] == ["Paw", "Flow"]
    assert meta["asynchrony_type"] == "double_triggering"
    assert "vent_mode" in meta


def test_catalog_keeps_frozen_record_ids() -> None:
    ids = {s["record_id"] for s in VENT_CATALOG}
    assert {"demo-ards-001", "demo-copd-002", "demo-normal-003", "demo-ards-004"} <= ids
