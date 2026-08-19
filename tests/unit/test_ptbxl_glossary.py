"""PTB-XL DE/SV → EN glossary: longest-match phrase replace."""

from __future__ import annotations

from physiorag.ingestion.ptbxl_glossary import apply_ptbxl_glossary, translate_ptbxl_report


def test_longest_match_normales_ekg_before_ekg() -> None:
    assert translate_ptbxl_report("normales ekg") == "normal ecg"
    assert translate_ptbxl_report("sonst normales ekg") == "otherwise normal ecg"


def test_core_german_tokens() -> None:
    assert translate_ptbxl_report("sinusrhythmus") == "sinus rhythm"
    assert translate_ptbxl_report("vorhofflimmern") == "atrial fibrillation"
    assert translate_ptbxl_report("vorhofflattern") == "atrial flutter"
    assert translate_ptbxl_report("sinusbradykardie") == "sinus bradycardia"


def test_full_ptbxl_sentences() -> None:
    assert translate_ptbxl_report("sinusrhythmus normales ekg") == "sinus rhythm normal ecg"
    assert (
        translate_ptbxl_report("sinusbradykardie sonst normales ekg")
        == "sinus bradycardia otherwise normal ecg"
    )
    out = translate_ptbxl_report(
        "sinusrhythmus linkstyp unspezifisches abnormales t"
    )
    assert out == "sinus rhythm left axis nonspecific abnormal t wave"


def test_umlaut_and_hyphen_folding() -> None:
    assert "qt prolongation" in translate_ptbxl_report("qt-verlÄngerung")
    assert "conduction" in translate_ptbxl_report("2:1 Überleitung")
    assert "incomplete right bundle branch block" in translate_ptbxl_report(
        "unvollständiger rechtsschenkelblock"
    )


def test_swedish_axis_and_rhythm() -> None:
    assert translate_ptbxl_report("sinusrytm normalt ekg") == "sinus rhythm normal ecg"
    assert "left axis" in translate_ptbxl_report("vänster el-axel")
    assert "extreme left axis" in translate_ptbxl_report("extrem vänster el-axel")
    assert "such as in" in translate_ptbxl_report("t-förändring, såsom vid anterolateral ischemi")
    assert "depression" in translate_ptbxl_report("st-t sänkning")


def test_english_passthrough() -> None:
    assert translate_ptbxl_report("sinus rhythm normal ecg") == "sinus rhythm normal ecg"


def test_unmatched_tokens_kept() -> None:
    out = translate_ptbxl_report("sinusrhythmus unter cordichin")
    assert out.startswith("sinus rhythm")
    assert "cordichin" in out


def test_empty_and_none_like() -> None:
    assert translate_ptbxl_report("") == ""
    assert translate_ptbxl_report("   ") == ""


def test_apply_skips_english_keeps_original() -> None:
    assert apply_ptbxl_glossary("Atrial Fibrillation") == "Atrial Fibrillation"
    assert apply_ptbxl_glossary("qt-prolongation") == "qt-prolongation"
    assert apply_ptbxl_glossary("vorhofflimmern") == "atrial fibrillation"
    assert apply_ptbxl_glossary("sinusrhythmus normales ekg") == "sinus rhythm normal ecg"


def test_does_not_eat_isolated_t_or_p() -> None:
    # Single-letter lead names must survive; they are not glossary keys.
    out = translate_ptbxl_report("sinusrhythmus t in v4")
    assert " t " in f" {out} "
    assert " v4" in out
