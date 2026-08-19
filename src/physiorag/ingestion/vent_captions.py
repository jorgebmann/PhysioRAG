"""Ventilator asynchrony catalog, templated bilingual captions, and a small
German→English query glossary.

These are **synthetic, labeled** pairs (a signal window + a caption that names
the asynchrony in *that* window). In the pairing-quality sense of the project
brief they are **medium tier**: good enough for a demo / retrieval eval, but not
a claim of CLIP-scale ICU-note↔waveform alignment. ICU free-text notes are never
used here as captions.

The glossary is query-side only (``hybrid_text`` ``/search`` + vent eval). It is
deliberately separate from the PTB-XL ECG glossary so the two never leak into
each other's search path.
"""

from __future__ import annotations

import re

# Asynchrony / ventilation catalog. Each entry is one scenario the demo can
# synthesize and caption. ``pattern`` drives the waveform shape; the remaining
# fields become structured, searchable metadata on every emitted epoch.
#
# ``record_id`` values for the first four entries are frozen: the smoke demo,
# API tests, and citation tests reference ``demo-ards-001`` .. ``demo-ards-004``.
VENT_CATALOG: tuple[dict, ...] = (
    {
        "record_id": "demo-ards-001",
        "asynchrony_type": "double_triggering",
        "pattern": "spike",
        "vent_mode": "PSV",
        "peep_cmh2o": 10,
        "diagnosis": "ARDS",
        "event": "patient_ventilator_asynchrony",
        "finding": "pressure_spike",
        "en": (
            "10-second ventilator pressure/flow window: ARDS patient breathing "
            "spontaneously against the ventilator, causing a clear pressure spike "
            "and flow reversal suggestive of patient-ventilator asynchrony "
            "(double triggering)."
        ),
        "de": (
            "10-Sekunden-Fenster mit Beatmungsdruck und Fluss: ARDS-Patient atmet "
            "spontan gegen das Beatmungsgerät und verursacht einen deutlichen "
            "Druckanstieg mit Flussumkehr, Hinweis auf Patienten-Ventilator-"
            "Asynchronie (Double Triggering)."
        ),
    },
    {
        "record_id": "demo-copd-002",
        "asynchrony_type": "air_trapping",
        "pattern": "trap",
        "vent_mode": "PSV",
        "peep_cmh2o": 5,
        "diagnosis": "COPD",
        "event": "air_trapping",
        "finding": "auto_peep",
        "en": (
            "Ventilator flow curve with incomplete expiration and rising "
            "end-expiratory pressure consistent with air trapping and auto-PEEP "
            "in a COPD exacerbation."
        ),
        "de": (
            "Beatmungs-Flusskurve mit unvollständiger Ausatmung und steigendem "
            "endexspiratorischem Druck, vereinbar mit Air Trapping und Auto-PEEP "
            "bei COPD-Exazerbation."
        ),
    },
    {
        "record_id": "demo-normal-003",
        "asynchrony_type": "none",
        "pattern": "normal",
        "vent_mode": "VCV",
        "peep_cmh2o": 5,
        "diagnosis": "post_op",
        "event": "controlled_ventilation",
        "finding": "normal",
        "en": (
            "Stable volume-controlled ventilation with regular pressure and flow "
            "waveforms and no obvious patient-ventilator asynchrony."
        ),
        "de": (
            "Stabile volumenkontrollierte Beatmung mit regelmäßigen Druck- und "
            "Flusskurven ohne offensichtliche Asynchronie."
        ),
    },
    {
        "record_id": "demo-ards-004",
        "asynchrony_type": "low_compliance",
        "pattern": "stiff",
        "vent_mode": "VCV",
        "peep_cmh2o": 12,
        "diagnosis": "ARDS",
        "event": "low_compliance",
        "finding": "high_peak_pressure",
        "en": (
            "ARDS low-compliance pressure curve with elevated peak pressures during "
            "controlled breaths and reduced tidal excursion."
        ),
        "de": (
            "ARDS-Druckkurve bei niedriger Compliance mit erhöhten Spitzendrücken "
            "unter kontrollierten Atemzügen und reduziertem Tidalvolumen."
        ),
    },
    {
        "record_id": "demo-async-005",
        "asynchrony_type": "ineffective_effort",
        "pattern": "ineffective",
        "vent_mode": "PSV",
        "peep_cmh2o": 8,
        "diagnosis": "COPD",
        "event": "patient_ventilator_asynchrony",
        "finding": "missed_trigger",
        "en": (
            "Pressure support window with ineffective triggering: a small patient "
            "effort produces a flow and pressure deflection during expiration that "
            "fails to trigger a ventilator breath (missed trigger)."
        ),
        "de": (
            "Fenster unter Druckunterstützung mit frustraner Triggerung: eine "
            "kleine Patientenanstrengung erzeugt während der Ausatmung einen Fluss- "
            "und Druckausschlag, der keinen Beatmungshub auslöst (verpasster "
            "Trigger)."
        ),
    },
    {
        "record_id": "demo-async-006",
        "asynchrony_type": "flow_starvation",
        "pattern": "flow_starv",
        "vent_mode": "VCV",
        "peep_cmh2o": 8,
        "diagnosis": "ARDS",
        "event": "patient_ventilator_asynchrony",
        "finding": "flow_starvation",
        "en": (
            "Volume-controlled breath with flow starvation (air hunger): the "
            "inspiratory pressure curve is scooped and concave as patient demand "
            "exceeds the set inspiratory flow."
        ),
        "de": (
            "Volumenkontrollierter Atemzug mit Flussmangel (Lufthunger): die "
            "inspiratorische Druckkurve ist eingedellt und konkav, da der "
            "Patientenbedarf den eingestellten Inspirationsfluss übersteigt."
        ),
    },
    {
        "record_id": "demo-async-007",
        "asynchrony_type": "delayed_cycling",
        "pattern": "delayed",
        "vent_mode": "PSV",
        "peep_cmh2o": 6,
        "diagnosis": "COPD",
        "event": "patient_ventilator_asynchrony",
        "finding": "prolonged_inspiration",
        "en": (
            "Pressure support window with delayed cycling: the inspiratory time is "
            "prolonged with a wide pressure plateau extending into neural "
            "expiration."
        ),
        "de": (
            "Fenster unter Druckunterstützung mit verzögertem Umschalten: die "
            "Inspirationszeit ist verlängert mit einem breiten Druckplateau bis in "
            "die neurale Ausatmung."
        ),
    },
    {
        "record_id": "demo-async-008",
        "asynchrony_type": "reverse_triggering",
        "pattern": "reverse",
        "vent_mode": "VCV",
        "peep_cmh2o": 10,
        "diagnosis": "ARDS",
        "event": "patient_ventilator_asynchrony",
        "finding": "reverse_triggering",
        "en": (
            "Volume-controlled window with reverse triggering: a passive patient "
            "effort is entrained by the mechanical breath, adding a late "
            "inspiratory flow and pressure deflection."
        ),
        "de": (
            "Volumenkontrolliertes Fenster mit Reverse Triggering: eine passive "
            "Patientenanstrengung wird durch den maschinellen Atemzug ausgelöst und "
            "fügt einen späten inspiratorischen Fluss- und Druckausschlag hinzu."
        ),
    },
)

# One SpO2 scenario stays alongside the vent catalog so ``--modality spo2`` has
# something to serve; it is not part of the vent asynchrony story.
SPO2_SCENARIO: dict = {
    "record_id": "demo-spo2-009",
    "modality": "spo2",
    "asynchrony_type": "none",
    "pattern": "desat",
    "diagnosis": "hypoxemia",
    "event": "desaturation",
    "finding": "low_spo2",
    "en": (
        "Photoplethysmogram window around an SpO2 desaturation episode with "
        "reduced pulse amplitude."
    ),
    "de": (
        "Photoplethysmogramm-Fenster um eine SpO2-Entsättigung mit reduzierter "
        "Pulsamplitude."
    ),
}


def build_caption(scenario: dict) -> str:
    """Bilingual indexed caption (English then German).

    Both languages are stored so BM25 still matches a German UI query even when
    the query-side glossary misses a phrase.
    """
    en = str(scenario.get("en", "")).strip()
    de = str(scenario.get("de", "")).strip()
    return f"{en} {de}".strip()


def build_metadata(scenario: dict, *, modality: str) -> dict:
    """Structured, searchable metadata for one scenario epoch."""
    meta: dict = {
        "asynchrony_type": scenario.get("asynchrony_type", "none"),
        "diagnosis": scenario.get("diagnosis"),
        "event": scenario.get("event"),
        "finding": scenario.get("finding"),
        # Synthetic labeled captions are medium-tier pairs, never strong CLIP.
        "pairing_tier": "medium",
        "label": scenario.get("asynchrony_type", scenario.get("finding", "epoch")),
        "text": build_caption(scenario),
    }
    if modality == "ventilator":
        meta["channels"] = ["Paw", "Flow"]
        meta["vent_mode"] = scenario.get("vent_mode")
        meta["peep_cmh2o"] = scenario.get("peep_cmh2o")
    return {k: v for k, v in meta.items() if v is not None}


# --- query-side German→English glossary -------------------------------------
# Longest-match phrase replacement, umlaut-folded and case-insensitive. Only
# ventilator vocabulary lives here; ECG terms are intentionally absent so the
# ``hybrid_text`` path does not rewrite PTB-XL German (see test_retrieval_modes).
_VENT_PHRASES: tuple[tuple[str, str], ...] = (
    ("beatmungsgeraet", "ventilator"),
    ("beatmungsdruck", "ventilator pressure"),
    ("beatmung", "ventilation"),
    ("druckanstieg", "pressure spike"),
    ("druckunterstuetzung", "pressure support"),
    ("spitzendruck", "peak pressure"),
    ("druckplateau", "pressure plateau"),
    ("flusskurve", "flow curve"),
    ("flussumkehr", "flow reversal"),
    ("flussmangel", "flow starvation"),
    ("lufthunger", "air hunger"),
    ("ausatmung", "expiration"),
    ("einatmung", "inspiration"),
    ("atemzug", "breath"),
    ("spontanatmung", "spontaneous breathing"),
    ("spontan", "spontaneous"),
    ("asynchronie", "asynchrony"),
    ("frustrane triggerung", "ineffective triggering"),
    ("verpasster trigger", "missed trigger"),
    ("verzoegertes umschalten", "delayed cycling"),
    ("endexspiratorischer druck", "end-expiratory pressure"),
    ("niedrige compliance", "low compliance"),
    ("druck", "pressure"),
    ("fluss", "flow"),
)


def _fold(text: str) -> str:
    t = (text or "").lower()
    return (
        t.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def _compile() -> tuple[dict[str, str], "re.Pattern[str]"]:
    mapping = {_fold(src): dst for src, dst in _VENT_PHRASES}
    keys = sorted(mapping, key=lambda k: (-len(k), k))
    pattern = re.compile(
        r"(?<![a-z0-9])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![a-z0-9])"
    )
    return mapping, pattern


_MAPPING, _PATTERN = _compile()


def apply_vent_glossary(text: str) -> str:
    """Rewrite German vent terms to English; return ``text`` unchanged otherwise.

    A hit is any replacement. If nothing matches, the original query is returned
    verbatim (casing preserved) so English queries are left alone.
    """
    if not (text or "").strip():
        return text
    folded = _fold(text).replace("-", " ")
    folded = re.sub(r"\s+", " ", folded).strip()
    translated = _PATTERN.sub(lambda m: _MAPPING[m.group(0)], folded)
    translated = re.sub(r"\s+", " ", translated).strip()
    if translated == folded:
        return text
    return translated
