"""PTB-XL machine-report glossary: German (and Swedish) → English.

PTB-XL ``report`` strings are short, repetitive statements — mostly German, with a
Swedish subset (``sinusrytm``, ``vänster el-axel``, …). MERL's text tower was trained
on English MIMIC reports, so eval can rewrite the report with longest-match
phrase replace before embedding. No model, no GPU.

Query-side only (eval + ``signal_aligned`` ``/search``). Does not change stored
captions or SCP codes. Already-English queries are left unchanged.
"""

from __future__ import annotations

import re

# Matched case-insensitively after umlaut folding (ä→ae, ö→oe, ü→ue, ß→ss, å→aa)
# and hyphen→space. Longest folded key wins at each position. No single-letter keys.
_PHRASES: tuple[tuple[str, str], ...] = (
    ("inkomplett hoegersidigt skaenkelblock", "incomplete right bundle branch block"),
    ("unvollstaendiger rechtsschenkelblock", "incomplete right bundle branch block"),
    ("unspezifische intraventrikulaere leitungsstoerung", "nonspecific intraventricular conduction delay"),
    ("unspezifischer intraventrikulaerer block", "nonspecific intraventricular block"),
    ("interponierte supraventrikulaere extrasystole", "interpolated supraventricular extrasystole"),
    ("avvikande qrs(t) foerlopp", "abnormal qrs t morphology"),
    ("unspezifisches abnormales t", "nonspecific abnormal t wave"),
    ("vaenster kammarbelastning", "left ventricular strain"),
    ("vaenster kammarhypertrofi", "left ventricular hypertrophy"),
    ("vaenster kammarkhypertrofi", "left ventricular hypertrophy"),
    ("tachykardes vorhofflimmern", "tachycardic atrial fibrillation"),
    ("supraventrikulaere extrasystole", "supraventricular extrasystole"),
    ("supraventrikulaere tachykardie", "supraventricular tachycardia"),
    ("supraventrikulaera extraslag", "supraventricular extrasystoles"),
    ("hoegersidigt skaenkelblock", "right bundle branch block"),
    ("vaenstersidigt skaenkelblock", "left bundle branch block"),
    ("ospecifikt skaenkelblock", "nonspecific bundle branch block"),
    ("vaenster fraemre hemiblock", "left anterior fascicular block"),
    ("linksanteriorer hemiblock", "left anterior fascicular block"),
    ("linksposteriorer hemiblock", "left posterior fascicular block"),
    ("anteroseptaler myokardschaden", "anteroseptal myocardial damage"),
    ("anterolateraler infarkt", "anterolateral infarct"),
    ("anteroseptaler infarkt", "anteroseptal infarct"),
    ("inferiorer myokardschaden", "inferior myocardial damage"),
    ("laaga qrs amplituder", "low qrs amplitudes"),
    ("inget saekert patologiskt", "no definite pathology"),
    ("inget saekert patologikst", "no definite pathology"),
    ("nicht auszuschliessen", "cannot be excluded"),
    ("kan ej uteslutas", "cannot be excluded"),
    ("boer oevervaegas", "should be considered"),
    ("aalder ej bestaembar", "age undetermined"),
    ("sonst normales ekg", "otherwise normal ecg"),
    ("eljest normalt ekg", "otherwise normal ecg"),
    ("periphere niederspannung", "peripheral low voltage"),
    ("extrem vaenster el axel", "extreme left axis"),
    ("ueberdrehter linkstyp", "extreme left axis"),
    ("vaenster el axel", "left axis"),
    ("vaenster belastning", "left ventricular strain"),
    ("vaenster belastung", "left ventricular strain"),
    ("septaler infarkt", "septal infarct"),
    ("inferiorer infarkt", "inferior infarct"),
    ("p sinistrocardiale", "left atrial enlargement"),
    ("st t saenkung", "st t depression"),
    ("st t foeraendring", "st t change"),
    ("t foeraendring", "t wave change"),
    ("p foerlaengd", "prolonged p wave"),
    ("qt verlaengerung", "qt prolongation"),
    ("p verbreiterung", "p wave widening"),
    ("st senkung", "st depression"),
    ("st hebung", "st elevation"),
    ("sannolikt aeldre", "probably old"),
    ("moejligen faersk", "possibly acute"),
    ("sinus arrhythmie", "sinus arrhythmia"),
    ("sinus arytmi", "sinus arrhythmia"),
    ("pacemaker ekg", "pacemaker ecg"),
    ("normales ekg", "normal ecg"),
    ("normalt ekg", "normal ecg"),
    ("abnormales t", "abnormal t wave"),
    ("t abnormal", "abnormal t wave"),
    ("alter unbest", "age undetermined"),
    ("saasom vid", "such as in"),
    ("el axel", "axis"),
    ("sinusrhythmus", "sinus rhythm"),
    ("sinusbradykardie", "sinus bradycardia"),
    ("sinustachykardie", "sinus tachycardia"),
    ("sinusbradykardi", "sinus bradycardia"),
    ("sinustachykardi", "sinus tachycardia"),
    ("sinusrytm", "sinus rhythm"),
    ("vorhofflimmern", "atrial fibrillation"),
    ("vorhofflattern", "atrial flutter"),
    ("foermaksflimmer", "atrial fibrillation"),
    ("foermaksfladder", "atrial flutter"),
    ("rechtsschenkelblock", "right bundle branch block"),
    ("linksschenkelblock", "left bundle branch block"),
    ("skaenkelblock", "bundle branch block"),
    ("linkshypertrophie", "left ventricular hypertrophy"),
    ("myokardschaden", "myocardial damage"),
    ("myokardaffektion", "myocardial involvement"),
    ("myokardskada", "myocardial injury"),
    ("niederspannung", "low voltage"),
    ("unspezifisches", "nonspecific"),
    ("unspezifischer", "nonspecific"),
    ("unspezifische", "nonspecific"),
    ("extrasystole", "extrasystole"),
    ("ersatzsystole", "escape beat"),
    ("arrhythmie", "arrhythmia"),
    ("tachykardie", "tachycardia"),
    ("tachycardie", "tachycardia"),
    ("linkstyp", "left axis"),
    ("rechtstyp", "right axis"),
    ("ueberleitung", "conduction"),
    ("auszuschliessen", "cannot be excluded"),
    ("wahrscheinlich", "probably"),
    ("unvollstaendiger", "incomplete"),
    ("hoegersidigt", "right sided"),
    ("anterolateraler", "anterolateral"),
    ("anteroseptaler", "anteroseptal"),
    ("inferiorer", "inferior"),
    ("interponierte", "interpolated"),
    ("supraventrikulaere", "supraventricular"),
    ("avvikande", "abnormal"),
    ("foerlopp", "morphology"),
    ("foeraendring", "change"),
    ("ischemi", "ischemia"),
    ("saenkning", "depression"),
    ("saenkung", "depression"),
    ("amplituder", "amplitudes"),
    ("extremitetsavledningarna", "limb leads"),
    ("infarkt", "infarct"),
    ("abnormales", "abnormal"),
    ("normales", "normal"),
    ("normalt", "normal"),
    ("periphere", "peripheral"),
    ("hemiblock", "hemiblock"),
    ("saasom", "such as"),
    ("ventrikulaere", "ventricular"),
    ("eller", "or"),
    ("hoeg", "high"),
    ("vid", "with"),
    ("ospecifikt", "nonspecific"),
    ("eljest", "otherwise"),
    ("sonst", "otherwise"),
    ("jetzt", "now"),
    ("verdacht", "suspicion of"),
    ("moeglich", "possible"),
    ("teilweise", "partially"),
    ("etwas", "somewhat"),
    ("langsamer", "slower"),
    ("unveraendert", "unchanged"),
    ("ekg", "ecg"),
)


def fold_umlauts(text: str) -> str:
    """Lowercase and fold äöüßå so glossary keys match PTB-XL spelling variants."""
    t = (text or "").lower()
    return (
        t.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("å", "aa")
    )


def normalize_ptbxl_report(text: str) -> str:
    """Fold spelling so glossary keys can match PTB-XL machine text."""
    folded = fold_umlauts(text).replace("-", " ")
    folded = re.sub(r"\(n\)", "", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _compile() -> tuple[dict[str, str], re.Pattern[str]]:
    mapping: dict[str, str] = {}
    for src, dst in _PHRASES:
        mapping[fold_umlauts(src)] = dst
    keys = sorted(mapping, key=lambda k: (-len(k), k))
    # Longest alternative first; Python ``|`` is first-match. Boundaries keep
    # short tokens from firing inside longer unmatched words.
    pattern = re.compile(
        r"(?<![a-z0-9])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![a-z0-9])"
    )
    return mapping, pattern


_MAPPING, _PATTERN = _compile()


def translate_ptbxl_report(text: str) -> str:
    """Replace PTB-XL report tokens with English, longest phrase first.

    Unmatched tokens (drug names, numbers, already-English words) are kept.
    Hyphens become spaces so ``qt-verlängerung`` hits ``qt verlaengerung``.
    Output is collapsed lowercase text suitable as a MERL query.
    """
    folded = normalize_ptbxl_report(text)
    if not folded:
        return ""

    def _repl(match: re.Match[str]) -> str:
        return _MAPPING[match.group(0)]

    out = _PATTERN.sub(_repl, folded)
    return re.sub(r"\s+", " ", out).strip()


def apply_ptbxl_glossary(text: str) -> str:
    """Rewrite PTB-XL-style DE/SV reports; return ``text`` unchanged otherwise.

    A hit is any glossary replacement: if the translated string equals the
    normalized input, no source phrase matched and the original query is kept
    (casing, hyphens, and all) so English MERL queries are not rewritten.
    """
    if not (text or "").strip():
        return text
    translated = translate_ptbxl_report(text)
    if translated == normalize_ptbxl_report(text):
        return text
    return translated
