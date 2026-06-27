"""Deterministic emergency red-flag safety net.

A medical triage assistant must never rely solely on a language model to catch
emergencies. This module scans the symptom text for hard red-flag patterns and
escalates the triage result to EMERGENCY regardless of what the model returned.
"""

import re

try:
    from mediscan_core import TriageResult, UrgencyLevel
except ImportError:
    from src.mediscan_core import TriageResult, UrgencyLevel

RED_FLAGS: dict[str, list[str]] = {
    "airway / anaphylaxis": [
        r"difficulty breathing", r"trouble breathing", r"can'?t breathe",
        r"swelling of (the )?(throat|tongue|lips)", r"throat .*clos", r"wheez",
    ],
    "spreading infection / sepsis": [
        r"spreading rapidly", r"red streaks?", r"high fever", r"fever .*(10[2-9]|4[0-9]\s?c)",
        r"pus .*spreading", r"hot,? swollen", r"rapidly worsening",
    ],
    "necrosis / severe skin": [
        r"black(ened)? (skin|tissue)", r"necro", r"tissue .*dying", r"large blisters?",
        r"skin .*(sloughing|peeling off)",
    ],
    "systemic deterioration": [
        r"confus(ed|ion)", r"unconscious", r"faint(ing|ed)", r"unresponsive",
        r"cannot (eat|drink|swallow)", r"severe(ly)? dehydrat",
    ],
    "high-risk patient": [
        r"newborn", r"infant .*fever", r"immunocompromised", r"pregnan",
    ],
}


def detect_red_flags(text: str) -> list[str]:
    text = (text or "").lower()
    return [label for label, patterns in RED_FLAGS.items() if any(re.search(p, text) for p in patterns)]


def escalate_for_safety(result: "TriageResult", symptom_text: str) -> tuple["TriageResult", list[str]]:
    flags = detect_red_flags(symptom_text)
    if flags and result is not None and result.urgency != UrgencyLevel.EMERGENCY:
        result.urgency = UrgencyLevel.EMERGENCY
        banner = "EMERGENCY RED FLAG(S) DETECTED — " + ", ".join(flags) + ". Seek immediate medical care now."
        result.when_to_seek_care = banner + " " + (result.when_to_seek_care or "")
    return result, flags
