"""Evidence grounding: the mechanism that makes a fabricated number impossible.

The model never types a measured number. It writes evidence keys in double
braces, {{shot1.AIM.elbow_bow_deg.mean}}, and the renderer substitutes the value
and unit from 05_metrics.json. This module:

  * finds every placeholder and rejects keys that are not in the evidence index;
  * finds every number typed outside a placeholder and rejects it unless it is
    an allowed reference (Section 4, Phase 5, Priority #2, Shot 1, a 0-10 scale
    mention) or, in prescriptive fields, a training dose (3 sets, 2-4 weeks);
  * substitutes placeholders with formatted values for rendering.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
REFERENCE_BEFORE = re.compile(
    r"(Section|Sections|Phase|Phases|Priority|Priorities|Shot|Shots|shot|shots|Top|#|Appendix|"
    r"PRIORITY|PHASE|SECTION|SHOT)\s*#?\s*$")
SCALE = re.compile(r"\b0\s*[-–]\s*10\b|/\s*10\b")
DOSE_AFTER = re.compile(
    r"^\s*(?:[-–]\s*\d+\s*)?(?:x|×|sets?|reps?|repetitions?|times?|sessions?|days?|weeks?|months?|"
    r"minutes?|mins?|seconds?|secs?|s\b|per\b|arrows?|ends?|breaths?|%)", re.I)


def keys_in(text: str) -> list[str]:
    return PLACEHOLDER.findall(text or "")


def check_text(text: str, evidence: dict, *, prescriptive: bool = False) -> list[str]:
    """Return a list of violations for one free-text field."""
    if not isinstance(text, str) or not text:
        return []
    problems = []
    for key in keys_in(text):
        if key not in evidence:
            problems.append(f"unknown evidence key {{{{{key}}}}}")
    stripped = PLACEHOLDER.sub(" ", text)
    stripped = SCALE.sub(" ", stripped)
    for m in NUMBER.finditer(stripped):
        before = stripped[max(0, m.start() - 16):m.start()]
        after = stripped[m.end():m.end() + 16]
        if REFERENCE_BEFORE.search(before):
            continue
        if prescriptive and DOSE_AFTER.match(after):
            continue
        problems.append(f"number '{m.group(0)}' typed outside an evidence placeholder "
                        f"(...{before.strip()[-12:]} {m.group(0)} {after.strip()[:12]}...)")
    return problems


def walk_strings(obj: Any, path: str = "") -> Iterable[tuple[str, str]]:
    """Yield (json-path, string) for every string in a nested structure."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, f"{path}[{i}]")


def check_object(obj: Any, evidence: dict, prescriptive_fields: set[str] | None = None,
                 skip_fields: set[str] | None = None) -> list[str]:
    """Ground every string in a section's JSON output.

    prescriptive_fields: leaf field names where training doses are allowed
    skip_fields: leaf field names holding enums or ids, not prose
    """
    prescriptive_fields = prescriptive_fields or set()
    skip_fields = skip_fields or set()
    out = []
    for path, text in walk_strings(obj):
        leaf = re.sub(r"\[\d+\]$", "", path.split(".")[-1])
        if leaf in skip_fields:
            continue
        for p in check_text(text, evidence, prescriptive=leaf in prescriptive_fields):
            out.append(f"{path}: {p}")
    # evidence_keys arrays must reference real keys
    for path, text in walk_strings(obj):
        if re.search(r"evidence_keys\[\d+\]$", path) and text not in evidence:
            out.append(f"{path}: unknown evidence key '{text}'")
    return out


def format_value(entry: dict) -> str:
    v, unit = entry.get("value"), entry.get("units") or ""
    if v is None:
        return "n/a"
    if isinstance(v, float):
        txt = f"{v:g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.4f}"
    else:
        txt = str(v)
    short = {"shoulder widths": "SW", "shoulder widths/s": "SW/s"}.get(unit, unit)
    if not short:
        return txt
    return f"{txt}{short}" if short == "%" else f"{txt} {short}"


def substitute(text: str, evidence: dict) -> str:
    if not isinstance(text, str):
        return text
    return PLACEHOLDER.sub(lambda m: format_value(evidence.get(m.group(1), {})), text)


def substitute_all(obj: Any, evidence: dict) -> Any:
    if isinstance(obj, str):
        return substitute(obj, evidence)
    if isinstance(obj, dict):
        return {k: substitute_all(v, evidence) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_all(v, evidence) for v in obj]
    return obj
