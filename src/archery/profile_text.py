"""Parse the operator's free-text input block into typed session fields.

The block is what the coach types at the start of a session:

    Name: Samarth
    Height: 170 cms
    Weight: 60 Kgs
    Age: 16 years
    Resting HR: 60 bpm
    Average HR: 80 bpm

Three rules make this safe to build a report on.

1. A value outside its plausible range is REJECTED with the line quoted. It is
   never clamped and never silently dropped, because a typo that becomes a
   number in the evidence file is worse than a run that refuses to start.
2. A line no field matches is kept verbatim in ``unparsed`` and travels into the
   report as a note. It never becomes evidence.
3. Nothing is inferred. An absent field stays absent, so the report can say
   NOT PROVIDED rather than reasoning from a default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# field -> (session path, unit shown to the operator, low, high)
FIELDS: dict[str, tuple[str, str, float, float]] = {
    "height":      ("height_cm", "cm", 100.0, 250.0),
    "weight":      ("weight_kg", "kg", 20.0, 200.0),
    "age":         ("age", "years", 5.0, 100.0),
    "resting hr":  ("physio.heart_rate.rest_bpm", "bpm", 25.0, 230.0),
    "average hr":  ("physio.heart_rate.mean_bpm", "bpm", 25.0, 230.0),
    "max hr":      ("physio.heart_rate.max_bpm", "bpm", 25.0, 230.0),
    "name":        ("athlete_name", "", 0.0, 0.0),
}

# Spellings a coach actually types, mapped onto the canonical label above.
ALIASES = {
    "ht": "height", "stature": "height",
    "wt": "weight", "mass": "weight", "body weight": "weight",
    "years": "age", "age in years": "age",
    "rest hr": "resting hr", "resting heart rate": "resting hr",
    "rhr": "resting hr", "resting": "resting hr",
    "avg hr": "average hr", "mean hr": "average hr",
    "average heart rate": "average hr", "session hr": "average hr",
    "maximum hr": "max hr", "peak hr": "max hr", "max heart rate": "max hr",
    "athlete": "name", "athlete name": "name",
}

INTEGER_FIELDS = {"age"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class ProfileError(ValueError):
    """A line parsed to a value the pipeline will not accept as evidence."""


@dataclass
class Profile:
    values: dict[str, float | str] = field(default_factory=dict)   # session path -> value
    unparsed: list[str] = field(default_factory=list)
    raw: str = ""

    def get(self, path: str):
        return self.values.get(path)


def split_block(text: str) -> list[str]:
    """Newlines and semicolons both end a field. Windows shells make multi-line
    quoting painful, so ';' is a first-class separator, not a fallback."""
    parts: list[str] = []
    for chunk in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        parts.extend(chunk.split(";"))
    return [p.strip() for p in parts if p.strip()]


def _canonical(label: str) -> str | None:
    key = re.sub(r"[^a-z ]+", " ", label.lower())
    key = re.sub(r"\s+", " ", key).strip()
    if key in FIELDS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    return None


def parse(text: str) -> Profile:
    """Parse the block. Raises ProfileError on an out-of-range or unreadable value."""
    prof = Profile(raw=text or "")
    problems: list[str] = []

    for line in split_block(prof.raw):
        label, sep, value = line.partition(":")
        name = _canonical(label) if sep else None
        if name is None:
            prof.unparsed.append(line)
            continue

        path, unit, low, high = FIELDS[name]

        if name == "name":
            text_value = value.strip()
            if text_value:
                prof.values[path] = text_value
            else:
                prof.unparsed.append(line)
            continue

        found = _NUM.search(value)
        if not found:
            problems.append(f'"{line}"  no number found after the colon')
            continue
        number = float(found.group())
        if not (low <= number <= high):
            problems.append(
                f'"{line}"  {number:g} is outside the plausible range '
                f"{low:g} to {high:g} {unit}")
            continue
        prof.values[path] = int(round(number)) if name in INTEGER_FIELDS else number

    if problems:
        raise ProfileError(
            "These inputs were not accepted. Nothing was written.\n  "
            + "\n  ".join(problems)
            + "\nFix the value, or drop the field so the report records it as NOT PROVIDED."
        )
    return prof


def apply_to_session(session: dict, prof: Profile) -> dict:
    """Write parsed values into a copy of the session dict, creating the nested
    physio containers on the way. The session file stays the single source of
    truth; these flags are only a way to fill it."""
    out = dict(session)
    for path, value in prof.values.items():
        node = out
        parts = path.split(".")
        for key in parts[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
            node[key] = child
            node = child
        node[parts[-1]] = value
    if prof.raw.strip():
        out["inputs_raw"] = prof.raw.strip()
        out["inputs_unparsed"] = list(prof.unparsed)
    return out
