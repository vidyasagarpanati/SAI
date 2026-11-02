"""Versioned output names. Nothing in outputs/ is ever overwritten."""
from __future__ import annotations

import re
from pathlib import Path


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "Athlete"


def next_versioned(directory: Path, stem: str, suffix: str) -> Path:
    """<stem>_v01<suffix>, then _v02, ... Returns the first name not yet taken.

    Matched without regard to case: macOS and Windows file systems are
    case-insensitive, so "Dhar_Ath_v01" and "dhar_ath_v01" are the same file,
    and a case-sensitive count would hand out a name that overwrites it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(stem)}_v(\d+){re.escape(suffix)}$", re.IGNORECASE)
    taken = [int(m.group(1)) for p in directory.iterdir() if (m := pat.match(p.name))]
    n = max(taken) + 1 if taken else 1
    while (directory / f"{stem}_v{n:02d}{suffix}").exists():
        n += 1
    return directory / f"{stem}_v{n:02d}{suffix}"
