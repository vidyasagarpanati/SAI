"""Step ids to something a person in the room can read.

The CLI prints `== S2 pose: started`, which is right for an operator and wrong
for a prospect watching over your shoulder. These labels say what the machine
is doing, in the order it does it, and the stage number comes from the step's
position in STEP_ORDER so a cached step that prints nothing still advances the
count honestly.
"""
from __future__ import annotations

import re

from archery.runstate import STEP_ORDER

LABEL = {
    "S0": "Reading the video",
    "SP": "Athlete and physiological inputs",
    "S1": "Extracting frames",
    "S2": "Pose estimation",
    "S3": "Joint angles",
    "S4": "Shot phases",
    "S5": "Statistics and evidence",
    "S6": "Key frames",
    "S7": "Annotated video",
    "S8": "Writing the report",
    "S9": "Verification",
    "S10": "Publishing the report",
}
TOTAL = len(STEP_ORDER)

# `== S2 pose: started` / `== S2 pose: PASS in 6.2 min` / `== S2 pose: FAILED after ...`
# FAILED before FAIL: alternation is first-match, and "FAIL" would swallow it.
STEP_LINE = re.compile(r"^==\s+(?P<step>S\w+)\s+\S+:\s*(?P<event>started|PASS|FAILED|FAIL)")


def index_of(step: str) -> int:
    """One-based position in the pipeline, or 0 for something unrecognised."""
    return STEP_ORDER.index(step) + 1 if step in STEP_ORDER else 0


def parse(line: str) -> tuple[str, str] | None:
    m = STEP_LINE.match(line.strip())
    return (m.group("step"), m.group("event")) if m else None


def label(step: str | None, index: int | None = None, total: int = TOTAL) -> str:
    if not step:
        return ""
    i = index or index_of(step)
    return f"Stage {i} of {total} - {LABEL.get(step, step)}"
