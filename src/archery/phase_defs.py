"""Phase names, badge numbers and colours, fixed by the master prompt (Section F)."""
from __future__ import annotations

# (code, badge number, display name, BGR colour for OpenCV overlays)
PHASES: list[tuple[str, int, str, tuple[int, int, int]]] = [
    ("STANCE", 1, "Stance / Setup", (128, 128, 128)),          # grey
    ("PRE_DRAW", 2, "Pre-draw", (0, 100, 0)),                  # dark green
    ("DRAW", 3, "Draw", (200, 90, 20)),                        # blue
    ("ANCHOR", 4, "Anchor", (150, 40, 130)),                   # purple
    ("AIM", 5, "Aim / Approach", (0, 140, 255)),               # orange
    ("EXPANSION", 6, "Expansion", (0, 215, 255)),              # yellow
    ("RELEASE", 7, "Release", (40, 40, 220)),                  # red
    ("FOLLOW_THROUGH", 8, "Follow-through", (20, 20, 139)),    # dark red
    ("RECOVERY", 9, "Recovery", (60, 90, 120)),                # grey-brown
]

ORDER = [p[0] for p in PHASES]
BADGE = {p[0]: p[1] for p in PHASES}
DISPLAY = {p[0]: p[2] for p in PHASES}
COLOR_BGR = {p[0]: p[3] for p in PHASES}
