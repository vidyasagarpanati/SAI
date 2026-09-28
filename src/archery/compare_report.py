"""Render the comparison payload to a versioned, self-contained HTML file."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from archery import __version__, charts
from archery.io_guard import guarded_open, guarded_path
from archery.phase_defs import BADGE, DISPLAY
from archery.versioning import next_versioned, safe

MEASURE_NAMES = {
    "elbow_bow_deg": "Bow elbow angle", "elbow_draw_deg": "Draw elbow angle",
    "shoulder_bow_deg": "Bow shoulder angle", "shoulder_draw_deg": "Draw shoulder angle",
    "trunk_inclination_deg": "Trunk inclination", "anchor_distance_norm": "Anchor distance",
}


def render(payload: dict, outputs_dir: Path, evidence_dir: Path, label: str) -> tuple[Path, Path]:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    payload = dict(payload)
    payload["phase_badges"] = {c: BADGE[c] for c in DISPLAY}
    payload["phase_names"] = {c: DISPLAY[c] for c in DISPLAY}
    payload["measure_names"] = MEASURE_NAMES

    # The numbers behind every mark, written before the HTML, so nothing
    # rendered is unsourced.
    ev_path = guarded_path(evidence_dir / "compare_evidence.json")
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    with guarded_open(ev_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    no_sd = [payload["phase_names"][c] for c, b in payload["baselines"].items()
             if b.get("sd_withheld")]
    excluded = sorted({f'{MEASURE_NAMES.get(m, m)} ({", ".join(spec["not_comparable"])})'
                       for m, spec in payload["positions"].items()
                       if spec.get("not_comparable")})

    root = Path(__file__).resolve().parents[2]
    env = Environment(loader=FileSystemLoader(str(root / "templates")),
                      autoescape=select_autoescape(["html"]))
    view = {
        "title": " vs ".join(s["label"] + (f' {s["session_date"]}' if s["session_date"] else "")
                             for s in payload["sessions"]),
        "payload": payload,
        "multi_athlete": len(payload["totals"]["athletes"]) > 1,
        "no_sd_phases": ", ".join(no_sd),
        "position_excluded": "; ".join(excluded),
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "provenance": [
            ("Runs compared", ", ".join(s["run_id"] for s in payload["sessions"])),
            ("Deviation threshold", f'{payload["sd_threshold"]} SD, leave-one-out'),
            ("Minimum shots for an SD", str(payload["min_shots_for_sd"])),
            ("Evidence file", str(ev_path)),
            ("Pipeline version", __version__),
            ("Model calls", "none: this report is computed, not written"),
        ],
    }
    # Charts are built from the payload and injected as markup, so the template
    # holds layout only.
    view["timeline"] = charts.shot_timeline(payload)
    view["timeline_legend"] = charts.timeline_legend(payload)
    view["phase_facets"] = charts.phase_facets(payload)
    view["position_facets"] = charts.position_facets(payload)
    for key in ("timeline", "timeline_legend", "phase_facets", "position_facets"):
        view[key] = _Raw(view[key])

    html = env.get_template("compare.html.j2").render(v=view)
    target = next_versioned(guarded_path(outputs_dir), f"Compare_{safe(label)}", ".html")
    with guarded_open(target, "w", encoding="utf-8") as fh:
        fh.write(html)
    return target, ev_path


class _Raw(str):
    """Markup this module generated, so it is inserted rather than escaped.
    Everything inside it went through charts.esc on the way in."""

    def __html__(self) -> str:
        return str(self)
