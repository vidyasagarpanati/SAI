"""Cross-video comparison: read finished runs, build the shot table, judge each
shot against the rest.

IN     runs/<id>/05_metrics.json for each run id passed on the command line
DO     flatten every shot of every run into one table of phase durations and
       key-frame positions, compute a leave-one-out baseline per phase, flag
       the shots that sit outside it, and group sessions by comparability
OUT    a comparison payload; compare_report.py renders it

Three rules, each of which exists because breaking it would produce a chart
that lies.

1. THE UNIT IS THE SHOT, NOT THE VIDEO. One run with several shots is a
   complete comparison on its own; several runs just widen the pool. And a
   two-shot video and a three-shot video have no shot 2 in common, so shots are
   pooled per phase and the shot index is only a label.
2. DURATIONS TRAVEL, ANGLES DO NOT. Phase durations are seconds and S0
   measures the true frame rate, so they compare across anything. Joint angles
   are image-plane projections, so they are only comparable inside a group of
   sessions filmed the same way.
3. LEAVE ONE OUT. A shot is judged against the mean and SD of all the OTHER
   shots. Pooling a shot into the baseline that judges it lets an extreme shot
   pull the line toward itself and hide.

Nothing here opens a video or writes into a run directory. 05_metrics.json is
already the frozen evidence file, so the comparison inherits the rule that a
number absent from it cannot reach a report.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from archery.phase_defs import DISPLAY, ORDER

# Positions worth trending. Angles only travel between sessions filmed the same
# way; anchor distance is normalised by shoulder width, so it travels further.
POSITION_MEASURES = ["elbow_bow_deg", "elbow_draw_deg", "shoulder_bow_deg",
                     "shoulder_draw_deg", "trunk_inclination_deg", "anchor_distance_norm"]
POSITION_PHASE = "AIM"
VIEW_FREE = {"anchor_distance_norm"}     # scale-normalised, not view-dependent


class CompareError(ValueError):
    """The comparison cannot be built. Never downgraded to an empty chart."""


@dataclass
class Session:
    run_id: str
    order: int                       # position on the command line, 1-based
    label: str
    athlete: str
    session_date: str | None
    camera_view: str
    measured_fps: float | None
    partial: bool
    n_shots: int


@dataclass
class Shot:
    run_id: str
    order: int
    session_label: str
    athlete: str
    shot: int
    label: str                       # "S1 shot 2"
    xi: int = 0                      # column on the comparison axis
    phases: dict = field(default_factory=dict)      # code -> {duration_s, start, end, confidence}
    positions: dict = field(default_factory=dict)   # measure -> value at the AIM key frame
    total_s: float | None = None


def _n(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_run(runs_dir: Path, run_id: str, order: int) -> tuple[Session, list[Shot]]:
    path = runs_dir / run_id / "05_metrics.json"
    if not path.is_file():
        raise CompareError(
            f"{run_id} has no 05_metrics.json.\n"
            f"  looked in: {path}\n"
            f"  Run the pipeline for that video at least as far as S5 first.")
    m = json.loads(path.read_text(encoding="utf-8"))

    sess = m.get("session", {})
    athlete_block = sess.get("athlete", {}) or {}
    label = f"S{order}"
    session = Session(
        run_id=run_id, order=order, label=label,
        athlete=str(athlete_block.get("athlete_name") or "UNKNOWN"),
        session_date=(athlete_block.get("session_date")
                      if isinstance(athlete_block.get("session_date"), str)
                      and athlete_block.get("session_date")[:2].isdigit() else None),
        camera_view=str(athlete_block.get("camera_view") or "UNSPECIFIED"),
        measured_fps=_n(sess.get("measured_fps")),
        partial=False,
        n_shots=int(sess.get("n_shots") or len(m.get("per_shot", []))),
    )

    shots: list[Shot] = []
    for entry in m.get("per_shot", []):
        s = Shot(run_id=run_id, order=order, session_label=label,
                 athlete=session.athlete, shot=int(entry["shot"]),
                 label=f"{label} shot {entry['shot']}")
        for code in ORDER:
            p = (entry.get("phases") or {}).get(code)
            # An undetected phase is a GAP. Writing zero here would draw a shot
            # that skipped a phase as a shot that did it instantly.
            if not p or not p.get("detected"):
                continue
            s.phases[code] = {
                "duration_s": _n(p.get("duration_s")),
                "start_t_s": _n(p.get("start_t_s")),
                "end_t_s": _n(p.get("end_t_s")),
                "confidence": p.get("boundary_confidence"),
            }
        aim = (entry.get("phases") or {}).get(POSITION_PHASE) or {}
        for name in POSITION_MEASURES:
            d = (aim.get("measures") or {}).get(name) or {}
            value = _n(d.get("at_key_frame"))
            if value is None:
                value = _n(d.get("mean"))
            if value is not None:
                s.positions[name] = {"value": value, "confidence": d.get("confidence"),
                                     "reason": d.get("confidence_reason")}
        spans = [p["duration_s"] for p in s.phases.values() if p["duration_s"] is not None]
        s.total_s = round(sum(spans), 3) if spans else None
        shots.append(s)

    if not shots:
        raise CompareError(f"{run_id} contains no detected shots, so there is "
                           f"nothing to compare from it.")
    return session, shots


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def baselines(shots: list[Shot], min_for_sd: int) -> dict:
    """Leave-one-out mean and SD per phase, per shot.

    Returns {phase: {"pooled": {...}, "per_shot": {shot_label: {...}}}}. The
    pooled figure is for display; every judgement uses the leave-one-out one,
    so the shot under test never contributes to the line it is measured against.
    """
    out: dict[str, dict] = {}
    for code in ORDER:
        rows = [(s.label, s.phases[code]["duration_s"]) for s in shots
                if code in s.phases and s.phases[code]["duration_s"] is not None]
        if not rows:
            continue
        values = [v for _, v in rows]
        mean, sd = _mean_sd(values)
        entry = {"n": len(values), "mean": mean,
                 "sd": sd if len(values) >= min_for_sd else None,
                 "sd_withheld": len(values) < min_for_sd,
                 "min_for_sd": min_for_sd, "per_shot": {}}
        for label, value in rows:
            others = [v for lab, v in rows if lab != label]
            o_mean, o_sd = _mean_sd(others)
            entry["per_shot"][label] = {
                "value": value, "mean": o_mean,
                "sd": o_sd if len(others) >= min_for_sd else None,
                "n_others": len(others)}
        out[code] = entry
    return out


def deviations(shots: list[Shot], base: dict, k: float) -> list[dict]:
    """Shots sitting beyond k leave-one-out SDs, worst first."""
    rows = []
    for s in shots:
        for code, phase in s.phases.items():
            b = (base.get(code) or {}).get("per_shot", {}).get(s.label)
            if not b or b["sd"] is None or not b["sd"] or b["value"] is None:
                continue
            z = (b["value"] - b["mean"]) / b["sd"]
            if abs(z) < k:
                continue
            rows.append({
                "shot": s.label, "session": s.session_label, "athlete": s.athlete,
                "phase": code, "phase_name": DISPLAY[code],
                "duration_s": round(b["value"], 3),
                "baseline_mean_s": round(b["mean"], 3),
                "baseline_sd_s": round(b["sd"], 3),
                "n_others": b["n_others"],
                "z": round(z, 2),
                "direction": "slower" if z > 0 else "faster",
                "boundary_confidence": phase.get("confidence"),
            })
    rows.sort(key=lambda r: -abs(r["z"]))
    return rows


def comparability(sessions: list[Session]) -> dict:
    """Which sessions may be compared on positions, and which may not.

    Durations are seconds and compare across everything. Angles are image-plane
    projections, so what decides comparability is the CAMERA VIEW, not how many
    sessions there are: three shots from one video share a view exactly, and
    two videos filmed from different positions never do.
    """
    groups: dict[str, list[str]] = {}
    for s in sessions:
        groups.setdefault(s.camera_view.strip().lower(), []).append(s.label)
    return {
        "groups": groups,
        "single_view": len(groups) == 1,
        "fps": sorted({round(s.measured_fps, 2) for s in sessions
                       if s.measured_fps is not None}),
    }


def warnings_for(sessions: list[Session]) -> list[str]:
    out = []
    if len(sessions) == 1:
        # Nothing about ordering, mixed athletes or mixed views can apply.
        return out
    athletes = sorted({s.athlete for s in sessions})
    if len(athletes) > 1:
        out.append(
            "MIXED ATHLETES: " + ", ".join(athletes) + ". Every series is labelled by "
            "athlete, and the baseline is no longer one athlete's own history, so a "
            "deviation here says a shot differs from the group, not from that "
            "archer's normal.")

    dated = [s for s in sessions if s.session_date]
    if len(dated) >= 2:
        by_order = [s.session_date for s in sorted(dated, key=lambda x: x.order)]
        if by_order != sorted(by_order):
            out.append(
                "ORDER DOES NOT MATCH DATES: the runs were given in an order whose "
                "session_date values are " + ", ".join(
                    f"{s.label}={s.session_date}" for s in sorted(dated, key=lambda x: x.order))
                + ". The charts follow the order you gave and nothing has been "
                  "reordered. Check the command if that was not deliberate.")
    undated = [s.label for s in sessions if not s.session_date]
    if undated:
        out.append("NO session_date on " + ", ".join(undated)
                   + ". The time axis is the order the runs were passed, which cannot "
                     "be checked against anything.")

    views = sorted({s.camera_view for s in sessions})
    if len(views) > 1:
        out.append("MIXED CAMERA VIEWS: " + ", ".join(views)
                   + ". Durations are charted across all sessions. Position charts are "
                     "drawn only within a view.")

    fps = sorted({round(s.measured_fps, 2) for s in sessions if s.measured_fps})
    if len(fps) > 1:
        out.append("Different capture rates across sessions: " + ", ".join(map(str, fps))
                   + " fps. Durations are in seconds, so this does not affect them; it "
                     "does affect how finely a phase boundary could be placed.")
    return out


def position_series(shots: list[Shot], sessions: list[Session], comp: dict) -> dict:
    """Per measure, the points that may honestly be drawn together.

    A group needs two or more SHOTS, not two or more sessions: within one video
    every shot was filmed from the same position, so the angles are as
    comparable as they ever get.
    """
    by_label = {s.label: s for s in sessions}
    out: dict[str, dict] = {}
    for name in POSITION_MEASURES:
        points = [{"shot": s.label, "session": s.session_label, "athlete": s.athlete,
                   "order": s.order, "xi": s.xi, "value": s.positions[name]["value"],
                   "confidence": s.positions[name].get("confidence"),
                   "reason": s.positions[name].get("reason"),
                   "view": by_label[s.session_label].camera_view}
                  for s in shots if name in s.positions]
        if len(points) < 2:
            continue
        if name in VIEW_FREE:
            out[name] = {"view_free": True, "groups": {"every session": points},
                         "not_comparable": []}
            continue
        grouped: dict[str, list] = {}
        for p in points:
            grouped.setdefault(p["view"], []).append(p)
        drawable = {view: pts for view, pts in grouped.items() if len(pts) >= 2}
        out[name] = {"view_free": False, "groups": drawable,
                     "not_comparable": sorted(set(grouped) - set(drawable))}
    return out


def build(runs_dir: Path, run_ids: list[str], sd_k: float, min_for_sd: int) -> dict:
    if not run_ids:
        raise CompareError("Give at least one run to compare.")
    if len(set(run_ids)) != len(run_ids):
        dupes = sorted({r for r in run_ids if run_ids.count(r) > 1})
        raise CompareError(f"The same run was passed more than once: {dupes}. "
                           f"A run compared against itself has no baseline.")

    sessions, shots, skipped = [], [], []
    for order, run_id in enumerate(run_ids, 1):
        try:
            session, run_shots = load_run(runs_dir, run_id, order)
        except CompareError as exc:
            skipped.append({"run_id": run_id, "reason": str(exc).splitlines()[0]})
            continue
        sessions.append(session)
        shots.extend(run_shots)

    if not sessions:
        raise CompareError(
            "No run could be read, so there is nothing to compare.\n  "
            + "\n  ".join(f"{s['run_id']}: {s['reason']}" for s in skipped))
    if len(shots) < 2:
        raise CompareError(
            f"Only one shot was found across {len(sessions)} run(s). A shot has "
            f"nothing to be compared against. Give a run with more shots, or add "
            f"another run.")

    # With one session the interesting axis is the shot; with several it is the
    # session, and shots from the same session share a column.
    within = len(sessions) == 1
    if within:
        axis_labels = [f"shot {s.shot}" for s in shots]
        for i, s in enumerate(shots):
            s.xi = i
    else:
        order_of = {s.label: i for i, s in enumerate(sorted(sessions, key=lambda x: x.order))}
        axis_labels = [s.label for s in sorted(sessions, key=lambda x: x.order)]
        for s in shots:
            s.xi = order_of[s.session_label]

    base = baselines(shots, min_for_sd)
    comp = comparability(sessions)

    # Leave-one-out costs a shot. With the three-shot floor for an SD, four
    # shots is the minimum at which anything can be flagged, and a report that
    # showed an empty table without saying so would read as "all clear".
    needed = min_for_sd + 1
    flagging = {
        "possible": len(shots) >= needed,
        "needed_shots": needed,
        "have_shots": len(shots),
        "reason": None if len(shots) >= needed else (
            f"{len(shots)} shots were analysed. Judging a shot against the others "
            f"leaves {len(shots) - 1}, below the {min_for_sd}-shot floor this "
            f"pipeline uses before it will report a standard deviation, so no shot "
            f"can be flagged. {needed} shots is the minimum. The durations below "
            f"are still shown and still comparable."),
    }
    return {
        "flagging": flagging,
        "schema_version": 1,
        "sd_threshold": sd_k,
        "min_shots_for_sd": min_for_sd,
        "within_one_session": within,
        "axis_labels": axis_labels,
        "axis_of": "shot" if within else "session",
        "sessions": [vars(s) for s in sessions],
        "shots": [{**vars(s)} for s in shots],
        "phase_order": [c for c in ORDER if any(c in s.phases for s in shots)],
        "phase_names": {c: DISPLAY[c] for c in ORDER},
        "baselines": base,
        "deviations": deviations(shots, base, sd_k),
        "comparability": comp,
        "positions": position_series(shots, sessions, comp),
        "warnings": warnings_for(sessions),
        "skipped_runs": skipped,
        "totals": {"sessions": len(sessions), "shots": len(shots),
                   "athletes": sorted({s.athlete for s in sessions})},
    }
