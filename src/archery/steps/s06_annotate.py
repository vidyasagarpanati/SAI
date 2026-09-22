"""S6 annotate.

IN     frames/, 02_landmarks.parquet, 03_kinematics.parquet, 04_phases.json,
       05_metrics.json
DO     one fully annotated reference frame per detected phase per shot
OUT    06_frames/shot<k>_p<badge>_<PHASE>.jpg, 06_manifest.json
VERIFY every detected phase has a frame; every overlay the master prompt
       requires is either drawn or explicitly skipped with a reason; labels do
       not sit on joints; files within size budget

Every number drawn on a key frame is the rounded value from 05_metrics.json
(at_key_frame), so the image and the evidence file agree to the last digit.

The coaching-implication box is interpretation, which belongs to the narrative
step. S6 marks it as pending; S10 re-renders the same frames through
``render_key_frame`` with the verified text from S8/S9.
"""
from __future__ import annotations

import cv2
import numpy as np

from archery import landmarks as L
from archery.context import Context
from archery.contracts import WARN, StepResult
from archery.framedata import FrameData
from archery.io_guard import guarded_path
from archery.overlay import REQUIRED_KEY_FRAME_ELEMENTS, render

PENDING_COACHING = ("[PENDING] Written by the narrative step from the verified findings, "
                    "then re-rendered into this box in the final report.")

COMMON = [("Bow elbow", "elbow_bow_deg", 1, "deg"), ("Draw elbow", "elbow_draw_deg", 1, "deg"),
          ("Bow shoulder", "shoulder_bow_deg", 1, "deg"), ("Draw shoulder", "shoulder_draw_deg", 1, "deg"),
          ("Trunk incl.", "trunk_inclination_deg", 1, "deg"), ("Shoulder tilt", "shoulder_tilt_deg", 1, "deg"),
          ("Pelvic tilt", "pelvic_tilt_deg", 1, "deg"), ("Head tilt", "head_tilt_deg", 1, "deg")]
PHASE_EXTRA = {
    "STANCE": [("Stance width", "stance_width_norm", 2, "SW"), ("Knee L", "knee_left_deg", 1, "deg"),
               ("Knee R", "knee_right_deg", 1, "deg")],
    "ANCHOR": [("Anchor dist.", "anchor_distance_norm", 3, "SW")],
    "AIM": [("Anchor dist.", "anchor_distance_norm", 3, "SW"), ("COM speed", "com_speed_norm_s", 3, "SW/s")],
    "EXPANSION": [("Anchor dist.", "anchor_distance_norm", 3, "SW")],
    "RELEASE": [("Draw wrist speed", "draw_wrist_speed_norm_s", 3, "SW/s")],
    "FOLLOW_THROUGH": [("Draw wrist speed", "draw_wrist_speed_norm_s", 3, "SW/s")],
}


def _fmt(v, dec, unit):
    return "n/a (withheld)" if v is None else f"{v:.{dec}f} {unit}"


def _phase_entry(metrics: dict, shot: int, code: str) -> dict:
    for s in metrics["per_shot"]:
        if s["shot"] == shot:
            return s["phases"].get(code, {})
    return {}


def _observation(code: str, entry: dict, vals: dict) -> str:
    parts = [f"[MEASURED] {code.replace('_', ' ').title()} phase "
             f"{entry.get('start_t_s')}-{entry.get('end_t_s')} s "
             f"({entry.get('duration_s')} s, {entry.get('boundary_confidence')} boundary)."]
    for label, key, dec, unit in COMMON[:5] + PHASE_EXTRA.get(code, []):
        v = vals.get(key)
        if v is not None:
            parts.append(f"{label} {v:.{dec}f} {unit}.")
    if code == "AIM":
        sx = (entry.get("com_sway_x_shoulder_widths") or {}).get("range")
        if sx is not None:
            parts.append(f"Pelvic-origin sway range {sx:.3f} SW across the phase.")
    return " ".join(parts)


def render_key_frame(fd: FrameData, draw_hand: str, shot: int, code: str,
                     coaching: str | None = None, track_conf: float = 0.5):
    """Render one key frame. Shared with S10 so the final report can re-render
    with verified coaching text through the identical code path."""
    sh = next(s for s in fd.phases["shots"] if s["shot"] == shot)
    ph = next(p for p in sh["phases"] if p["phase"] == code)
    kf = ph["key_frame"]
    entry = _phase_entry(fd.metrics, shot, code)
    vals = {m: d.get("at_key_frame") for m, d in entry.get("measures", {}).items()}

    ID = L.ID
    drw, bow = L.side(L.draw_side(draw_hand)), L.side(L.bow_side(draw_hand))
    swpx = fd.shoulder_width_px()
    extras: dict = {}
    by_code = {p["phase"]: p for p in sh["phases"]}

    if code == "ANCHOR":
        prev_pts, prev_ids = [], []
        for other in fd.phases["shots"]:
            if other["shot"] >= shot:
                continue
            a = next((p for p in other["phases"] if p["phase"] == "ANCHOR" and p["key_frame"] is not None), None)
            if a:
                prev_pts.append(fd.pts[a["key_frame"], drw["wrist"]])
                prev_ids.append(other["shot"])
        extras.update(previous_anchor_points=prev_pts, previous_anchor_shots=prev_ids)
    if code == "AIM":
        hip = (fd.pts[ph["start_frame"]:ph["end_frame"] + 1, ID["LEFT_HIP"]] +
               fd.pts[ph["start_frame"]:ph["end_frame"] + 1, ID["RIGHT_HIP"]]) / 2
        extras["pelvis_trail"] = hip
    if code in ("RELEASE", "FOLLOW_THROUGH"):
        pre = next((by_code[c]["end_frame"] for c in ("EXPANSION", "AIM", "ANCHOR")
                    if by_code.get(c, {}).get("detected")), None)
        if pre is not None and kf is not None and kf > pre:
            path = fd.pts[pre:kf + 1, drw["wrist"]]
            extras["draw_hand_path"] = path
            if np.isfinite(path[[0, -1]]).all() and swpx > 0:
                extras["draw_hand_displacement_sw"] = round(
                    float(np.linalg.norm(path[-1] - path[0]) / swpx), 2)
        r = sh["release_frame"]
        if kf is not None and kf > r:
            extras["bow_wrist_path"] = fd.pts[r:kf + 1, bow["wrist"]]
        ref = next((by_code[c]["key_frame"] for c in ("PRE_DRAW", "STANCE")
                    if by_code.get(c, {}).get("detected")), None)
        if ref is not None and kf is not None and swpx > 0:
            sg_now = (fd.pts[kf, ID["LEFT_SHOULDER"], 0] + fd.pts[kf, ID["RIGHT_SHOULDER"], 0]) / 2
            sg_ref = (fd.pts[ref, ID["LEFT_SHOULDER"], 0] + fd.pts[ref, ID["RIGHT_SHOULDER"], 0]) / 2
            if np.isfinite([sg_now, sg_ref]).all():
                extras["sg_shift_sw"] = round(float((sg_now - sg_ref) / swpx), 3)

    measurements = [(label, _fmt(vals.get(key), dec, unit))
                    for label, key, dec, unit in COMMON + PHASE_EXTRA.get(code, [])]
    frame = cv2.imread(str(fd.frames[kf]))
    canvas, rep = render(frame, fd.pts[kf], fd.vis[kf], vals, draw_hand=draw_hand, shot=shot,
                         phase=code, t_s=float(fd.kin.at[kf, "t_s"]), frame_idx=int(kf), mode="key",
                         track_conf=track_conf, measurements=measurements,
                         observation=_observation(code, entry, vals),
                         coaching=coaching or PENDING_COACHING, extras=extras)
    return canvas, rep, kf


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S6")
    fd = FrameData.load(ctx.run_dir)
    out_dir = guarded_path(ctx.key_frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = int(ctx.cfg.get("render.jpeg_quality_embed", 85)) + 5
    track = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))

    manifest, missing_frames, coverage_gaps, overlaps, not_extracted = [], [], [], 0, []
    for sh in fd.phases["shots"]:
        for p in sh["phases"]:
            code = p["phase"]
            if not p["detected"]:
                not_extracted.append({"shot": sh["shot"], "phase": code,
                                      "reason": p["reason_not_detected"]})
                continue
            if p["key_frame"] is None:
                missing_frames.append(f"shot {sh['shot']} {code}")
                continue
            canvas, rep, kf = render_key_frame(fd, ctx.draw_hand, sh["shot"], code, track_conf=track)
            name = f"shot{sh['shot']}_p{p['badge']}_{code}.jpg"
            path = out_dir / name
            cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
            gaps = rep.missing(REQUIRED_KEY_FRAME_ELEMENTS)
            if gaps:
                coverage_gaps.append(f"{name}: {gaps}")
            overlaps += rep.label_joint_overlaps
            manifest.append({
                "shot": sh["shot"], "phase": code, "badge": p["badge"],
                "display_name": p["display_name"], "key_frame": kf,
                "t_s": p["key_frame_t_s"], "file": str(path), "bytes": path.stat().st_size,
                "width": canvas.shape[1], "height": canvas.shape[0],
                "elements_drawn": sorted(rep.drawn), "elements_skipped": rep.skipped,
                "n_labels": rep.n_labels, "label_joint_overlaps": rep.label_joint_overlaps,
                "coaching_text": "pending",
            })

    out = ctx.write_json("06_manifest.json", {
        "frames": manifest, "phases_without_frame": not_extracted,
        "required_elements": REQUIRED_KEY_FRAME_ELEMENTS,
        "note": "Undetected phases get the '[ANNOTATED FRAME NOT AVAILABLE]' box in the report.",
    })

    res.check("key_frames_rendered", bool(manifest), f"{len(manifest)} annotated key frames")
    res.check("every_detected_phase_has_frame", not missing_frames,
              f"No key frame for: {missing_frames}" if missing_frames else "One frame per detected phase.")
    res.check("required_overlays_accounted_for", not coverage_gaps,
              "; ".join(coverage_gaps) if coverage_gaps else
              "Every required overlay is drawn or skipped with a stated reason on every frame.")
    skipped = sorted({k for m in manifest for k in m["elements_skipped"]})
    res.check("no_overlays_skipped", not skipped,
              f"Skipped on some frames (landmarks below confidence): {skipped}" if skipped
              else "Nothing skipped.", severity=WARN)
    res.check("labels_clear_of_joints", overlaps == 0,
              f"{overlaps} label(s) had to be placed over a landmark.", severity=WARN)
    big = [m["file"] for m in manifest if m["bytes"] > 3_000_000]
    res.check("frame_size_budget", not big, f"Over 3 MB: {big}", severity=WARN)

    res.outputs["key_frames_dir"] = str(out_dir)
    res.outputs["manifest"] = str(out)
    res.stats = {"n_frames": len(manifest), "phases_without_frame": len(not_extracted),
                 "label_joint_overlaps": overlaps}
    return res
