"""S5 stats: the evidence file.

IN     00_ingest.json, 02_pose_quality.json, 03_kinematics.parquet,
       03_quality.json, 04_phases.json, config/benchmarks.json
DO     per-phase statistics for every shot, cross-shot consistency, timing,
       benchmark comparison (cited entries only), overall analysis confidence
OUT    05_metrics.json
VERIFY no value computed from withheld data, no NaN in the output, SD and CV
       suppressed below the minimum shot count, every benchmark used carries a
       source, the evidence index is populated

05_metrics.json is the ONLY source of numbers for the report. S8 receives slices
of it, and S9 rejects any number in the narrative that is not in
``evidence_index``. Values are rounded here, once, to the precision the report
displays, so the grounding check compares like with like.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from archery.context import Context
from archery.contracts import WARN, StepResult

NOT_ASSESSABLE = "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO"
INDIVIDUAL = "Individualized assessment required"

# measure -> (units, decimals, is_ratio_scale). CV is only meaningful for
# ratio-scale quantities; a tilt that hovers around zero has no meaningful CV.
MEASURES: dict[str, tuple[str, int, bool]] = {
    "elbow_bow_deg": ("deg", 1, True), "elbow_draw_deg": ("deg", 1, True),
    "shoulder_bow_deg": ("deg", 1, True), "shoulder_draw_deg": ("deg", 1, True),
    "wrist_bow_deg": ("deg", 1, True), "wrist_draw_deg": ("deg", 1, True),
    "hip_left_deg": ("deg", 1, True), "hip_right_deg": ("deg", 1, True),
    "knee_left_deg": ("deg", 1, True), "knee_right_deg": ("deg", 1, True),
    "ankle_left_deg": ("deg", 1, True), "ankle_right_deg": ("deg", 1, True),
    "trunk_inclination_deg": ("deg", 1, False), "neck_inclination_deg": ("deg", 1, False),
    "head_tilt_deg": ("deg", 1, False), "shoulder_tilt_deg": ("deg", 1, False),
    "pelvic_tilt_deg": ("deg", 1, False), "shoulder_hip_separation_deg": ("deg", 1, False),
    "anchor_distance_norm": ("shoulder widths", 3, True),
    "stance_width_norm": ("shoulder widths", 3, True),
    "bow_arm_elevation_norm": ("shoulder widths", 3, False),
    "draw_wrist_speed_norm_s": ("shoulder widths/s", 3, True),
    "com_speed_norm_s": ("shoulder widths/s", 3, True),
}
CORE_AIM = ["elbow_bow_deg", "elbow_draw_deg", "shoulder_bow_deg", "shoulder_draw_deg",
            "trunk_inclination_deg", "anchor_distance_norm"]


def _r(x, d):
    if x is None:
        return None
    x = float(x)
    return round(x, d) if math.isfinite(x) else None


def _confidence(n_valid: int, share: float, boundary: str | None) -> str | None:
    if n_valid == 0:
        return None
    if share >= 0.9 and n_valid >= 5:
        c = "HIGH"
    elif share >= 0.6 and n_valid >= 3:
        c = "MEDIUM"
    else:
        c = "LOW"
    if boundary == "LOW" and c == "HIGH":
        c = "MEDIUM"          # a shaky phase boundary caps what we can claim
    return c


def describe(values: np.ndarray, decimals: int, boundary: str | None) -> dict:
    n_total = int(values.size)
    v = values[np.isfinite(values)]
    n = int(v.size)
    share = n / n_total if n_total else 0.0
    conf = _confidence(n, share, boundary)
    if n == 0:
        return {"n_frames": n_total, "n_valid": 0, "valid_share": 0.0, "confidence": None,
                "status": NOT_ASSESSABLE}
    return {
        "n_frames": n_total, "n_valid": n, "valid_share": round(share, 3),
        "min": _r(v.min(), decimals), "max": _r(v.max(), decimals),
        "mean": _r(v.mean(), decimals), "range": _r(v.max() - v.min(), decimals),
        "sd": _r(v.std(ddof=1), decimals) if n >= 3 else None,
        "confidence": conf,
    }


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S5")
    ingest = ctx.read_json("00_ingest.json")
    pose_q = ctx.read_json("02_pose_quality.json")
    kin_q = ctx.read_json("03_quality.json")
    phases = ctx.read_json("04_phases.json")
    k = pd.read_parquet(ctx.artefact("03_kinematics.parquet"))
    fps = float(phases["analysis_fps"])
    min_shots = int(ctx.cfg.get("stats.min_shots_for_sd", 3))

    shoulder_w = float(np.nanmedian(k["shoulder_width_norm"])) if "shoulder_width_norm" in k else np.nan
    measures = {m: spec for m, spec in MEASURES.items() if m in k.columns}
    evidence: dict[str, dict] = {}

    def ev(key: str, value, units: str, confidence: str | None = None, n: int | None = None):
        if value is None:
            return
        evidence[key] = {"value": value, "units": units, "confidence": confidence, "n": n}

    # ---- session -----------------------------------------------------------
    probe = ingest["probe"]
    session = {
        "athlete": ingest["session"],
        "video_file": ingest["video_path"].replace("\\", "/").split("/")[-1],
        "video_sha256": ingest["video_sha256"],
        "measured_fps": _r(ingest["measured_fps"], 3),
        "analysis_fps": fps,
        "duration_s": _r(probe.get("duration_s"), 2),
        "resolution": f"{probe.get('width')}x{probe.get('height')}",
        "codec": probe.get("codec"),
        "metadata_source": probe.get("source"),
        "n_shots": phases["n_shots"],
        "pose_estimation": "MediaPipe Pose Landmarker (full, 33 landmarks), VIDEO mode, CPU",
        "analysis_method": "Deterministic computer-vision pipeline: frame decode, pose estimation, "
                           "kinematics, rule-based phase segmentation, descriptive statistics",
    }
    ev("session.measured_fps", session["measured_fps"], "fps", "HIGH")
    ev("session.analysis_fps", fps, "fps", "HIGH")
    ev("session.duration_s", session["duration_s"], "s", "HIGH")
    ev("session.n_shots", phases["n_shots"], "shots", "HIGH")

    # ---- data quality and overall analysis confidence ----------------------
    nan_rates = kin_q.get("nan_rate_by_measure", {})
    worst_core = max((nan_rates.get(c, 1.0) for c in CORE_AIM), default=1.0)
    det, usable = float(pose_q.get("detection_rate", 0)), float(kin_q.get("usable_frame_share", 0))
    if det >= 0.95 and usable >= 0.90 and worst_core <= 0.10:
        overall = "HIGH"
    elif det >= 0.85 and usable >= 0.70 and worst_core <= 0.30:
        overall = "MEDIUM"
    else:
        overall = "LOW"
    quality = {
        "pose_detection_rate_pct": _r(100 * det, 1),
        "usable_frame_share_pct": _r(100 * usable, 1),
        "mean_visible_landmarks": _r(np.mean(pose_q.get("per_frame_visible_count", [0])), 1),
        "worst_core_measure_missing_pct": _r(100 * worst_core, 1),
        "missing_pct_by_measure": {m: _r(100 * v, 1) for m, v in nan_rates.items()},
        "withheld_reason_by_measure": kin_q.get("dominant_gate_reason", {}),
        "implausible_values": kin_q.get("implausible_values", {}),
        "draw_hand_check": kin_q.get("draw_hand_check"),
        "overall_analysis_confidence": overall,
        "overall_confidence_rule": "HIGH: detection>=95%, usable>=90%, worst core missing<=10%. "
                                   "MEDIUM: >=85%, >=70%, <=30%. Otherwise LOW.",
    }
    for key in ("pose_detection_rate_pct", "usable_frame_share_pct", "mean_visible_landmarks",
                "worst_core_measure_missing_pct"):
        ev(f"quality.{key}", quality[key], "%" if key.endswith("pct") else "landmarks", "HIGH")

    # ---- per shot, per phase ----------------------------------------------
    per_shot = []
    for sh in phases["shots"]:
        s_out = {"shot": sh["shot"], "release_t_s": sh["release_t_s"],
                 "release_confidence": sh["release_confidence"], "phases": {}}
        ev(f"shot{sh['shot']}.release_t_s", sh["release_t_s"], "s", sh["release_confidence"])
        for p in sh["phases"]:
            code = p["phase"]
            if not p["detected"]:
                s_out["phases"][code] = {"detected": False, "reason": p["reason_not_detected"]}
                continue
            seg = k.iloc[p["start_frame"]:p["end_frame"] + 1]
            kf = p["key_frame"]
            entry = {"detected": True, "start_t_s": p["start_t_s"], "end_t_s": p["end_t_s"],
                     "duration_s": p["duration_s"], "key_frame": kf,
                     "key_frame_t_s": p["key_frame_t_s"],
                     "boundary_confidence": p["boundary_confidence"], "measures": {}}
            base = f"shot{sh['shot']}.{code}"
            ev(f"{base}.start_t_s", p["start_t_s"], "s", p["boundary_confidence"])
            ev(f"{base}.end_t_s", p["end_t_s"], "s", p["boundary_confidence"])
            ev(f"{base}.duration_s", p["duration_s"], "s", p["boundary_confidence"])
            for m, (units, dec, _) in measures.items():
                d = describe(seg[m].to_numpy(dtype=float), dec, p["boundary_confidence"])
                if kf is not None and np.isfinite(k.at[kf, m]):
                    d["at_key_frame"] = _r(k.at[kf, m], dec)
                entry["measures"][m] = d
                for stat in ("min", "max", "mean", "range", "sd", "at_key_frame"):
                    ev(f"{base}.{m}.{stat}", d.get(stat), units, d.get("confidence"), d.get("n_valid"))
            # centre-of-mass sway in shoulder widths: the aim-stability measure
            if np.isfinite(shoulder_w) and shoulder_w > 0:
                for axis in ("x", "y"):
                    c = seg[f"com_{axis}_norm"].to_numpy(dtype=float) / shoulder_w
                    d = describe(c, 3, p["boundary_confidence"])
                    sway = {"range": d.get("range"), "sd": d.get("sd"),
                            "n_valid": d.get("n_valid"), "confidence": d.get("confidence")}
                    entry[f"com_sway_{axis}_shoulder_widths"] = sway
                    ev(f"{base}.com_sway_{axis}.range", sway["range"], "shoulder widths",
                       sway["confidence"], sway["n_valid"])
                    ev(f"{base}.com_sway_{axis}.sd", sway["sd"], "shoulder widths",
                       sway["confidence"], sway["n_valid"])
            s_out["phases"][code] = entry
        per_shot.append(s_out)

    # ---- cross-shot consistency -------------------------------------------
    n_shots = len(per_shot)
    enough = n_shots >= min_shots
    suppressed_reason = (None if enough else
                         f"{NOT_ASSESSABLE} (shot-to-shot variation requires at least "
                         f"{min_shots} shots; {n_shots} detected)")
    phase_codes = [p["phase"] for p in phases["shots"][0]["phases"]] if phases["shots"] else []
    cross: dict[str, dict] = {}
    timing: dict[str, dict] = {}
    for code in phase_codes:
        durs = [s["phases"][code]["duration_s"] for s in per_shot
                if s["phases"].get(code, {}).get("detected")]
        t_entry = {"n_shots": len(durs), "values_s": durs}
        if len(durs) >= min_shots:
            arr = np.array(durs)
            t_entry.update({"mean_s": _r(arr.mean(), 3), "sd_s": _r(arr.std(ddof=1), 3),
                            "range_s": _r(arr.max() - arr.min(), 3),
                            "cv_pct": _r(100 * arr.std(ddof=1) / arr.mean(), 1) if arr.mean() > 0 else None})
            for stat in ("mean_s", "sd_s", "range_s", "cv_pct"):
                ev(f"all.{code}.duration.{stat}", t_entry[stat], "%" if stat == "cv_pct" else "s",
                   "HIGH" if len(durs) >= 5 else "MEDIUM", len(durs))
        else:
            t_entry["status"] = suppressed_reason or NOT_ASSESSABLE
        timing[code] = t_entry

        cross[code] = {}
        for m, (units, dec, ratio) in measures.items():
            vals = [s["phases"][code]["measures"][m].get("mean") for s in per_shot
                    if s["phases"].get(code, {}).get("detected")
                    and s["phases"][code]["measures"][m].get("confidence")]
            vals = [v for v in vals if v is not None]
            c = {"n_shots": len(vals), "per_shot_means": vals}
            if len(vals) >= min_shots:
                arr = np.array(vals, dtype=float)
                sd = arr.std(ddof=1)
                c.update({"mean": _r(arr.mean(), dec), "sd": _r(sd, dec),
                          "range": _r(arr.max() - arr.min(), dec),
                          "cv_pct": (_r(100 * sd / abs(arr.mean()), 1)
                                     if ratio and abs(arr.mean()) > 1e-6 else None)})
                if not ratio:
                    c["cv_note"] = "CV not reported: measure is not ratio-scale (can be near zero)"
                for stat in ("mean", "sd", "range"):
                    ev(f"all.{code}.{m}.{stat}", c[stat], units, "MEDIUM", len(vals))
                ev(f"all.{code}.{m}.cv_pct", c.get("cv_pct"), "%", "MEDIUM", len(vals))
            else:
                c["status"] = suppressed_reason or NOT_ASSESSABLE
            cross[code][m] = c

    # ---- consistency rankings (only when there is enough data) -------------
    rankings: dict = {"status": suppressed_reason} if not enough else {}
    if enough:
        angle_ms = [m for m, (u, _, ratio) in measures.items() if u == "deg" and ratio]
        phase_cv = {}
        for code in phase_codes:
            cvs = [cross[code][m].get("cv_pct") for m in angle_ms]
            cvs = [v for v in cvs if v is not None]
            if cvs:
                phase_cv[code] = _r(float(np.mean(cvs)), 1)
        if phase_cv:
            rankings["mean_joint_angle_cv_pct_by_phase"] = phase_cv
            rankings["most_consistent_phase"] = min(phase_cv, key=phase_cv.get)
            rankings["least_consistent_phase"] = max(phase_cv, key=phase_cv.get)
        sds = [(code, m, cross[code][m]["sd"]) for code in phase_codes for m in measures
               if measures[m][0] == "deg" and cross[code][m].get("sd") is not None]
        if sds:
            code, m, v = max(sds, key=lambda x: x[2])
            rankings["largest_angular_variation"] = {"phase": code, "measure": m, "sd_deg": v}
        pos = [(code, m, cross[code][m]["sd"]) for code in phase_codes for m in measures
               if measures[m][0] == "shoulder widths" and cross[code][m].get("sd") is not None]
        if pos:
            code, m, v = max(pos, key=lambda x: x[2])
            rankings["largest_positional_variation"] = {"phase": code, "measure": m,
                                                        "sd_shoulder_widths": v}
        tims = [(code, timing[code]["sd_s"]) for code in phase_codes if timing[code].get("sd_s") is not None]
        if tims:
            code, v = max(tims, key=lambda x: x[1])
            rankings["largest_timing_variation"] = {"phase": code, "sd_s": v}

    # ---- benchmarks: cited entries only -------------------------------------
    usable_bm, rejected_bm = [], []
    for e in ctx.cfg.benchmarks.get("entries", []):
        why = None
        if not e.get("source"):
            why = "no source"
        elif e.get("status") == "NEEDS_SOURCE":
            why = "status NEEDS_SOURCE"
        elif e.get("target") is None and e.get("range") is None:
            why = "no target or range"
        elif not e.get("measure") or not e.get("phase"):
            why = "no measure/phase mapping to pipeline output"
        if why:
            rejected_bm.append({"id": e.get("id"), "reason": why})
            continue
        m, code = e["measure"], e["phase"]
        c = cross.get(code, {}).get(m, {})
        athlete = c.get("mean")
        basis = "cross-shot mean"
        if athlete is None and per_shot:
            athlete = per_shot[0]["phases"].get(code, {}).get("measures", {}).get(m, {}).get("mean")
            basis = "shot 1 mean (single shot)"
        target = e.get("target")
        row = {"id": e["id"], "variable": e.get("variable"), "measure": m, "phase": code,
               "athlete_value": athlete, "athlete_basis": basis, "target": target,
               "range": e.get("range"), "units": e.get("units"),
               "difference": _r(athlete - target, 1) if athlete is not None and target is not None else None,
               "evidence_class": e.get("evidence_class"), "source": e["source"],
               "confidence": "MEDIUM" if athlete is not None else None}
        usable_bm.append(row)
        ev(f"benchmark.{e['id']}.athlete_value", athlete, e.get("units") or "", row["confidence"])
        ev(f"benchmark.{e['id']}.target", target, e.get("units") or "", "HIGH")
        ev(f"benchmark.{e['id']}.difference", row["difference"], e.get("units") or "", row["confidence"])
    benchmarked = {r["measure"] for r in usable_bm}
    benchmarks = {"rows": usable_bm, "rejected_entries": rejected_bm,
                  "no_benchmark": {m: INDIVIDUAL for m in measures if m not in benchmarked}}

    payload = {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "config_hash": ctx.cfg.config_hash,
        "session": session,
        "data_quality": quality,
        "phase_timeline": [{"shot": sh["shot"], "phases": [
            {k2: p[k2] for k2 in ("phase", "badge", "display_name", "detected", "start_t_s",
                                  "end_t_s", "duration_s", "key_frame", "key_frame_t_s",
                                  "boundary_confidence", "driving_signal", "reason_not_detected")}
            for p in sh["phases"]]} for sh in phases["shots"]],
        "per_shot": per_shot,
        "cross_shot": {"min_shots_for_sd": min_shots, "n_shots": n_shots,
                       "status": suppressed_reason, "by_phase": cross},
        "timing": timing,
        "consistency_rankings": rankings,
        "benchmarks": benchmarks,
        "measure_units": {m: spec[0] for m, spec in measures.items()},
        "evidence_index": evidence,
    }

    text = json.dumps(payload, indent=2, allow_nan=False, default=str)
    from archery.io_guard import guarded_open
    out = ctx.artefact("05_metrics.json")
    with guarded_open(out, "w", encoding="utf-8") as fh:
        fh.write(text)

    # ---- verification ---------------------------------------------------------
    res.check("metrics_written_without_nan", True, f"{len(text) / 1024:.0f} KB, strict JSON (no NaN)")
    res.check("evidence_index_populated", len(evidence) > 0, f"{len(evidence)} evidence entries")
    leaks = [f"{code}.{m}" for code in cross for m, c in cross[code].items()
             if c.get("sd") is not None and c["n_shots"] < min_shots]
    res.check("sd_suppressed_below_min_shots", not leaks,
              f"SD reported with too few shots: {leaks[:5]}" if leaks else
              ("Cross-shot SD computed." if enough else suppressed_reason))
    res.check("benchmarks_all_cited", all(r.get("source") for r in usable_bm),
              f"{len(usable_bm)} cited benchmark(s) used, {len(rejected_bm)} rejected: "
              f"{[r['reason'] for r in rejected_bm]}")
    res.check("benchmarks_available", bool(usable_bm),
              f"No usable benchmarks. Section 7 will render '{INDIVIDUAL}' for every variable "
              f"until cited entries are added to config/benchmarks.json.", severity=WARN)
    missing_aim = []
    for s in per_shot:
        aim = s["phases"].get("AIM", {})
        if not aim.get("detected"):
            missing_aim.append(f"shot {s['shot']}: AIM not detected")
            continue
        for m in CORE_AIM:
            if m in aim["measures"] and not aim["measures"][m].get("confidence"):
                missing_aim.append(f"shot {s['shot']}: {m}")
    res.check("core_aim_measures_available", not missing_aim,
              f"Missing at full draw: {missing_aim}" if missing_aim else
              "All core full-draw measures available in every shot.", severity=WARN)
    res.check("overall_confidence_assigned", overall in ("HIGH", "MEDIUM", "LOW"),
              f"Overall analysis confidence: {overall}")

    res.outputs["metrics"] = str(out)
    res.stats = {"n_shots": n_shots, "evidence_entries": len(evidence),
                 "overall_confidence": overall, "benchmarks_used": len(usable_bm)}
    return res
