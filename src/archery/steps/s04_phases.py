"""S4 phases.

IN     03_kinematics.parquet, 03_quality.json, 01_frames.json,
       config/phase_rules.yaml, session manual_phase_overrides
DO     deterministic shot detection, then phase segmentation inside each shot
OUT    04_phases.json
VERIFY at least one shot, exactly one release per shot, phases never overlap,
       every detected phase has a usable key frame, identical input -> identical
       output

How a shot is found (all thresholds in phase_rules.yaml):
  1. ANCHORED   draw wrist within anchor_distance_enter of the mouth, with
                hysteresis up to anchor_distance_exit
  2. RELEASE    draw-wrist speed peak near the end of each anchored run
  3. DRAW       walking back from anchor while the anchor distance shrinks
  4. PRE_DRAW   walking back from draw while the bow arm is still rising
  5. STANCE     everything before that, back to the previous shot
  6. ANCHOR     from anchor until the anchor distance settles
  7. EXPANSION  sustained draw-elbow angle change immediately before release
  8. AIM        between a settled anchor and expansion
  9. FOLLOW_THROUGH, RECOVERY   fixed window after release, then until the bow
                arm drops back

A phase the signals do not support is recorded as detected=false with the
reason. It is never filled in with a guessed boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from archery.context import Context
from archery.contracts import WARN, StepResult
from archery.phase_defs import BADGE, DISPLAY, ORDER


# ---------------------------------------------------------------- helpers
def _frames(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


def anchored_mask(a: np.ndarray, enter: float, exit_: float, max_gap: int) -> np.ndarray:
    """Hysteresis state machine over the anchor-distance signal."""
    out = np.zeros(a.size, dtype=bool)
    state, gap = False, 0
    for i, v in enumerate(a):
        if not np.isfinite(v):
            if state:
                gap += 1
                if gap > max_gap:
                    state = False
            out[i] = state
            continue
        gap = 0
        if not state and v < enter:
            state = True
        elif state and v > exit_:
            state = False
        out[i] = state
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, mask.size - 1))
    return out


def walk_back(cond: np.ndarray, start: int, floor: int, tolerance: int) -> int:
    """From ``start`` walk toward ``floor`` while ``cond`` holds, forgiving up to
    ``tolerance`` consecutive failures. Returns the first index of the run."""
    j, misses, first_ok = start, 0, start + 1
    while j >= floor:
        if cond[j]:
            misses = 0
            first_ok = j
        else:
            misses += 1
            if misses > tolerance:
                break
        j -= 1
    return first_ok


def grade(signal: np.ndarray, idx: int, threshold: float | None, usable: np.ndarray,
          h: int) -> str:
    """HIGH clean crossing, MEDIUM oscillating, LOW missing data around the boundary."""
    lo, hi = max(0, idx - h), min(signal.size, idx + h + 1)
    window = signal[lo:hi]
    if window.size == 0 or not np.all(np.isfinite(window)) or not np.all(usable[lo:hi]):
        return "LOW"
    if threshold is None:
        return "HIGH"
    crossings = int(np.sum(np.diff(np.sign(window - threshold)) != 0))
    return "HIGH" if crossings <= 1 else "MEDIUM"


def key_frame(start: int, end: int, n_visible: np.ndarray, usable: np.ndarray,
              prefer: int | None = None) -> int | None:
    """Best-measured frame in a phase: most visible landmarks, nearest the midpoint."""
    if end < start:
        return None
    idx = np.arange(start, end + 1)
    ok = idx[usable[start:end + 1]]
    if ok.size == 0:
        return None
    if prefer is not None and usable[prefer] and start <= prefer <= end:
        return int(prefer)
    mid = (start + end) / 2.0
    best = max(ok, key=lambda i: (n_visible[i], -abs(i - mid)))
    return int(best)


# ---------------------------------------------------------------- main
def run(ctx: Context) -> StepResult:
    res = StepResult(step="S4")
    rules = ctx.cfg.phase_rules
    sd_, ph, bc = rules["shot_detection"], rules["phases"], rules["boundary_confidence"]

    fps = float(ctx.read_json("01_frames.json")["analysis_fps"])
    k = pd.read_parquet(ctx.artefact("03_kinematics.parquet"))
    n = len(k)
    t = k["t_s"].to_numpy()
    A = k["anchor_distance_norm"].to_numpy()
    A_rate = k["anchor_distance_rate_s"].to_numpy()
    A_sd = k["anchor_distance_rolling_sd"].to_numpy()
    V = k["draw_wrist_speed_norm_s"].to_numpy()
    E_rate = k["draw_elbow_rate_deg_s"].to_numpy()
    B = k["bow_arm_elevation_norm"].to_numpy()
    usable = k["frame_usable"].to_numpy().astype(bool)
    n_visible = k["n_visible_landmarks"].to_numpy()
    h = int(bc["hysteresis_frames"])

    # ---- 1-2. anchored runs and releases ---------------------------------
    mask = anchored_mask(A, sd_["anchor_distance_enter"], sd_["anchor_distance_exit"],
                         _frames(sd_["max_nan_gap_s"], fps))
    min_hold = _frames(sd_["min_anchor_hold_s"], fps)
    anchored = [(s, e) for s, e in runs(mask) if e - s + 1 >= min_hold]

    v_finite = V[np.isfinite(V)]
    v_thresh = (float(v_finite.mean() + sd_["release_velocity_z"] * v_finite.std())
                if v_finite.size else np.inf)

    shots_raw = []
    for s, e in anchored:
        lo = max(s + 1, e - _frames(sd_["release_search_before_s"], fps))
        hi = min(n - 1, e + _frames(sd_["release_search_after_s"], fps))
        seg = V[lo:hi + 1]
        if np.isfinite(seg).any():
            r = lo + int(np.nanargmax(seg))
            r_conf = "HIGH" if V[r] >= v_thresh else "MEDIUM"
        else:
            r, r_conf = min(e + 1, n - 1), "LOW"
        shots_raw.append({"anchor_start": s, "anchor_run_end": e, "release": r,
                          "release_conf": r_conf, "release_peak_speed": float(V[r])
                          if np.isfinite(V[r]) else None})

    wb = _frames(ph["release"]["window_before_s"], fps)
    wa = _frames(ph["release"]["window_after_s"], fps)
    ft = _frames(ph["follow_through"]["duration_after_release_s"], fps)
    overrides = (ctx.session.get("manual_phase_overrides") or {})

    shots, cursor = [], 0
    for idx, sh in enumerate(shots_raw):
        s, r = sh["anchor_start"], sh["release"]
        next_anchor = shots_raw[idx + 1]["anchor_start"] if idx + 1 < len(shots_raw) else n
        win_lo, win_hi = cursor, next_anchor - 1
        bounds: dict[str, dict] = {}

        # ---- 3. DRAW: anchor distance shrinking, walking back from anchor
        shrinking = np.nan_to_num(A_rate, nan=0.0) <= -ph["draw"]["min_decrease_rate_per_s"]
        draw_start = walk_back(shrinking, s - 1, win_lo, h)
        draw_ok = draw_start < s
        bounds["DRAW"] = {"start": draw_start, "end": s - 1, "detected": draw_ok,
                          "driving_signal": "anchor_distance_rate_s",
                          "confidence": grade(A_rate, draw_start,
                                              -ph["draw"]["min_decrease_rate_per_s"], usable, h)
                          if draw_ok else "LOW",
                          "reason": None if draw_ok else
                          "anchor distance never decreased at the required rate before anchoring"}
        if not draw_ok:
            draw_start = s

        # ---- 4. PRE_DRAW: bow arm rising before the draw
        pre_seg = B[win_lo:draw_start + 1]
        b_rest = float(np.nanpercentile(pre_seg, 10)) if np.isfinite(pre_seg).any() else np.nan
        b_draw = float(np.nanmedian(B[draw_start:s + 1])) if np.isfinite(B[draw_start:s + 1]).any() else np.nan
        rise = b_draw - b_rest if np.isfinite(b_rest) and np.isfinite(b_draw) else np.nan
        setup_start, pre_ok, pre_reason = draw_start, False, None
        if np.isfinite(rise) and rise >= ph["pre_draw"]["min_rise_shoulder_widths"]:
            thr = b_rest + ph["pre_draw"]["bow_arm_rise_fraction"] * rise
            below = np.where(np.nan_to_num(B[win_lo:draw_start], nan=np.inf) <= thr)[0]
            if below.size:
                setup_start = win_lo + int(below[-1]) + 1
                pre_ok = setup_start < draw_start
            else:
                pre_reason = "bow arm already raised at the start of the shot window"
        else:
            pre_reason = (f"bow-arm rise {rise:.2f} shoulder widths is below the "
                          f"{ph['pre_draw']['min_rise_shoulder_widths']} needed"
                          if np.isfinite(rise) else "bow-arm elevation signal unavailable")
        bounds["PRE_DRAW"] = {"start": setup_start, "end": draw_start - 1, "detected": pre_ok,
                              "driving_signal": "bow_arm_elevation_norm",
                              "confidence": grade(B, setup_start, None, usable, h) if pre_ok else "LOW",
                              "reason": pre_reason}

        # ---- 5. STANCE
        stance_ok = setup_start > win_lo
        bounds["STANCE"] = {"start": win_lo, "end": setup_start - 1, "detected": stance_ok,
                            "driving_signal": "shot_window_start",
                            "confidence": "HIGH" if stance_ok else "LOW",
                            "reason": None if stance_ok else "no frames before setup in this shot window"}

        # ---- 6-8. ANCHOR, AIM, EXPANSION inside the anchored run
        pre_release_end = max(s, r - wb - 1)
        settled = np.where(np.nan_to_num(A_sd[s:pre_release_end + 1], nan=np.inf)
                           < ph["anchor"]["stability_tolerance"])[0]
        stab = _frames(ph["anchor"]["stability_window_s"], fps)
        settle = s + int(settled[0]) if settled.size else pre_release_end
        anchor_end = min(pre_release_end, max(s + stab - 1, settle))
        bounds["ANCHOR"] = {"start": s, "end": anchor_end, "detected": True,
                            "driving_signal": "anchor_distance_norm",
                            "confidence": grade(A, s, sd_["anchor_distance_enter"], usable, h),
                            "reason": None if settled.size else
                            "anchor distance never settled below the stability tolerance"}

        er = np.nan_to_num(E_rate, nan=0.0)
        sign = np.sign(er[pre_release_end]) or 1.0
        moving = (np.abs(er) >= ph["expansion"]["min_elbow_rate_deg_per_s"]) & (np.sign(er) == sign)
        exp_start = walk_back(moving, pre_release_end, anchor_end + 1, h)
        exp_len_s = (pre_release_end - exp_start + 1) / fps
        exp_ok = (exp_start <= pre_release_end
                  and exp_len_s >= ph["expansion"]["min_duration_s"])
        if exp_ok and exp_len_s > ph["expansion"]["max_duration_s"]:
            exp_start = pre_release_end - _frames(ph["expansion"]["max_duration_s"], fps) + 1
        if not exp_ok:
            exp_start = pre_release_end + 1
        bounds["EXPANSION"] = {"start": exp_start, "end": pre_release_end, "detected": exp_ok,
                               "driving_signal": "draw_elbow_rate_deg_s",
                               "confidence": grade(E_rate, exp_start, None, usable, h) if exp_ok else "LOW",
                               "reason": None if exp_ok else
                               "no sustained draw-elbow movement before release at the required rate"}

        aim_ok = exp_start - 1 > anchor_end
        bounds["AIM"] = {"start": anchor_end + 1, "end": exp_start - 1, "detected": aim_ok,
                         "driving_signal": "anchor_distance_rolling_sd",
                         "confidence": bounds["ANCHOR"]["confidence"] if aim_ok else "LOW",
                         "reason": None if aim_ok else "no settled hold between anchor and expansion"}

        # ---- RELEASE, FOLLOW_THROUGH, RECOVERY
        rel_lo, rel_hi = max(0, r - wb), min(n - 1, r + wa)
        bounds["RELEASE"] = {"start": rel_lo, "end": rel_hi, "detected": True,
                             "driving_signal": "draw_wrist_speed_norm_s",
                             "confidence": sh["release_conf"], "reason": None, "event_frame": r}
        ft_lo, ft_hi = rel_hi + 1, min(n - 1, win_hi, rel_hi + ft)
        bounds["FOLLOW_THROUGH"] = {"start": ft_lo, "end": ft_hi, "detected": ft_hi >= ft_lo,
                                    "driving_signal": "fixed window after release",
                                    "confidence": sh["release_conf"],
                                    "reason": None if ft_hi >= ft_lo else "video ends at release"}

        rec_lo = ft_hi + 1
        rec_cap = min(n - 1, win_hi, rec_lo + _frames(ph["recovery"]["max_duration_s"], fps) - 1)
        rec_hi, rec_conf, rec_reason = rec_cap, "MEDIUM", "bow arm did not drop within the window"
        if np.isfinite(rise) and rise > 0 and rec_lo <= rec_cap:
            thr = b_rest + ph["recovery"]["bow_arm_drop_fraction"] * rise
            dropped = np.where(np.nan_to_num(B[rec_lo:rec_cap + 1], nan=np.inf) <= thr)[0]
            if dropped.size:
                rec_hi, rec_conf, rec_reason = rec_lo + int(dropped[0]), "HIGH", None
        bounds["RECOVERY"] = {"start": rec_lo, "end": rec_hi, "detected": rec_hi >= rec_lo,
                              "driving_signal": "bow_arm_elevation_norm",
                              "confidence": rec_conf if rec_hi >= rec_lo else "LOW",
                              "reason": rec_reason if rec_hi >= rec_lo else "no frames after follow-through"}

        # ---- manual overrides for this shot (seconds), recorded as such
        ov = overrides.get(str(idx + 1)) or overrides.get(idx + 1) or {}
        for code, span in ov.items():
            if code in bounds and isinstance(span, (list, tuple)) and len(span) == 2:
                a0 = int(np.searchsorted(t, float(span[0])))
                a1 = int(np.searchsorted(t, float(span[1]), side="right")) - 1
                bounds[code].update({"start": a0, "end": a1, "detected": True,
                                     "confidence": "MANUAL", "driving_signal": "manual override",
                                     "reason": None})

        # ---- assemble in canonical order, enforce no overlap, pick key frames
        phases, last_end = [], win_lo - 1
        for code in ORDER:
            b = bounds[code]
            start, end = max(int(b["start"]), last_end + 1), int(b["end"])
            detected = bool(b["detected"]) and end >= start
            prefer = b.get("event_frame")
            kf = key_frame(start, end, n_visible, usable, prefer) if detected else None
            phases.append({
                "phase": code, "badge": BADGE[code], "display_name": DISPLAY[code],
                "detected": detected,
                "start_frame": start if detected else None,
                "end_frame": end if detected else None,
                "start_t_s": round(float(t[start]), 3) if detected else None,
                "end_t_s": round(float(t[end]), 3) if detected else None,
                "duration_s": round((end - start + 1) / fps, 3) if detected else None,
                "key_frame": kf,
                "key_frame_t_s": round(float(t[kf]), 3) if kf is not None else None,
                "boundary_confidence": b["confidence"] if detected else None,
                "driving_signal": b["driving_signal"],
                "reason_not_detected": None if detected else (b.get("reason") or "zero-length after ordering"),
            })
            if detected:
                last_end = end
        cursor = last_end + 1
        shots.append({
            "shot": idx + 1,
            "release_frame": r, "release_t_s": round(float(t[r]), 3),
            "release_confidence": sh["release_conf"],
            "release_peak_draw_wrist_speed": (round(sh["release_peak_speed"], 3)
                                              if sh["release_peak_speed"] is not None else None),
            "window": [win_lo, win_hi],
            "source": "manual+rules" if ov else "rules",
            "phases": phases,
        })

    # ---- signal summary: the numbers to tune thresholds from ---------------
    def pct(x):
        x = x[np.isfinite(x)]
        return ({f"p{q}": round(float(np.percentile(x, q)), 4) for q in (1, 5, 25, 50, 75, 95, 99)}
                if x.size else None)

    payload = {
        "n_shots": len(shots),
        "analysis_fps": fps,
        "rules_version": rules.get("version"),
        "thresholds_used": rules,
        "release_speed_threshold": round(v_thresh, 4) if np.isfinite(v_thresh) else None,
        "shots": shots,
        "signal_summary": {
            "anchor_distance_norm": pct(A),
            "anchor_distance_rate_s": pct(A_rate),
            "draw_wrist_speed_norm_s": pct(V),
            "draw_elbow_rate_deg_s": pct(E_rate),
            "bow_arm_elevation_norm": pct(B),
            "frames_anchored": int(mask.sum()),
            "anchored_runs_before_min_hold_filter": len(runs(mask)),
        },
    }
    out = ctx.write_json("04_phases.json", payload)

    # ---- verification -------------------------------------------------------
    min_a = float(np.nanmin(A)) if np.isfinite(A).any() else None
    res.check("shots_detected", len(shots) > 0,
              f"{len(shots)} shot(s) detected." if shots else
              f"No shot detected. Smallest anchor distance on this footage is "
              f"{min_a if min_a is None else round(min_a, 3)} shoulder widths against an enter "
              f"threshold of {sd_['anchor_distance_enter']}. Read signal_summary in "
              f"04_phases.json and adjust config/phase_rules.yaml.")
    expected = ctx.session.get("number_of_shots_expected")
    if isinstance(expected, int):
        res.check("shot_count_matches_expected", len(shots) == expected,
                  f"Detected {len(shots)}, session says {expected}.", severity=WARN)

    overlap, gaps, missing_kf, undetected = [], [], [], []
    for sh in shots:
        prev = None
        for p in sh["phases"]:
            if not p["detected"]:
                undetected.append(f"shot {sh['shot']} {p['phase']}: {p['reason_not_detected']}")
                continue
            if p["key_frame"] is None:
                missing_kf.append(f"shot {sh['shot']} {p['phase']}")
            if prev is not None:
                if p["start_frame"] <= prev["end_frame"]:
                    overlap.append(f"shot {sh['shot']} {prev['phase']}/{p['phase']}")
                elif p["start_frame"] > prev["end_frame"] + 1:
                    gaps.append(f"shot {sh['shot']} {prev['phase']}->{p['phase']}")
            prev = p
        rel = [p for p in sh["phases"] if p["phase"] == "RELEASE" and p["detected"]]
        res.check(f"shot{sh['shot']}_single_release", len(rel) == 1,
                  f"release at {sh['release_t_s']} s ({sh['release_confidence']})")
    res.check("phases_do_not_overlap", not overlap, f"Overlaps: {overlap}" if overlap else "No overlaps.")
    res.check("phases_contiguous", not gaps, f"Gaps: {gaps}" if gaps else "No gaps.", severity=WARN)
    res.check("key_frames_usable", not missing_kf,
              f"No usable key frame for: {missing_kf}" if missing_kf else
              "Every detected phase has a key frame with adequate landmark quality.", severity=WARN)
    res.check("all_phases_detected", not undetected,
              "; ".join(undetected) if undetected else "All nine phases detected in every shot.",
              severity=WARN)

    res.outputs["phases"] = str(out)
    res.stats = {"n_shots": len(shots),
                 "release_t_s": [s["release_t_s"] for s in shots],
                 "undetected_phases": len(undetected)}
    return res
