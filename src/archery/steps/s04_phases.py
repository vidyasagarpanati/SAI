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
                hysteresis up to anchor_distance_exit. Anchored runs split by a
                gap shorter than merge_anchor_gap_s are one hold (tracking flicker)
  2. RELEASE    draw-wrist speed peak near the end of each anchored run. It is
                a real release when it clears the clip-wide threshold or is
                release_vs_draw_speed_ratio times this shot's own draw speed
  3. DRAW       walking back from anchor while the anchor distance shrinks
  4. PRE_DRAW   walking back from draw while the bow arm is still rising
  5. STANCE     everything before that, back to the previous shot
  6. ANCHOR     from anchor until the anchor distance settles
  7. EXPANSION  sustained draw-elbow angle change immediately before release
  8. AIM        between a settled anchor and expansion
  9. FOLLOW_THROUGH, RECOVERY   fixed window after release, then until the bow
                arm drops back
 10. SHOT CYCLE the timed shot, hook-and-grip to release (see shot_cycle in
                phase_rules.yaml). Recorded beside the phases; it moves none of them.
 11. COMPLETE   a candidate counts as a shot only when the whole sequence is
                there: a real release (speed spike) and every step listed in
                shot_detection.required_steps. A broken sequence (a let-down, a
                hand passing the face, a draw with no release) is recorded under
                attempts_not_counted with what was missing, and its frames fall
                to the next shot's window. Two broken pieces, the first with no
                release and the second with no draw of its own, less than
                join_fragment_gap_s apart, are one hold split by a tracking loss
                and are joined first. Windows are rebuilt until every remaining
                candidate is complete.

A phase the signals do not support is recorded as detected=false with the
reason. It is never filled in with a guessed boundary.
"""
from __future__ import annotations

import copy

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


def slow_motion_rules(rules: dict, factor: float) -> dict:
    """The rules for a slow-motion video: shot-detection windows widened.

    S1 has already put every time back into real seconds, so the rules hold as
    written. The windows are widened on top of that because the factor is the
    operator's estimate (an "8x" clip may really be 7.5x) and the release is
    spread over many more frames. A normal video (factor 1) gets the rules
    unchanged."""
    sm = rules.get("slow_motion") or {}
    if factor <= 1 or not sm:
        return rules
    out = copy.deepcopy(rules)
    scale = float(sm.get("window_scale", 1.0))
    for section, key in (("shot_detection", "release_search_before_s"),
                         ("shot_detection", "release_search_after_s"),
                         ("shot_detection", "merge_anchor_gap_s"),
                         ("shot_detection", "join_fragment_gap_s"),
                         ("shot_detection", "max_nan_gap_s"),
                         ("shot_cycle", "hook_max_lead_s")):
        if key in (out.get(section) or {}):
            out[section][key] = float(out[section][key]) * scale
    if "release_velocity_z" in sm:
        out["shot_detection"]["release_velocity_z"] = float(sm["release_velocity_z"])
    out["slow_motion_applied"] = {"factor": factor, **sm}
    return out


# ---------------------------------------------------------------- main
def run(ctx: Context) -> StepResult:
    res = StepResult(step="S4")
    frames_meta = ctx.read_json("01_frames.json")
    fps = float(frames_meta["analysis_fps"])
    slow = float(frames_meta.get("slow_motion_factor") or 1.0)
    rules = slow_motion_rules(ctx.cfg.phase_rules, slow)
    sd_, ph, bc = rules["shot_detection"], rules["phases"], rules["boundary_confidence"]
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
    VB = k["bow_wrist_speed_norm_s"].to_numpy()
    # Runs analysed before the hand-gap signal existed have no column; every
    # shot then starts at setup, flagged, rather than failing.
    G_gap = (k["hand_gap_norm"].to_numpy() if "hand_gap_norm" in k.columns
             else np.full(n, np.nan))
    h = int(bc["hysteresis_frames"])
    cyc = rules.get("shot_cycle") or {}

    # ---- 1-2. anchored runs and releases ---------------------------------
    mask = anchored_mask(A, sd_["anchor_distance_enter"], sd_["anchor_distance_exit"],
                         _frames(sd_["max_nan_gap_s"], fps))
    min_hold = _frames(sd_["min_anchor_hold_s"], fps)
    merge_gap = _frames(sd_.get("merge_anchor_gap_s", 0.0), fps) if sd_.get("merge_anchor_gap_s") else 0
    merged: list[list[int]] = []
    for s, e in runs(mask):
        if merged and s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    anchored = [(s, e) for s, e in merged if e - s + 1 >= min_hold]

    v_finite = V[np.isfinite(V)]
    v_thresh = (float(v_finite.mean() + sd_["release_velocity_z"] * v_finite.std())
                if v_finite.size else np.inf)

    def candidate(s: int, e: int) -> dict:
        lo = max(s + 1, e - _frames(sd_["release_search_before_s"], fps))
        hi = min(n - 1, e + _frames(sd_["release_search_after_s"], fps))
        seg = V[lo:hi + 1]
        if np.isfinite(seg).any():
            r = lo + int(np.nanargmax(seg))
            r_conf = "HIGH" if V[r] >= v_thresh else "MEDIUM"
        else:
            r, r_conf = min(e + 1, n - 1), "LOW"
        return {"anchor_start": s, "anchor_run_end": e, "release": r,
                "release_conf": r_conf, "release_peak_speed": float(V[r])
                if np.isfinite(V[r]) else None}

    shots_raw = [candidate(s, e) for s, e in anchored]
    draw_ratio = sd_.get("release_vs_draw_speed_ratio")

    wb = _frames(ph["release"]["window_before_s"], fps)
    wa = _frames(ph["release"]["window_after_s"], fps)
    ft = _frames(ph["follow_through"]["duration_after_release_s"], fps)
    overrides = (ctx.session.get("manual_phase_overrides") or {})

    def build(cands: list[dict]) -> list[dict]:
        shots, cursor = [], 0
        for idx, sh in enumerate(cands):
            s, r = sh["anchor_start"], sh["release"]
            next_anchor = cands[idx + 1]["anchor_start"] if idx + 1 < len(cands) else n
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

            # A release is the string snapping the hand away: much faster than
            # this shot's own draw. A let-down eases off no faster than it drew.
            # The global threshold alone fails on long clips, where walking and
            # fetching arrows inflate the spread of hand speeds.
            draw_v = V[draw_start:s]
            draw_v = draw_v[np.isfinite(draw_v)]
            peak = sh["release_peak_speed"]
            draw_p90 = float(np.percentile(draw_v, 90)) if draw_ok and draw_v.size else None
            spike = sh["release_conf"] == "HIGH" or bool(
                draw_ratio and draw_p90 and peak is not None and peak >= draw_ratio * draw_p90)

            # ---- 4. PRE_DRAW: bow arm rising before the draw
            pre_seg = B[win_lo:draw_start + 1]
            b_rest = float(np.nanpercentile(pre_seg, 10)) if np.isfinite(pre_seg).any() else np.nan
            b_draw = float(np.nanmedian(B[draw_start:s + 1])) if np.isfinite(B[draw_start:s + 1]).any() else np.nan
            rise = b_draw - b_rest if np.isfinite(b_rest) and np.isfinite(b_draw) else np.nan
            setup_start, pre_ok, pre_reason = draw_start, False, None
            # The setup movement happened (bow arm down, then raised) even when
            # a noisy draw boundary leaves the PRE_DRAW phase zero frames long.
            setup_seen = False
            if np.isfinite(rise) and rise >= ph["pre_draw"]["min_rise_shoulder_widths"]:
                thr = b_rest + ph["pre_draw"]["bow_arm_rise_fraction"] * rise
                below = np.where(np.nan_to_num(B[win_lo:draw_start], nan=np.inf) <= thr)[0]
                if below.size:
                    setup_start = win_lo + int(below[-1]) + 1
                    pre_ok = setup_start < draw_start
                    setup_seen = True
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
            # Follow-through means the bow arm stays up after the release. An
            # arm already dropping is a broken sequence, not a follow-through.
            ft_ok, ft_reason = ft_hi >= ft_lo, None if ft_hi >= ft_lo else "video ends at release"
            held_min = ph["follow_through"].get("min_bow_arm_held_fraction")
            if ft_ok and held_min and np.isfinite(rise) and rise > 0:
                thr = b_rest + ph["recovery"]["bow_arm_drop_fraction"] * rise
                held = float(np.mean(np.nan_to_num(B[ft_lo:ft_hi + 1], nan=-np.inf) > thr))
                if held < held_min:
                    ft_ok = False
                    ft_reason = (f"bow arm up for only {held:.0%} of the follow-through window "
                                 f"(at least {held_min:.0%} needed)")
            bounds["FOLLOW_THROUGH"] = {"start": ft_lo, "end": ft_hi, "detected": ft_ok,
                                        "driving_signal": "fixed window after release; bow_arm_elevation_norm",
                                        "confidence": sh["release_conf"] if ft_ok else "LOW",
                                        "reason": ft_reason}

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

            # ---- shot cycle: hook and grip, then release ---------------------
            gap = np.nan_to_num(G_gap, nan=np.inf)
            close = gap <= cyc.get("hook_max_hand_gap", 1.0)
            still = ((np.nan_to_num(V, nan=np.inf) <= cyc.get("hook_max_hand_speed", 0.5))
                     & (np.nan_to_num(VB, nan=np.inf) <= cyc.get("hook_max_hand_speed", 0.5)))
            lead_lo = max(win_lo, setup_start - _frames(cyc.get("hook_max_lead_s", 8.0), fps))
            candidates = np.where((close & still)[lead_lo:setup_start])[0]
            hook = None
            if candidates.size:
                last = lead_lo + int(candidates[-1])
                first = walk_back(close, last, win_lo, h)
                if last - first + 1 >= _frames(cyc.get("hook_min_hold_s", 0.3), fps):
                    hook = first
            if "HOOK" in ov and isinstance(ov["HOOK"], (int, float)):
                c0, event, c_conf = int(np.searchsorted(t, float(ov["HOOK"]))), "MANUAL", "MANUAL"
            elif hook is not None:
                c0, event = hook, "HOOK_GRIP"
                c_conf = grade(G_gap, hook, cyc.get("hook_max_hand_gap", 1.0), usable, h)
            else:
                c0, event, c_conf = setup_start, "SETUP_FALLBACK", "LOW"
            cycle = {"start_event": event, "start_frame": c0, "start_t_s": round(float(t[c0]), 3),
                     "start_confidence": c_conf,
                     "end_event": "RELEASE", "end_frame": r, "end_t_s": round(float(t[r]), 3),
                     "end_confidence": sh["release_conf"],
                     "duration_s": round((r - c0) / fps, 3),
                     "reason": None if event != "SETUP_FALLBACK" else
                     "hook and grip not seen (hands not together and still before setup); "
                     "the shot is timed from the bow arm starting to rise"}

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
                "draw_wrist_speed_p90_during_draw": round(draw_p90, 3) if draw_p90 else None,
                "release_spike": spike or "RELEASE" in ov,
                "window": [win_lo, win_hi],
                "source": "manual+rules" if ov else "rules",
                "cycle": cycle,
                "phases": phases,
                # Movements the signals support, before phases are trimmed to
                # fit together. Completeness is judged on these.
                "steps_seen": sorted({code for code in ORDER if bounds[code]["detected"]
                                      and bounds[code]["end"] >= bounds[code]["start"]}
                                     | ({"PRE_DRAW"} if setup_seen else set())
                                     | {p["phase"] for p in phases if p["detected"]}),
            })

        return shots

    # ---- 11. only complete sequences count as shots ------------------------
    required = list(sd_.get("required_steps") or [])
    need_spike = bool(sd_.get("require_release_spike", False))

    def missing_steps(shot: dict) -> list[str]:
        found = set(shot["steps_seen"])
        if shot["cycle"]["start_event"] in ("HOOK_GRIP", "MANUAL"):
            found.add("HOOK")
        out = [code for code in required if code not in found]
        if need_spike and not shot["release_spike"] and "RELEASE" not in out:
            out.append("RELEASE")
        return out

    join_gap = _frames(sd_.get("join_fragment_gap_s") or 0.0, fps)

    def join_fragments(cands: list[dict], shots: list[dict]) -> bool:
        """One hold broken in two by a long tracking loss: the first piece has
        no release, the second has no draw of its own (the archer never drew
        again, so it is not a let-down and re-draw). Joined in place."""
        for i in range(len(shots) - 1):
            a, b = cands[i], cands[i + 1]
            if (not shots[i]["release_spike"] and missing_steps(shots[i])
                    and missing_steps(shots[i + 1])
                    and "DRAW" not in shots[i + 1]["steps_seen"]
                    and b["anchor_start"] - a["anchor_run_end"] - 1 <= join_gap):
                cands[i:i + 2] = [candidate(a["anchor_start"], b["anchor_run_end"])]
                return True
        return False

    def attempt(c: dict, shot: dict, missing: list[str], why: str) -> dict:
        return {"anchor_t_s": round(float(t[c["anchor_start"]]), 3),
                "release_t_s": round(float(t[c["release"]]), 3),
                "release_peak_draw_wrist_speed": (round(c["release_peak_speed"], 3)
                                                  if c["release_peak_speed"] is not None else None),
                "draw_wrist_speed_p90_during_draw": shot["draw_wrist_speed_p90_during_draw"],
                "missing": missing, "reason": why}

    # Rebuilt until stable: dropping a candidate widens its neighbours' windows,
    # which can change what they find. Candidates only ever join or leave, so
    # the list shrinks every pass and this ends.
    kept, attempts = list(shots_raw), []
    while True:
        shots = build(kept)
        if join_gap and join_fragments(kept, shots):
            continue
        bad = [i for i, sh in enumerate(shots) if missing_steps(sh)]
        if not bad:
            break
        for i in reversed(bad):
            miss = missing_steps(shots[i])
            why = ("no release: draw-hand speed at the end of the hold was no faster "
                   "than a let-down"
                   if miss == ["RELEASE"] else
                   "sequence incomplete: " + ", ".join(DISPLAY.get(m, m) for m in miss) + " not seen")
            attempts.append(attempt(kept.pop(i), shots[i], miss, why))
    attempts.sort(key=lambda a: a["anchor_t_s"])

    # ---- signal summary: the numbers to tune thresholds from ---------------
    def pct(x):
        x = x[np.isfinite(x)]
        return ({f"p{q}": round(float(np.percentile(x, q)), 4) for q in (1, 5, 25, 50, 75, 95, 99)}
                if x.size else None)

    payload = {
        "n_shots": len(shots),
        "analysis_fps": fps,
        "slow_motion_factor": slow,
        "rules_version": rules.get("version"),
        "thresholds_used": rules,
        "release_speed_threshold": round(v_thresh, 4) if np.isfinite(v_thresh) else None,
        "shots": shots,
        "required_steps": required + (["RELEASE spike"] if need_spike else []),
        "attempts_not_counted": attempts,
        "signal_summary": {
            "anchor_distance_norm": pct(A),
            "anchor_distance_rate_s": pct(A_rate),
            "draw_wrist_speed_norm_s": pct(V),
            "draw_elbow_rate_deg_s": pct(E_rate),
            "bow_arm_elevation_norm": pct(B),
            "hand_gap_norm": pct(G_gap),
            "frames_anchored": int(mask.sum()),
            "anchored_runs_before_min_hold_filter": len(runs(mask)),
            "anchored_runs_after_merge_and_hold": len(anchored),
        },
    }
    out = ctx.write_json("04_phases.json", payload)

    # ---- verification -------------------------------------------------------
    min_a = float(np.nanmin(A)) if np.isfinite(A).any() else None
    res.check("shots_detected", len(shots) > 0,
              f"{len(shots)} shot(s) detected." if shots else
              (f"No complete shot. {len(attempts)} attempt(s) were found but each broke the "
               f"sequence: " + "; ".join(f"{a['anchor_t_s']} s {a['reason']}" for a in attempts)
               + ". Read attempts_not_counted and signal_summary in 04_phases.json."
               if attempts else
               f"No shot detected. Smallest anchor distance on this footage is "
               f"{min_a if min_a is None else round(min_a, 3)} shoulder widths against an enter "
               f"threshold of {sd_['anchor_distance_enter']}. Read signal_summary in "
               f"04_phases.json and adjust config/phase_rules.yaml."))
    res.check("incomplete_attempts_not_counted", not attempts,
              f"{len(attempts)} attempt(s) not counted as shots: "
              + "; ".join(f"{a['anchor_t_s']} s {a['reason']}" for a in attempts)
              if attempts else "Every candidate was a complete shot.", severity=WARN)
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
    fallback = [s["shot"] for s in shots if s["cycle"]["start_event"] == "SETUP_FALLBACK"]
    res.check("shot_start_hook_seen", not fallback,
              f"Hook and grip not seen for shot(s) {fallback}; timed from setup instead, "
              f"LOW confidence. Tune shot_cycle in phase_rules.yaml from "
              f"signal_summary.hand_gap_norm." if fallback else
              "Every shot is timed from hook and grip to release.", severity=WARN)
    res.check("all_phases_detected", not undetected,
              "; ".join(undetected) if undetected else "All nine phases detected in every shot.",
              severity=WARN)

    res.outputs["phases"] = str(out)
    res.stats = {"n_shots": len(shots), "attempts_not_counted": len(attempts),
                 "release_t_s": [s["release_t_s"] for s in shots],
                 "undetected_phases": len(undetected)}
    return res
