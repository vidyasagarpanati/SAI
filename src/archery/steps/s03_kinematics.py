"""S3 kinematics.

IN     02_landmarks.parquet, 02_pose_quality.json
DO     joint angles, anatomical reference frames, reference lines, normalised
       signals for phase detection
OUT    03_kinematics.parquet, 03_quality.json
VERIFY every angle gated on landmark confidence, NaN rate reported per angle,
       anatomical plausibility bounds asserted, draw-hand setting cross-checked
       against the observed geometry

Coordinate policy, stated once and applied everywhere:
  - True joint angles (elbow, shoulder, hip, knee, ankle, wrist) are computed in
    MediaPipe WORLD space, metres, because image space distorts them with
    perspective.
  - Orientation measures against gravity or the image frame (trunk inclination,
    pelvic tilt, shoulder tilt, head tilt) are computed in IMAGE space, because
    that is where the camera's vertical actually lives.
  - Distances used as signals are normalised by shoulder width so they are
    independent of camera distance and subject size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from archery import geometry as G
from archery import landmarks as L
from archery.context import Context
from archery.contracts import WARN, StepResult
from archery.io_guard import guarded_path

# Anatomical plausibility bounds. A value outside these is not silently dropped,
# it is counted and reported, because a systematic excursion means the pose is
# wrong, not the archer.
PLAUSIBLE = {
    "elbow": (0.0, 190.0),
    "shoulder": (0.0, 190.0),
    "wrist": (0.0, 190.0),
    "hip": (0.0, 190.0),
    "knee": (0.0, 190.0),
    "ankle": (0.0, 190.0),
    "trunk_inclination": (-60.0, 60.0),
    "neck_inclination": (-70.0, 70.0),
    "head_tilt": (-60.0, 60.0),
    "shoulder_tilt": (-45.0, 45.0),
    "pelvic_tilt": (-45.0, 45.0),
    "shoulder_hip_separation": (-60.0, 60.0),
}

# Longest key first, so "shoulder_tilt" wins over "shoulder" for shoulder_tilt_deg.
_PLAUSIBLE_KEYS = sorted(PLAUSIBLE, key=len, reverse=True)


def _bounds_for(measure: str) -> tuple[float, float] | None:
    for key in _PLAUSIBLE_KEYS:
        if measure.startswith(key):
            return PLAUSIBLE[key]
    return None

GATE_OK = "OK"
GATE_NO_DETECTION = "NO_DETECTION"
GATE_INSUFFICIENT = "INSUFFICIENT_LANDMARKS"
GATE_LOW_CONFIDENCE = "LOW_CONFIDENCE"


def _load_arrays(ctx: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    df = pd.read_parquet(ctx.artefact("02_landmarks.parquet"))
    n_frames = int(df["frame"].max()) + 1
    img = df[["x", "y"]].to_numpy(np.float64).reshape(n_frames, 33, 2)
    world = df[["wx", "wy", "wz"]].to_numpy(np.float64).reshape(n_frames, 33, 3)
    vis = df["visibility"].to_numpy(np.float64).reshape(n_frames, 33)
    t_ms = df["t_ms"].to_numpy(np.float64).reshape(n_frames, 33)[:, 0]
    return img, world, vis, t_ms


def _gate(vis: np.ndarray, ids: list[int], min_conf: float,
          frame_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask of usable frames, reason code array) for one measurement."""
    conf_ok = np.all(vis[:, ids] >= min_conf, axis=1)
    detected = np.isfinite(vis[:, ids]).all(axis=1)
    usable = conf_ok & frame_ok & detected
    reason = np.where(~detected, GATE_NO_DETECTION,
             np.where(~frame_ok, GATE_INSUFFICIENT,
             np.where(~conf_ok, GATE_LOW_CONFIDENCE, GATE_OK)))
    return usable, reason


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S3")
    frames_meta = ctx.read_json("01_frames.json")
    pose_quality = ctx.read_json("02_pose_quality.json")
    analysis_fps = float(frames_meta["analysis_fps"])
    dt = 1.0 / analysis_fps

    img, world, vis, t_ms = _load_arrays(ctx)
    n = img.shape[0]

    min_conf = float(ctx.cfg.get("quality_gates.angle_min_confidence", 0.4))
    min_visible = int(ctx.cfg.get("quality_gates.min_visible_landmarks", 20))
    track_conf = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))

    n_visible = np.nansum(vis >= track_conf, axis=1)
    frame_ok = n_visible >= min_visible

    draw = L.side(L.draw_side(ctx.draw_hand))
    bow = L.side(L.bow_side(ctx.draw_hand))
    ID = L.ID

    out: dict[str, np.ndarray] = {
        "frame": np.arange(n),
        "t_ms": t_ms,
        "t_s": t_ms / 1000.0,
        "n_visible_landmarks": n_visible,
        "frame_usable": frame_ok,
    }
    reasons: dict[str, np.ndarray] = {}
    gated: list[str] = []

    def add_angle(name: str, vertex: int, p1: int, p2: int, space: str = "world") -> None:
        ids = [vertex, p1, p2]
        usable, reason = _gate(vis, ids, min_conf, frame_ok)
        src = world if space == "world" else img
        values = G.angle_at(src[:, vertex], src[:, p1], src[:, p2])
        values = np.where(usable, values, np.nan)
        out[name] = values
        reasons[name] = reason
        gated.append(name)

    # -- true joint angles, world space -------------------------------------
    for label, s in (("bow", bow), ("draw", draw)):
        add_angle(f"elbow_{label}_deg", s["elbow"], s["shoulder"], s["wrist"])
        add_angle(f"shoulder_{label}_deg", s["shoulder"], s["elbow"], s["hip"])
        add_angle(f"wrist_{label}_deg", s["wrist"], s["elbow"], s["index"])
    for label, prefix in (("left", "LEFT"), ("right", "RIGHT")):
        s = L.side(prefix)
        add_angle(f"hip_{label}_deg", s["hip"], s["shoulder"], s["knee"])
        add_angle(f"knee_{label}_deg", s["knee"], s["hip"], s["ankle"])
        add_angle(f"ankle_{label}_deg", s["ankle"], s["knee"], s["foot"])

    # -- orientation measures, image space ----------------------------------
    sh_l, sh_r = ID["LEFT_SHOULDER"], ID["RIGHT_SHOULDER"]
    hip_l, hip_r = ID["LEFT_HIP"], ID["RIGHT_HIP"]
    ear_l, ear_r = ID["LEFT_EAR"], ID["RIGHT_EAR"]

    shoulder_mid = (img[:, sh_l] + img[:, sh_r]) / 2.0
    pelvis_mid = (img[:, hip_l] + img[:, hip_r]) / 2.0

    u, r = _gate(vis, [sh_l, sh_r], min_conf, frame_ok)
    out["shoulder_tilt_deg"] = np.where(u, G.signed_tilt(img[:, sh_l], img[:, sh_r]), np.nan)
    reasons["shoulder_tilt_deg"] = r
    gated.append("shoulder_tilt_deg")

    u, r = _gate(vis, [hip_l, hip_r], min_conf, frame_ok)
    out["pelvic_tilt_deg"] = np.where(u, G.signed_tilt(img[:, hip_l], img[:, hip_r]), np.nan)
    reasons["pelvic_tilt_deg"] = r
    gated.append("pelvic_tilt_deg")

    u, r = _gate(vis, [ear_l, ear_r], min_conf, frame_ok)
    out["head_tilt_deg"] = np.where(u, G.signed_tilt(img[:, ear_l], img[:, ear_r]), np.nan)
    reasons["head_tilt_deg"] = r
    gated.append("head_tilt_deg")

    u, r = _gate(vis, [sh_l, sh_r, hip_l, hip_r], min_conf, frame_ok)
    out["trunk_inclination_deg"] = np.where(
        u, G.line_angle_from_vertical(shoulder_mid, pelvis_mid), np.nan)
    reasons["trunk_inclination_deg"] = r
    gated.append("trunk_inclination_deg")

    u, r = _gate(vis, [ID["NOSE"], sh_l, sh_r], min_conf, frame_ok)
    out["neck_inclination_deg"] = np.where(
        u, G.line_angle_from_vertical(img[:, ID["NOSE"]], shoulder_mid), np.nan)
    reasons["neck_inclination_deg"] = r
    gated.append("neck_inclination_deg")

    # Shoulder-to-hip separation: how far the shoulder line is rotated off the
    # pelvic line in the image plane. A proxy for trunk rotation, not a true one.
    out["shoulder_hip_separation_deg"] = out["shoulder_tilt_deg"] - out["pelvic_tilt_deg"]
    gated.append("shoulder_hip_separation_deg")

    # -- normalisation scale -------------------------------------------------
    shoulder_width = G.distance(img[:, sh_l], img[:, sh_r])
    shoulder_width = np.where(shoulder_width > 1e-4, shoulder_width, np.nan)
    out["shoulder_width_norm"] = shoulder_width

    # -- signals used by S4 phase detection ----------------------------------
    anchor_ref = (img[:, ID["MOUTH_LEFT"]] + img[:, ID["MOUTH_RIGHT"]]) / 2.0
    draw_wrist = img[:, draw["wrist"]]
    bow_wrist = img[:, bow["wrist"]]

    u, r = _gate(vis, [draw["wrist"], ID["MOUTH_LEFT"], ID["MOUTH_RIGHT"], sh_l, sh_r],
                 min_conf, frame_ok)
    out["anchor_distance_norm"] = np.where(
        u, G.distance(draw_wrist, anchor_ref) / shoulder_width, np.nan)
    reasons["anchor_distance_norm"] = r
    gated.append("anchor_distance_norm")

    # Speeds are differentiated from SMOOTHED positions. Differencing raw landmark
    # positions amplifies MediaPipe's frame-to-frame jitter and reports noise as
    # movement (found by inspecting rendered frames: a held aim read ~1 SW/s).
    sw_med = np.nanmedian(shoulder_width)
    win_p = int(ctx.cfg.get("smoothing.window", 9))
    ord_p = int(ctx.cfg.get("smoothing.polyorder", 2))

    def smooth_xy(p: np.ndarray) -> np.ndarray:
        return np.stack([G.smooth_nan(p[:, 0], win_p, ord_p),
                         G.smooth_nan(p[:, 1], win_p, ord_p)], axis=1)

    out["draw_wrist_speed_norm_s"] = G.speed(smooth_xy(draw_wrist), dt) / sw_med
    out["bow_wrist_speed_norm_s"] = G.speed(smooth_xy(bow_wrist), dt) / sw_med

    out["com_x_norm"] = pelvis_mid[:, 0]
    out["com_y_norm"] = pelvis_mid[:, 1]
    out["com_speed_norm_s"] = G.speed(smooth_xy(pelvis_mid), dt) / sw_med

    u, r = _gate(vis, [ID["LEFT_HEEL"], ID["RIGHT_HEEL"], sh_l, sh_r], min_conf, frame_ok)
    out["stance_width_norm"] = np.where(
        u, G.distance(img[:, ID["LEFT_HEEL"]], img[:, ID["RIGHT_HEEL"]]) / shoulder_width, np.nan)
    reasons["stance_width_norm"] = r
    gated.append("stance_width_norm")

    # Bow-arm elevation: how far the bow wrist sits above the bow shoulder,
    # normalised. Drives the setup and recovery phase rules.
    u, r = _gate(vis, [bow["wrist"], bow["shoulder"], sh_l, sh_r], min_conf, frame_ok)
    out["bow_arm_elevation_norm"] = np.where(
        u, (img[:, bow["shoulder"], 1] - img[:, bow["wrist"], 1]) / shoulder_width, np.nan)
    reasons["bow_arm_elevation_norm"] = r
    gated.append("bow_arm_elevation_norm")

    # -- smoothing -----------------------------------------------------------
    smoothing = bool(ctx.cfg.get("smoothing.enabled", True))
    window = int(ctx.cfg.get("smoothing.window", 9))
    polyorder = int(ctx.cfg.get("smoothing.polyorder", 2))
    smoothed_cols = []
    if smoothing and str(ctx.cfg.get("smoothing.method", "savgol")) == "savgol":
        for name in list(out.keys()):
            if name in ("frame", "t_ms", "t_s", "n_visible_landmarks", "frame_usable",
                        "draw_wrist_speed_norm_s", "bow_wrist_speed_norm_s", "com_speed_norm_s"):
                continue
            out[name] = G.smooth_nan(out[name], window, polyorder)
            smoothed_cols.append(name)

    # -- derivatives after smoothing ----------------------------------------
    out["anchor_distance_rate_s"] = G.rate_of_change(out["anchor_distance_norm"], dt)
    out["draw_elbow_rate_deg_s"] = G.rate_of_change(out["elbow_draw_deg"], dt)
    out["anchor_distance_rolling_sd"] = G.rolling_std(
        out["anchor_distance_norm"], max(3, int(round(0.1 * analysis_fps))))

    df_out = pd.DataFrame(out)
    out_parquet = guarded_path(ctx.artefact("03_kinematics.parquet"))
    df_out.to_parquet(out_parquet, index=False)

    # -- quality accounting --------------------------------------------------
    nan_rates = {}
    dominant_reason = {}
    for name in gated:
        col = out[name]
        nan_rates[name] = float(np.mean(~np.isfinite(col)))
        if name in reasons:
            bad = reasons[name][~np.isfinite(col)]
            if bad.size:
                values, counts = np.unique(bad, return_counts=True)
                dominant_reason[name] = str(values[int(np.argmax(counts))])

    implausible = {}
    for name in gated:
        bounds = _bounds_for(name)
        if bounds is None:
            continue
        lo, hi = bounds
        finite = out[name][np.isfinite(out[name])]
        if finite.size:
            bad = int(np.sum((finite < lo) | (finite > hi)))
            if bad:
                implausible[name] = {"count": bad, "share": bad / finite.size,
                                     "bounds": [lo, hi]}

    # -- draw-hand cross-check ----------------------------------------------
    # At the most-anchored frame the bow arm is the extended one. If the draw arm
    # reads as more extended, draw_hand in session.json is probably inverted.
    draw_hand_check_detail = "Not assessable: no usable anchored frame."
    draw_hand_ok = True
    anchor_series = out["anchor_distance_norm"]
    if np.isfinite(anchor_series).any():
        k = int(np.nanargmin(anchor_series))
        e_bow, e_draw = out["elbow_bow_deg"][k], out["elbow_draw_deg"][k]
        if np.isfinite(e_bow) and np.isfinite(e_draw):
            draw_hand_ok = e_bow > e_draw
            draw_hand_check_detail = (
                f"At the most-anchored frame ({k}, t={out['t_s'][k]:.2f}s) the "
                f"bow elbow reads {e_bow:.1f} deg and the draw elbow {e_draw:.1f} deg. "
                f"The bow arm should be the more extended one. "
                f"session.json says draw_hand={ctx.draw_hand!r}."
                + ("" if draw_hand_ok else " This looks inverted. Check the setting.")
            )

    quality = {
        "n_frames": n,
        "analysis_fps": analysis_fps,
        "coordinate_policy": {
            "joint_angles": "MediaPipe world landmarks (metres, hip-centred)",
            "orientation_measures": "image space (camera vertical)",
            "distances": "normalised by shoulder width",
        },
        "draw_hand": ctx.draw_hand,
        "bow_side": L.bow_side(ctx.draw_hand),
        "draw_side": L.draw_side(ctx.draw_hand),
        "gates": {"angle_min_confidence": min_conf, "min_visible_landmarks": min_visible},
        "smoothing": {"enabled": smoothing, "window": window, "polyorder": polyorder,
                      "columns": smoothed_cols},
        "nan_rate_by_measure": nan_rates,
        "dominant_gate_reason": dominant_reason,
        "implausible_values": implausible,
        "usable_frame_share": float(np.mean(frame_ok)),
        "pose_detection_rate": pose_quality.get("detection_rate"),
        "draw_hand_check": {"ok": bool(draw_hand_ok), "detail": draw_hand_check_detail},
    }
    out_quality = ctx.write_json("03_quality.json", quality)

    core = ["elbow_bow_deg", "elbow_draw_deg", "shoulder_bow_deg", "shoulder_draw_deg",
            "trunk_inclination_deg", "anchor_distance_norm"]
    worst = max((nan_rates.get(c, 1.0) for c in core), default=1.0)

    res.check("kinematics_written", out_parquet.is_file(), str(out_parquet))
    res.check("core_measures_available", worst <= 0.40,
              "Worst NaN rate among the core measures is "
              f"{worst:.1%}. Above 40% the shot cannot be measured reliably. "
              f"Per-measure rates: "
              + ", ".join(f"{c}={nan_rates.get(c, 1.0):.0%}" for c in core))
    res.check("no_systematic_implausible_values",
              all(v["share"] < 0.05 for v in implausible.values()),
              f"Angles outside anatomical bounds: {implausible}" if implausible
              else "All angles within anatomical plausibility bounds.")
    res.check("anchor_signal_present", bool(np.isfinite(out["anchor_distance_norm"]).any()),
              "Anchor-distance signal exists, so phase detection can run.")
    res.check("draw_hand_consistent_with_geometry", draw_hand_ok, draw_hand_check_detail,
              severity=WARN)
    res.check("shoulder_width_scale_stable",
              float(np.nanstd(shoulder_width) / (np.nanmedian(shoulder_width) + 1e-9)) < 0.25,
              "Shoulder width varies by "
              f"{float(np.nanstd(shoulder_width) / (np.nanmedian(shoulder_width) + 1e-9)):.1%} "
              "across the clip. Large variation means the archer moves toward or away from "
              "the camera, which weakens every normalised distance.", severity=WARN)

    res.outputs["kinematics"] = str(out_parquet)
    res.outputs["kinematics_quality"] = str(out_quality)
    res.stats = {
        "n_frames": n,
        "usable_frame_share": round(float(np.mean(frame_ok)), 4),
        "worst_core_nan_rate": round(worst, 4),
        "n_measures": len(gated),
    }
    return res
