"""S2 pose.

IN     frames/, 01_frames.json
DO     MediaPipe Pose Landmarker over every frame in VIDEO running mode
OUT    02_landmarks.parquet, 02_pose_quality.json
VERIFY one row block per frame, detection rate recorded, frames below the
       visible-landmark floor flagged, confidence distribution reported

This is the most expensive step in the pipeline, so it is aggressively cached.
It runs on CPU by design: the GPU stays free for Ollama.

Both coordinate spaces are stored:
  x, y, z       normalised image space, origin top-left, used for overlays and
                for any angle measured against the image frame (tilts, plumb line)
  wx, wy, wz    MediaPipe world landmarks, metres, origin at the hip midpoint,
                used for true joint angles because they are not distorted by
                perspective
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from archery.context import Context
from archery.contracts import WARN, StepResult
from archery.io_guard import guarded_path
from archery.landmarks import NAMES

N_LANDMARKS = 33
COLUMNS = ["frame", "t_ms", "lm", "x", "y", "z", "visibility", "presence",
           "wx", "wy", "wz"]


MEDIAPIPE_DLL_HELP = """
MediaPipe's native bindings failed to load.

On Windows this is almost always one of two things:

 1. Clashing OpenCV installs. opencv-python, opencv-python-headless and
    opencv-contrib-python all install into the same site-packages\\cv2
    directory. MediaPipe is built against opencv-contrib-python, so any other
    one installed on top leaves mismatched native DLLs behind.

        .\\scripts\\repair_opencv.ps1

 2. Missing Visual C++ runtime.

        winget install Microsoft.VCRedist.2015+.x64

Then confirm with:

        archery doctor

Original error: {error}
"""


def _import_mediapipe():
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        return mp_python, vision
    except ImportError as exc:
        raise ImportError(MEDIAPIPE_DLL_HELP.format(error=exc)) from exc


def _build_landmarker(ctx: Context):
    mp_python, vision = _import_mediapipe()

    model_path = ctx.cfg.paths.models_dir / ctx.cfg.get("paths.pose_model_file")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe pose model missing: {model_path}\n"
            f"Rerun scripts/bootstrap.ps1, or download pose_landmarker_full.task by hand."
        )
    delegate_name = str(ctx.cfg.get("pose.delegate", "CPU")).upper()
    delegate = getattr(mp_python.BaseOptions.Delegate, delegate_name)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path), delegate=delegate),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=int(ctx.cfg.get("pose.num_poses", 1)),
        min_pose_detection_confidence=float(ctx.cfg.get("pose.min_pose_detection_confidence", 0.5)),
        min_pose_presence_confidence=float(ctx.cfg.get("pose.min_pose_presence_confidence", 0.5)),
        min_tracking_confidence=float(ctx.cfg.get("pose.min_pose_tracking_confidence", 0.5)),
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options), model_path


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S2")
    _import_mediapipe()          # fail early with the remediation message
    import cv2
    import mediapipe as mp

    frames_meta = ctx.read_json("01_frames.json")
    frames_dir = Path(frames_meta["frames_dir"])
    analysis_fps = float(frames_meta["analysis_fps"])
    frame_paths = sorted(frames_dir.glob("f*.jpg"))

    res.check("frames_available", bool(frame_paths), f"{len(frame_paths)} frames in {frames_dir}")
    if not frame_paths:
        return res

    landmarker, model_path = _build_landmarker(ctx)

    track_conf = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))
    low_conf = float(ctx.cfg.get("quality_gates.landmark_low_confidence_flag", 0.3))
    min_visible = int(ctx.cfg.get("quality_gates.min_visible_landmarks", 20))

    n = len(frame_paths)
    arr = np.full((n, N_LANDMARKS, 8), np.nan, dtype=np.float32)
    detected = np.zeros(n, dtype=bool)
    unreadable: list[str] = []

    with landmarker:
        for i, path in enumerate(frame_paths):
            image_bgr = cv2.imread(str(path))
            if image_bgr is None:
                unreadable.append(path.name)
                continue
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(i * 1000.0 / analysis_fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if not result.pose_landmarks:
                continue
            detected[i] = True
            lms = result.pose_landmarks[0]
            worlds = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            for j in range(N_LANDMARKS):
                lm = lms[j]
                arr[i, j, 0] = lm.x
                arr[i, j, 1] = lm.y
                arr[i, j, 2] = lm.z
                arr[i, j, 3] = getattr(lm, "visibility", np.nan)
                arr[i, j, 4] = getattr(lm, "presence", np.nan)
                if worlds is not None:
                    w = worlds[j]
                    arr[i, j, 5] = w.x
                    arr[i, j, 6] = w.y
                    arr[i, j, 7] = w.z

    frame_idx = np.repeat(np.arange(n, dtype=np.int32), N_LANDMARKS)
    lm_idx = np.tile(np.arange(N_LANDMARKS, dtype=np.int16), n)
    flat = arr.reshape(n * N_LANDMARKS, 8)
    df = pd.DataFrame({
        "frame": frame_idx,
        "t_ms": (frame_idx * (1000.0 / analysis_fps)).astype(np.float32),
        "lm": lm_idx,
        "x": flat[:, 0], "y": flat[:, 1], "z": flat[:, 2],
        "visibility": flat[:, 3], "presence": flat[:, 4],
        "wx": flat[:, 5], "wy": flat[:, 6], "wz": flat[:, 7],
    })

    out_parquet = guarded_path(ctx.artefact("02_landmarks.parquet"))
    df.to_parquet(out_parquet, index=False)

    vis = arr[:, :, 3]
    n_visible = np.nansum(vis >= track_conf, axis=1)
    n_low = np.nansum((vis < low_conf) & ~np.isnan(vis), axis=1)
    insufficient = np.where(n_visible < min_visible)[0]

    detection_rate = float(detected.mean())
    usable_rate = float((n_visible >= min_visible).mean())

    per_landmark = {
        NAMES[j]: {
            "mean_visibility": float(np.nanmean(vis[:, j])) if np.isfinite(vis[:, j]).any() else None,
            "frames_above_track_threshold": int(np.nansum(vis[:, j] >= track_conf)),
            "frames_low_confidence": int(np.nansum(vis[:, j] < low_conf)),
        }
        for j in range(N_LANDMARKS)
    }

    quality = {
        "model": str(model_path.name),
        "delegate": ctx.cfg.get("pose.delegate", "CPU"),
        "n_frames": n,
        "analysis_fps": analysis_fps,
        "detection_rate": detection_rate,
        "usable_frame_rate": usable_rate,
        "min_visible_landmarks_required": min_visible,
        "frames_insufficient_landmarks": insufficient.tolist()[:500],
        "n_frames_insufficient_landmarks": int(insufficient.size),
        "unreadable_frames": unreadable,
        "per_frame_visible_count": n_visible.astype(int).tolist(),
        "per_frame_low_confidence_count": n_low.astype(int).tolist(),
        "per_landmark": per_landmark,
        "thresholds": {
            "track_confidence": track_conf,
            "low_confidence_flag": low_conf,
            "min_visible_landmarks": min_visible,
        },
    }
    out_quality = ctx.write_json("02_pose_quality.json", quality)

    res.check("rows_per_frame_correct", len(df) == n * N_LANDMARKS,
              f"{len(df)} rows for {n} frames x {N_LANDMARKS} landmarks")
    res.check("no_unreadable_frames", not unreadable, f"Unreadable: {unreadable[:10]}")
    res.check("pose_detected_in_most_frames", detection_rate >= 0.80,
              f"Archer detected in {detection_rate:.1%} of frames. "
              f"Below 80% the video is not usable for consistent measurement.")
    res.check("enough_usable_frames", usable_rate >= 0.60,
              f"{usable_rate:.1%} of frames have at least {min_visible} of 33 landmarks visible.")
    res.check("world_landmarks_present", bool(np.isfinite(arr[:, :, 5]).any()),
              "World landmarks available, joint angles will use metric space.",
              severity=WARN)
    res.check("low_confidence_share_acceptable",
              float(np.nanmean(n_low)) <= 8,
              f"Mean {float(np.nanmean(n_low)):.1f} landmarks per frame below the "
              f"{low_conf} confidence flag.", severity=WARN)

    res.outputs["landmarks"] = str(out_parquet)
    res.outputs["pose_quality"] = str(out_quality)
    res.stats = {
        "n_frames": n,
        "detection_rate": round(detection_rate, 4),
        "usable_frame_rate": round(usable_rate, 4),
        "mean_visible_landmarks": round(float(np.mean(n_visible)), 2),
    }
    return res
