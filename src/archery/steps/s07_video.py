"""S7 annotated video.

IN     frames/, 02_landmarks.parquet, 03_kinematics.parquet, 04_phases.json,
       00_ingest.json
DO     render every analysed frame through the same overlay as S6 (video mode:
       skeleton, landmarks, reference lines, live joint angles, phase badge,
       live angle table), pipe to ffmpeg, then cut one clip per detected phase
OUT    07_video/annotated_full.mp4, 07_video/shot<k>_p<badge>_<PHASE>.mp4,
       outputs/<Athlete>_Annotated_<date>_vNN.mp4 (versioned, never overwritten)
VERIFY frame count and duration match the analysed frames, one clip per
       detected phase, source video untouched

Angles in the video are live per-frame values from 03_kinematics.parquet,
rounded to one decimal. Gated values show as n/a, never interpolated.
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from archery.config import sha256_file
from archery.context import Context
from archery.contracts import WARN, StepResult
from archery.framedata import FrameData
from archery.io_guard import guarded_path
from archery.overlay import render
from archery.steps.s00_ingest import _ffmpeg_exe
from archery.versioning import next_versioned, safe

LIVE = [("Bow elbow", "elbow_bow_deg", 1, "deg"), ("Draw elbow", "elbow_draw_deg", 1, "deg"),
        ("Bow shoulder", "shoulder_bow_deg", 1, "deg"), ("Draw shoulder", "shoulder_draw_deg", 1, "deg"),
        ("Trunk incl.", "trunk_inclination_deg", 1, "deg"), ("Shoulder tilt", "shoulder_tilt_deg", 1, "deg"),
        ("Anchor dist.", "anchor_distance_norm", 3, "SW"), ("Knee L", "knee_left_deg", 1, "deg"),
        ("Knee R", "knee_right_deg", 1, "deg")]
ANGLE_COLS = None


def _values(row) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, (float, np.floating)) and np.isfinite(v):
            out[k] = round(float(v), 3 if k.endswith("_norm") or k.endswith("_s") else 1)
    return out


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S7")
    ingest = ctx.read_json("00_ingest.json")
    fd = FrameData.load(ctx.run_dir, with_metrics=False)
    codes, shots = fd.phase_of_frame()
    track = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))
    crf = str(ctx.cfg.get("video.output_crf", 20))
    preset = str(ctx.cfg.get("video.output_preset", "medium"))
    ffmpeg = _ffmpeg_exe()

    vdir = guarded_path(ctx.video_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    full = vdir / "annotated_full.mp4"
    n = len(fd.frames)

    if ctx.cfg.get("video.render_full_annotated", True):
        first, _ = render(cv2.imread(str(fd.frames[0])), fd.pts[0], fd.vis[0], {},
                          draw_hand=ctx.draw_hand, shot=None, phase=None, t_s=0.0, frame_idx=0,
                          mode="video", track_conf=track)
        H, W = first.shape[:2]
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", repr(fd.fps), "-i", "-",
               "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(full)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        report_every = max(1, n // 10)
        try:
            for i, fpath in enumerate(fd.frames):
                row = fd.kin.iloc[i].to_dict()
                vals = _values(row)
                table = [(lab, "n/a" if vals.get(k) is None else f"{vals[k]:.{d}f} {u}")
                         for lab, k, d, u in LIVE]
                canvas, _ = render(cv2.imread(str(fpath)), fd.pts[i], fd.vis[i], vals,
                                   draw_hand=ctx.draw_hand, shot=shots[i], phase=codes[i],
                                   t_s=float(row["t_s"]), frame_idx=i, mode="video",
                                   track_conf=track, measurements=table)
                if canvas.shape[:2] != (H, W):
                    canvas = cv2.resize(canvas, (W, H))
                proc.stdin.write(canvas.tobytes())
                if (i + 1) % report_every == 0 or i + 1 == n:
                    print(f"  video {i + 1:>6}/{n}  {100 * (i + 1) / n:5.1f}%", flush=True)
        finally:
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            code = proc.wait()
        res.check("ffmpeg_encode_ok", code == 0, err.strip()[:1500] or "Encoded annotated_full.mp4")
        if code != 0:
            return res

    got = _frame_count(full) if full.is_file() else 0
    res.check("full_video_frame_count", abs(got - n) <= 1, f"{got} frames encoded for {n} analysed frames")

    clips, clip_fail = [], []
    if ctx.cfg.get("video.render_phase_clips", True):
        for sh in fd.phases["shots"]:
            for p in sh["phases"]:
                if not p["detected"]:
                    continue
                name = f"shot{sh['shot']}_p{p['badge']}_{p['phase']}.mp4"
                dst = vdir / name
                start = p["start_frame"] / fd.fps
                dur = max((p["end_frame"] - p["start_frame"] + 1) / fd.fps, 1.0 / fd.fps)
                cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(full),
                       "-ss", f"{start:.4f}", "-t", f"{dur:.4f}", "-c:v", "libx264",
                       "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p", "-an", str(dst)]
                ok = subprocess.run(cmd, capture_output=True).returncode == 0 and dst.is_file()
                (clips if ok else clip_fail).append(name)

    n_detected = sum(p["detected"] for sh in fd.phases["shots"] for p in sh["phases"])
    res.check("phase_clips_rendered", not clip_fail and len(clips) == n_detected,
              f"{len(clips)} of {n_detected} phase clips written" +
              (f"; failed: {clip_fail}" if clip_fail else ""), severity=WARN)

    session = ingest.get("session", {})
    date = session.get("session_date")
    date = date if isinstance(date, str) and len(date) == 10 else dt.date.today().isoformat()
    stem = f"{safe(session.get('athlete_name', 'Athlete'))}_Annotated_{date.replace('-', '')}"
    published = next_versioned(guarded_path(ctx.cfg.paths.outputs_dir), stem, ".mp4")
    shutil.copyfile(full, guarded_path(published))
    res.check("versioned_copy_published", published.is_file(), f"Published {published.name}")

    res.check("source_unmodified", sha256_file(ingest["video_path"]) == ingest["video_sha256"]
              if Path(ingest["video_path"]).is_file() else True,
              "Source video hash unchanged after rendering.")

    res.outputs["annotated_full"] = str(full)
    res.outputs["published_video"] = str(published)
    res.outputs["clips_dir"] = str(vdir)
    ctx.write_json("07_video.json", {"full": str(full), "published": str(published),
                                     "clips": clips, "fps": fd.fps, "frames": n,
                                     "note": "Live angles are per-frame values from 03_kinematics, "
                                             "rounded; gated values render as n/a."})
    res.stats = {"frames": n, "clips": len(clips), "published": published.name}
    return res
