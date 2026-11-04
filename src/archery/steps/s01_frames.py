"""S1 frames.

IN     00_ingest.json
DO     one ffmpeg decode pass to JPEG at the configured analysis rate
OUT    frames/f######.jpg, 01_frames.json
VERIFY frame count within tolerance of expectation, no zero-byte frames, first
       and last frame decodable, source untouched (hash re-checked)

The video is decoded exactly once for the whole pipeline. S2, S6 and S7 all read
these JPEGs, so a rerun of any later step never re-decodes.

Slow motion. A clip with the slow motion baked in (a 30 fps file in which one
real second lasts slow_motion_factor seconds) is turned back into real time
here, once: analysis_fps is the REAL-time rate of the decoded frames,
decode_fps * slow_motion_factor. Every later step measures real seconds and
needs no change. decode_fps is the rate in the file's own time, which is what
S7 plays the annotated video back at, so the slow motion is kept for viewing.
A factor of 1 (the default) is a normal video and nothing changes.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from archery.config import sha256_file
from archery.context import Context
from archery.contracts import WARN, StepResult
from archery.io_guard import guarded_path
from archery.steps.s00_ingest import _ffmpeg_exe

FRAME_GLOB = "f*.jpg"
MAX_SLOW_MOTION = 32.0


def slow_motion_factor(session: dict) -> tuple[float, str | None]:
    """How many seconds of video one real second lasts, and why it was ignored.

    Optional, like every session field: absent, blank or unusable means a
    normal video (1), with a reason to warn about, never a failed run."""
    raw = session.get("slow_motion_factor")
    if raw in (None, ""):
        return 1.0, None
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return 1.0, f"slow_motion_factor {raw!r} is not a number; treated as a normal video"
    if not 1.0 <= f <= MAX_SLOW_MOTION:
        return 1.0, (f"slow_motion_factor {f:g} is outside 1 to {MAX_SLOW_MOTION:g}; "
                     f"treated as a normal video")
    return f, None


def _ffmpeg_quality(jpeg_quality: int) -> int:
    """Map a 0-100 quality to ffmpeg's mjpeg -q:v scale (2 best, 31 worst)."""
    jpeg_quality = max(1, min(100, int(jpeg_quality)))
    return max(2, min(31, round(2 + (100 - jpeg_quality) * (29 / 99))))


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S1")
    ingest = ctx.read_json("00_ingest.json")
    video = Path(ingest["video_path"])

    native_fps = float(ingest["measured_fps"] or 0)
    target_fps = float(ctx.cfg.get("frames.analysis_fps", 60))
    factor, ignored = slow_motion_factor(ctx.session)
    res.check("slow_motion_factor_usable", ignored is None,
              ignored or (f"Slow motion x{factor:g}." if factor > 1 else "Normal video."),
              severity=WARN)
    if factor > 1:
        # Real-time rate target_fps needs only target_fps / factor frames per
        # second of video, so a long slow-motion clip stays under max_frames.
        # The floor keeps the release snap sampled densely in real time.
        floor = float(ctx.cfg.get("frames.slow_motion_min_decode_fps", 15))
        decode_fps = max(target_fps / factor, floor)
        decode_fps = min(native_fps, decode_fps) if native_fps else decode_fps
    else:
        decode_fps = min(native_fps, target_fps) if native_fps else target_fps
    analysis_fps = decode_fps * factor
    duration = ingest["probe"].get("duration_s")
    max_frames = int(ctx.cfg.get("frames.max_frames", 20000))

    # Frames decoded at another rate must never be reused, so a slow-motion
    # decode gets its own folder. Normal videos keep the original path.
    frames_dir = guarded_path(ctx.frames_dir if factor == 1
                              else ctx.frames_dir / f"slowmo_x{factor:g}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frames_dir.glob(FRAME_GLOB))
    if not existing:
        cmd = [
            _ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vf", f"fps={decode_fps}",
            "-q:v", str(_ffmpeg_quality(ctx.cfg.get("frames.jpeg_quality", 92))),
            "-frames:v", str(max_frames),
            "-start_number", "0",
            str(frames_dir / "f%06d.jpg"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        res.check("ffmpeg_decode_ok", proc.returncode == 0,
                  (proc.stderr or "").strip()[:2000] or "ffmpeg decode completed.")
        if proc.returncode != 0:
            return res
        existing = sorted(frames_dir.glob(FRAME_GLOB))
    else:
        res.check("ffmpeg_decode_ok", True,
                  f"Reused {len(existing)} frames already decoded for this run.")

    n = len(existing)
    res.check("frames_produced", n > 0,
              f"{n} frames at {decode_fps:g} fps" + (
                  f" of video; slow motion x{factor:g}, so {analysis_fps:g} fps of real time"
                  if factor > 1 else ""))
    if n == 0:
        return res

    if duration:
        expected = min(int(round(duration * decode_fps)), max_frames)
        tolerance = max(2, int(0.02 * expected))
        res.check("frame_count_matches_duration", abs(n - expected) <= tolerance,
                  f"Got {n}, expected about {expected} (tolerance {tolerance}).",
                  severity=WARN, evidence={"got": n, "expected": expected})

    empty = [p.name for p in existing if p.stat().st_size == 0]
    res.check("no_zero_byte_frames", not empty, f"Zero-byte frames: {empty[:10]}")

    import cv2
    ok_first = cv2.imread(str(existing[0])) is not None
    ok_last = cv2.imread(str(existing[-1])) is not None
    res.check("endpoints_decodable", ok_first and ok_last,
              f"first={ok_first} last={ok_last}")

    res.check("source_unmodified", sha256_file(video) == ingest["video_sha256"],
              "Source video hash unchanged after decoding.")

    res.check("frame_cap_not_hit", n < max_frames,
              f"Hit the {max_frames}-frame cap. The video is longer than the "
              f"configured analysis window and is truncated.",
              severity=WARN)

    import cv2 as _cv2
    _first = _cv2.imread(str(existing[0]))
    if _first is None:
        raise RuntimeError(f"Could not read the first extracted frame: {existing[0]}")
    _h, _w = _first.shape[:2]

    payload = {
        "frames_dir": str(frames_dir),
        "n_frames": n,
        # The decoded frame size. Landmarks are normalised against THIS, so any
        # image-plane angle has to be un-normalised with it or the aspect ratio
        # skews every angle (1.78x on 16:9).
        "frame_width": int(_w),
        "frame_height": int(_h),
        "analysis_fps": analysis_fps,
        "decode_fps": decode_fps,
        "slow_motion_factor": factor,
        "native_fps": native_fps,
        "frame_pattern": "f%06d.jpg",
        "first_frame": existing[0].name,
        "last_frame": existing[-1].name,
        "timestamps_s": None,
        "note": ("Frame i is at t = i / analysis_fps seconds of REAL time, and at "
                 "i / decode_fps seconds in the video file. They differ only for slow motion."),
    }
    out = ctx.write_json("01_frames.json", payload)
    res.outputs["frames_index"] = str(out)
    res.outputs["frames_dir"] = str(frames_dir)
    res.stats = {"n_frames": n, "analysis_fps": analysis_fps, "decode_fps": decode_fps,
                 "slow_motion_factor": factor}
    return res
