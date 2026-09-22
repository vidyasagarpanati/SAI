"""Build a run directory from synthetic landmarks and push it through S3..S5."""
from __future__ import annotations

import json
from pathlib import Path

from archery import io_guard
from archery.config import load_config
from archery.context import Context

ROOT = Path(__file__).resolve().parents[1]


def make_run(run_dir: Path, df, fps: float, session: dict | None = None) -> Context:
    run_dir.mkdir(parents=True, exist_ok=True)
    io_guard.configure([run_dir])
    df.to_parquet(run_dir / "02_landmarks.parquet", index=False)
    n = int(df["frame"].max()) + 1
    (run_dir / "01_frames.json").write_text(json.dumps(
        {"frames_dir": str(run_dir / "frames"), "n_frames": n, "analysis_fps": fps}))
    (run_dir / "02_pose_quality.json").write_text(json.dumps(
        {"detection_rate": 1.0, "per_frame_visible_count": [31] * n}))
    sess = {"draw_hand": "right", "athlete_name": "Synthetic", "bow_type": "Recurve",
            "camera_view": "side-on", **(session or {})}
    (run_dir / "00_ingest.json").write_text(json.dumps({
        "video_path": "C:/videos/synthetic.mov", "video_sha256": "0" * 64,
        "measured_fps": fps, "session": sess,
        "probe": {"duration_s": n / fps, "width": 1920, "height": 1080,
                  "codec": "h264", "source": "synthetic"}}))
    return Context(cfg=load_config(ROOT), run_id=run_dir.name, run_dir=run_dir,
                   video_path=Path("synthetic.mov"), session=sess)
