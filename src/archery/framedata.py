"""Loads per-frame landmark pixels and kinematics once, for S6, S7 and S10."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class FrameData:
    frames: list[Path]
    fps: float
    width: int
    height: int
    pts: np.ndarray          # (n, 33, 2) pixels
    vis: np.ndarray          # (n, 33)
    kin: pd.DataFrame
    phases: dict
    metrics: dict | None

    @classmethod
    def load(cls, run_dir: Path, with_metrics: bool = True) -> "FrameData":
        fm = json.loads((run_dir / "01_frames.json").read_text(encoding="utf-8"))
        frames = sorted(Path(fm["frames_dir"]).glob("f*.jpg"))
        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        lm = pd.read_parquet(run_dir / "02_landmarks.parquet")
        n = int(lm["frame"].max()) + 1
        xy = lm[["x", "y"]].to_numpy(float).reshape(n, 33, 2) * np.array([w, h])
        vis = lm["visibility"].to_numpy(float).reshape(n, 33)
        kin = pd.read_parquet(run_dir / "03_kinematics.parquet")
        phases = json.loads((run_dir / "04_phases.json").read_text(encoding="utf-8"))
        metrics = None
        mp = run_dir / "05_metrics.json"
        if with_metrics and mp.is_file():
            metrics = json.loads(mp.read_text(encoding="utf-8"))
        return cls(frames, float(fm["analysis_fps"]), w, h, xy, vis, kin, phases, metrics)

    def shoulder_width_px(self) -> float:
        from archery.landmarks import ID
        d = np.linalg.norm(self.pts[:, ID["LEFT_SHOULDER"]] - self.pts[:, ID["RIGHT_SHOULDER"]], axis=1)
        return float(np.nanmedian(d))

    def phase_of_frame(self) -> tuple[list[str | None], list[int | None]]:
        n = len(self.kin)
        codes: list[str | None] = [None] * n
        shots: list[int | None] = [None] * n
        for sh in self.phases["shots"]:
            for p in sh["phases"]:
                if p["detected"]:
                    for i in range(p["start_frame"], min(n, p["end_frame"] + 1)):
                        codes[i], shots[i] = p["phase"], sh["shot"]
        return codes, shots
