"""S3 is exercised against a synthetic archer whose true geometry is known."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archery import io_guard
from archery.config import load_config
from archery.context import Context
from archery.steps import s03_kinematics
import synthetic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ctx(tmp_path_factory) -> Context:
    run_dir = tmp_path_factory.mktemp("run")
    io_guard.configure([run_dir])
    cfg = load_config(ROOT)
    df, truth = synthetic.build()
    df.to_parquet(run_dir / "02_landmarks.parquet", index=False)
    (run_dir / "01_frames.json").write_text(json.dumps({
        "frames_dir": str(run_dir / "frames"), "n_frames": truth["n_frames"],
        "analysis_fps": truth["fps"], "native_fps": truth["fps"],
    }))
    (run_dir / "02_pose_quality.json").write_text(json.dumps({"detection_rate": 1.0}))
    c = Context(cfg=cfg, run_id="synthetic", run_dir=run_dir,
                video_path=Path("synthetic.mov"),
                session={"draw_hand": "right", "athlete_name": "Synthetic"})
    c.truth = truth
    return c


@pytest.fixture(scope="module")
def result(ctx):
    return s03_kinematics.run(ctx)


def test_step_passes_its_own_checks(result):
    assert result.passed, [f"{c.name}: {c.detail}" for c in result.failures]


def test_bow_arm_is_the_extended_one_at_anchor(ctx, result):
    df = pd.read_parquet(ctx.run_dir / "03_kinematics.parquet")
    k = int(df["anchor_distance_norm"].idxmin())
    assert df.loc[k, "elbow_bow_deg"] > df.loc[k, "elbow_draw_deg"], (
        "For a right-draw archer the LEFT arm holds the bow and should be the "
        "more extended arm at anchor."
    )


def test_anchor_distance_is_minimal_during_the_held_phase(ctx, result):
    df = pd.read_parquet(ctx.run_dir / "03_kinematics.parquet")
    held = df[(df["t_s"] >= 2.5) & (df["t_s"] <= 3.1)]["anchor_distance_norm"]
    early = df[(df["t_s"] >= 0.2) & (df["t_s"] <= 0.9)]["anchor_distance_norm"]
    assert held.mean() < early.mean() * 0.6


def test_release_produces_the_fastest_draw_wrist(ctx, result):
    df = pd.read_parquet(ctx.run_dir / "03_kinematics.parquet")
    peak_t = float(df.loc[df["draw_wrist_speed_norm_s"].idxmax(), "t_s"])
    assert 3.4 <= peak_t <= 3.75, f"Release speed peak landed at {peak_t:.2f}s"


def test_every_angle_is_gated_not_guessed(ctx, result):
    quality = json.loads((ctx.run_dir / "03_quality.json").read_text())
    assert "nan_rate_by_measure" in quality
    assert quality["draw_hand_check"]["ok"] is True
    assert not quality["implausible_values"], quality["implausible_values"]


def test_low_confidence_landmarks_do_not_silently_produce_values(ctx):
    """Drop the far-side ear below the gate and the head-tilt measure must vanish."""
    import copy
    run_dir = ctx.run_dir.parent / "run_lowconf"
    run_dir.mkdir(exist_ok=True)
    io_guard.configure([ctx.run_dir, run_dir])
    df, truth = synthetic.build()
    from archery.landmarks import ID
    df.loc[df["lm"] == ID["LEFT_EAR"], "visibility"] = 0.15
    df.to_parquet(run_dir / "02_landmarks.parquet", index=False)
    for name in ("01_frames.json", "02_pose_quality.json"):
        (run_dir / name).write_text((ctx.run_dir / name).read_text())
    c2 = copy.replace(ctx, run_dir=run_dir) if hasattr(copy, "replace") else Context(
        cfg=ctx.cfg, run_id="lowconf", run_dir=run_dir, video_path=ctx.video_path,
        session=ctx.session)
    s03_kinematics.run(c2)
    out = pd.read_parquet(run_dir / "03_kinematics.parquet")
    assert out["head_tilt_deg"].isna().all(), (
        "head_tilt was computed from a landmark below the confidence gate"
    )


def test_determinism(ctx):
    """Same input, same config, byte-identical output."""
    import hashlib
    first = hashlib.sha256((ctx.run_dir / "03_kinematics.parquet").read_bytes()).hexdigest()
    s03_kinematics.run(ctx)
    second = hashlib.sha256((ctx.run_dir / "03_kinematics.parquet").read_bytes()).hexdigest()
    assert first == second
