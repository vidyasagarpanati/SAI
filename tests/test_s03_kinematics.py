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
        "frame_width": 1000, "frame_height": 1000,
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


def test_held_aim_is_not_reported_as_movement(ctx, result):
    """Speeds come from smoothed positions, so jitter during a held aim stays small."""
    df = pd.read_parquet(ctx.run_dir / "03_kinematics.parquet")
    held = df[(df["t_s"] >= 2.6) & (df["t_s"] <= 3.1)]["com_speed_norm_s"]
    assert held.median() < 0.3, f"held-aim COM speed {held.median():.3f} SW/s is jitter, not motion"


# --------------------------------------------------------------- angle spaces
# A real frame showed the draw elbow at 109.9 deg where the picture plainly
# showed roughly 20. The joint angles were computed in MediaPipe's world
# landmarks, whose depth is inferred from one view, and at full draw the draw
# upper arm points partly along the camera axis. These tests pin the fix.

def _run_with_frame(tmp_path, width, height, df):
    import pandas as pd
    from archery.config import load_config
    io_guard.configure([tmp_path])
    df.to_parquet(tmp_path / "02_landmarks.parquet", index=False)
    n = int(df["frame"].max()) + 1
    (tmp_path / "01_frames.json").write_text(json.dumps({
        "frames_dir": str(tmp_path / "frames"), "n_frames": n, "analysis_fps": 60.0,
        "frame_width": width, "frame_height": height}))
    (tmp_path / "02_pose_quality.json").write_text(json.dumps({"detection_rate": 1.0}))
    c = Context(cfg=load_config(ROOT), run_id="aspect", run_dir=tmp_path,
                video_path=Path("synthetic.mov"), session={"draw_hand": "right"})
    s03_kinematics.run(c)
    return pd.read_parquet(tmp_path / "03_kinematics.parquet")


def test_a_right_angle_in_pixels_is_not_a_right_angle_in_normalised_coordinates():
    """Why the image-plane work is done in pixels. MediaPipe normalises each
    axis independently, so on 16:9 the x axis is stretched 1.78x against y."""
    from archery.geometry import angle_at
    w, h = 1920.0, 1080.0
    vertex, along_x, along_y = (960.0, 540.0), (1460.0, 540.0), (960.0, 40.0)
    px = [np.array([p]) for p in (vertex, along_x, along_y)]
    assert angle_at(*px)[0] == pytest.approx(90.0, abs=0.01)
    norm = [np.array([[p[0] / w, p[1] / h]]) for p in (vertex, along_x, along_y)]
    assert angle_at(*norm)[0] == pytest.approx(90.0, abs=0.01)   # axis-aligned survives
    # An off-axis angle does not.
    oblique = (1460.0, 40.0)
    true_deg = angle_at(np.array([vertex]), np.array([along_x]), np.array([oblique]))[0]
    skewed = angle_at(np.array([[vertex[0] / w, vertex[1] / h]]),
                      np.array([[along_x[0] / w, along_x[1] / h]]),
                      np.array([[oblique[0] / w, oblique[1] / h]]))[0]
    assert abs(true_deg - skewed) > 8.0, (true_deg, skewed)


def test_every_joint_is_measured_in_both_spaces_with_the_gap_recorded(ctx, result):
    import pandas as pd
    k = pd.read_parquet(ctx.run_dir / "03_kinematics.parquet")
    joints = [c for c in k.columns
              if c.endswith("_deg") and f"{c}_3d" in k.columns]
    assert len(joints) >= 14, joints
    for j in joints:
        assert f"{j}_2d3d_diff" in k.columns
        gap = k[f"{j}_2d3d_diff"].to_numpy(dtype=float)
        finite = gap[np.isfinite(gap)]
        assert finite.size and finite.min() >= 0.0        # a magnitude, never negative
        assert finite.max() <= 180.0


def test_a_planar_archer_reads_the_same_in_both_spaces(ctx, result):
    """The synthetic archer has no real depth, so the two spaces must agree.
    Any disagreement here would mean the fix introduced its own error."""
    import pandas as pd
    k = pd.read_parquet(ctx.run_dir / "03_kinematics.parquet")
    for j in ("elbow_bow_deg", "elbow_draw_deg", "knee_left_deg"):
        gap = k[f"{j}_2d3d_diff"].to_numpy(dtype=float)
        assert np.nanmedian(gap) < 1.0, (j, np.nanmedian(gap))


def test_the_draw_elbow_is_closed_at_full_draw(ctx, result):
    """The reported defect: a folded draw arm read 109.9 deg. At anchor the
    upper arm and the forearm both point forward from the elbow, so the
    included angle is small."""
    import pandas as pd
    k = pd.read_parquet(ctx.run_dir / "03_kinematics.parquet")
    at_anchor = k.iloc[int(2.6 * 60):int(3.1 * 60)]["elbow_draw_deg"]
    assert at_anchor.median() < 60.0, at_anchor.median()
    assert k.iloc[int(2.6 * 60):int(3.1 * 60)]["elbow_bow_deg"].median() > 150.0


def test_the_declared_frame_size_actually_reaches_the_angles(ctx, tmp_path):
    import pandas as pd
    df = pd.read_parquet(ctx.run_dir / "02_landmarks.parquet")
    square = _run_with_frame(tmp_path / "sq", 1000, 1000, df)
    wide = _run_with_frame(tmp_path / "wide", 1920, 1080, df)
    i = int(2.8 * 60)
    assert square.at[i, "elbow_draw_deg"] != pytest.approx(wide.at[i, "elbow_draw_deg"], abs=0.5)
    # The 3D value is independent of how the frame is cropped, so it must not move.
    assert square.at[i, "elbow_draw_deg_3d"] == pytest.approx(
        wide.at[i, "elbow_draw_deg_3d"], abs=0.01)


def test_missing_frame_size_refuses_rather_than_defaulting(ctx, tmp_path):
    import pandas as pd
    from archery.config import load_config
    run = tmp_path / "nosize"
    run.mkdir(parents=True)
    io_guard.configure([run])
    pd.read_parquet(ctx.run_dir / "02_landmarks.parquet").to_parquet(
        run / "02_landmarks.parquet", index=False)
    (run / "01_frames.json").write_text(json.dumps(
        {"frames_dir": str(run), "n_frames": 300, "analysis_fps": 60.0}))
    (run / "02_pose_quality.json").write_text(json.dumps({"detection_rate": 1.0}))
    c = Context(cfg=load_config(ROOT), run_id="nosize", run_dir=run,
                video_path=Path("x.mov"), session={"draw_hand": "right"})
    with pytest.raises(RuntimeError, match="frame_width"):
        s03_kinematics.run(c)


def test_both_shoulder_definitions_are_reported_and_differ(ctx, result):
    import pandas as pd
    k = pd.read_parquet(ctx.run_dir / "03_kinematics.parquet")
    i = int(2.8 * 60)
    for side in ("bow", "draw"):
        assert f"shoulder_{side}_girdle_deg" in k.columns
        assert k.at[i, f"shoulder_{side}_girdle_deg"] != pytest.approx(
            k.at[i, f"shoulder_{side}_deg"], abs=1.0)
