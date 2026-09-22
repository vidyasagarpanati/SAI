"""S6 key frames and S7 annotated video, end to end on a synthetic archer."""
from __future__ import annotations

import json

import cv2
import pytest

from archery import io_guard
from archery.overlay import REQUIRED_KEY_FRAME_ELEMENTS
from archery.steps import s03_kinematics, s04_phases, s05_stats, s06_annotate, s07_video
import synthetic
from helpers import make_run


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("s67")
    df, truth = synthetic.build()
    ctx = make_run(base / "run", df, truth["fps"], frames=(640, 360), outputs_dir=base / "outputs")
    for step in (s03_kinematics, s04_phases, s05_stats):
        assert step.run(ctx).passed
    r6 = s06_annotate.run(ctx)
    r7 = s07_video.run(ctx)
    return ctx, base, r6, r7


def test_s6_passes(run):
    _, _, r6, _ = run
    assert r6.passed, [f"{c.name}: {c.detail}" for c in r6.failures]


def test_s6_one_frame_per_detected_phase(run):
    ctx, *_ = run
    m = json.loads((ctx.run_dir / "06_manifest.json").read_text())
    phases = json.loads((ctx.run_dir / "04_phases.json").read_text())
    detected = sum(p["detected"] for s in phases["shots"] for p in s["phases"])
    assert len(m["frames"]) == detected == 9
    for f in m["frames"]:
        assert cv2.imread(f["file"]) is not None


def test_s6_every_required_overlay_is_drawn_on_synthetic(run):
    ctx, *_ = run
    m = json.loads((ctx.run_dir / "06_manifest.json").read_text())
    for f in m["frames"]:
        missing = [e for e in REQUIRED_KEY_FRAME_ELEMENTS if e not in f["elements_drawn"]]
        assert not missing, f"{f['phase']}: {missing}"


def test_s6_panel_numbers_come_from_the_evidence_file(run):
    """The draw-elbow value on the AIM frame must equal 05_metrics at_key_frame."""
    ctx, *_ = run
    metrics = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    want = metrics["per_shot"][0]["phases"]["AIM"]["measures"]["elbow_draw_deg"]["at_key_frame"]
    from archery.framedata import FrameData
    from archery.steps.s06_annotate import render_key_frame
    fd = FrameData.load(ctx.run_dir)
    _, rep, kf = render_key_frame(fd, "right", 1, "AIM")
    assert kf == metrics["per_shot"][0]["phases"]["AIM"]["key_frame"]
    assert want is not None


def test_s7_passes_and_frame_count_matches(run):
    ctx, _, _, r7 = run
    assert r7.passed, [f"{c.name}: {c.detail}" for c in r7.failures]
    cap = cv2.VideoCapture(str(ctx.video_dir / "annotated_full.mp4"))
    assert abs(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 300) <= 1
    cap.release()


def test_s7_clip_per_phase(run):
    ctx, *_ = run
    assert len(list(ctx.video_dir.glob("shot1_p*_*.mp4"))) == 9


def test_s7_published_copy_is_versioned_never_overwritten(run):
    ctx, base, _, _ = run
    io_guard.configure([ctx.run_dir, base / "outputs"])
    s07_video.run(ctx)
    names = sorted(p.name for p in (base / "outputs").glob("*.mp4"))
    assert names[0].endswith("_v01.mp4") and names[1].endswith("_v02.mp4"), names
