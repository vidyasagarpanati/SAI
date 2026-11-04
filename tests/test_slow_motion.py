"""Slow-motion videos: the session's slow_motion_factor turns video time back
into real time in S1, and S4 widens its shot-detection windows on top."""
from __future__ import annotations

import json
import subprocess

import pytest

from archery.config import sha256_file
from archery.steps import s01_frames, s03_kinematics, s04_phases
from archery.steps.s00_ingest import _ffmpeg_exe
from archery.steps.s04_phases import slow_motion_rules
import synthetic
from helpers import make_run


def _clip(path, seconds=2, fps=30):
    subprocess.run([_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=duration={seconds}:size=160x120:rate={fps}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _s1(tmp_path, session):
    (tmp_path / "src").mkdir()
    video = _clip(tmp_path / "src" / "clip.mp4")
    df, _ = synthetic.build()
    ctx = make_run(tmp_path / "run", df, 30.0, session=session)
    ingest = json.loads((ctx.run_dir / "00_ingest.json").read_text())
    ingest.update({"video_path": str(video), "video_sha256": sha256_file(video),
                   "measured_fps": 30.0})
    ingest["probe"]["duration_s"] = 2.0
    (ctx.run_dir / "00_ingest.json").write_text(json.dumps(ingest))
    r1 = s01_frames.run(ctx)
    return r1, json.loads((ctx.run_dir / "01_frames.json").read_text())


def test_normal_video_is_unchanged(tmp_path):
    r1, fm = _s1(tmp_path, {})
    assert r1.passed, [f"{c.name}: {c.detail}" for c in r1.failures]
    assert fm["slow_motion_factor"] == 1.0
    assert fm["decode_fps"] == fm["analysis_fps"] == 30.0
    assert fm["frames_dir"].endswith("frames")          # the original folder


def test_slow_motion_times_are_real_time(tmp_path):
    """A 2 s clip at 4x is half a second of real shooting. Decoded at 15 fps of
    video, each frame is 1/60 s of real time."""
    r1, fm = _s1(tmp_path, {"slow_motion_factor": 4})
    assert r1.passed, [f"{c.name}: {c.detail}" for c in r1.failures]
    assert fm["decode_fps"] == 15.0 and fm["analysis_fps"] == 60.0
    assert abs(fm["n_frames"] - 30) <= 2
    assert abs(fm["n_frames"] / fm["analysis_fps"] - 0.5) < 0.05
    assert "slowmo_x4" in fm["frames_dir"]   # never reuses frames decoded at another rate


@pytest.mark.parametrize("raw", [None, "", 0.5, 0, -4, 100, "fast", "8x", [8]])
def test_unusable_factor_means_normal_video(raw):
    factor, why = s01_frames.slow_motion_factor({"slow_motion_factor": raw})
    assert factor == 1.0
    assert (why is None) == (raw in (None, ""))


def test_unusable_factor_warns_but_does_not_fail_s1(tmp_path):
    r1, fm = _s1(tmp_path, {"slow_motion_factor": "fast"})
    assert r1.passed, [f"{c.name}: {c.detail}" for c in r1.failures]
    assert fm["slow_motion_factor"] == 1.0
    assert any(c.name == "slow_motion_factor_usable" for c in r1.warnings)


def test_unusable_factor_passes_the_session_schema():
    """S0 fails on schema errors, so the schema must not reject what S1 forgives."""
    from helpers import ROOT
    from archery.steps.s00_ingest import validate_session
    for raw in ("fast", 0.5, 100, None):
        errors = validate_session({"slow_motion_factor": raw},
                                  ROOT / "schemas" / "session.schema.json")
        assert not [e for e in errors if "slow_motion_factor" in e]


def test_rules_unchanged_for_normal_video(tmp_path):
    df, truth = synthetic.build()
    ctx = make_run(tmp_path / "r", df, truth["fps"])
    rules = ctx.cfg.phase_rules
    assert slow_motion_rules(rules, 1.0) is rules


def test_slow_motion_widens_the_shot_windows(tmp_path):
    df, truth = synthetic.build()
    ctx = make_run(tmp_path / "r", df, truth["fps"])
    base = ctx.cfg.phase_rules
    wide = slow_motion_rules(base, 8.0)
    scale = base["slow_motion"]["window_scale"]
    for key in ("release_search_before_s", "release_search_after_s", "merge_anchor_gap_s"):
        assert wide["shot_detection"][key] == pytest.approx(base["shot_detection"][key] * scale)
    assert wide["shot_cycle"]["hook_max_lead_s"] == pytest.approx(
        base["shot_cycle"]["hook_max_lead_s"] * scale)
    assert base["shot_detection"]["merge_anchor_gap_s"] == 0.5     # original untouched


def test_slow_motion_shot_is_detected_in_real_time(tmp_path):
    """The synthetic archer stands in for an 8x clip decoded at 7.5 fps of video:
    60 fps of real time after S1. One shot, released at 3.5 s of REAL time."""
    df, truth = synthetic.build()
    ctx = make_run(tmp_path / "r", df, truth["fps"])
    fm = json.loads((ctx.run_dir / "01_frames.json").read_text())
    fm.update({"decode_fps": 7.5, "slow_motion_factor": 8.0})
    (ctx.run_dir / "01_frames.json").write_text(json.dumps(fm))
    s03_kinematics.run(ctx)
    r4 = s04_phases.run(ctx)
    assert r4.passed, [f"{c.name}: {c.detail}" for c in r4.failures]
    p = json.loads((ctx.run_dir / "04_phases.json").read_text())
    assert p["n_shots"] == 1 and p["slow_motion_factor"] == 8.0
    assert 3.45 <= p["shots"][0]["release_t_s"] <= 3.65
    assert p["thresholds_used"]["slow_motion_applied"]["factor"] == 8.0


class _Form(dict):
    def getlist(self, key):
        return [self[key]] if key in self else []


def test_form_default_is_normal_video():
    from archery.web.app import _session_from_form
    session, _ = _session_from_form(_Form(athlete_name="A", slow_motion_factor="1"))
    assert "slow_motion_factor" not in session
    session, _ = _session_from_form(_Form(athlete_name="A", slow_motion_factor="8"))
    assert session["slow_motion_factor"] == 8.0
    for raw in ("", "fast", "0", "99"):          # never an error, always normal
        session, _ = _session_from_form(_Form(slow_motion_factor=raw))
        assert "slow_motion_factor" not in session
    session, _ = _session_from_form(_Form())       # field missing altogether
    assert "slow_motion_factor" not in session


def test_form_offers_normal_video_first_and_selected():
    from helpers import ROOT
    html = (ROOT / "templates" / "web" / "new.html.j2").read_text()
    assert '<option value="1" selected>Normal video</option>' in html
