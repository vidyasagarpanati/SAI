"""Every session form field is optional. Leaving the form blank, or filling a
field with something unusable, must warn and carry on, never fail the run."""
from __future__ import annotations

import subprocess

import pytest

from archery import io_guard
from archery.config import load_config
from archery.context import Context
from archery.steps import s00_ingest
from archery.steps.s00_ingest import NOT_PROVIDED, _ffmpeg_exe
from helpers import ROOT


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("src") / "clip.mp4"
    subprocess.run([_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=160x120:rate=30", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


def _s0(tmp_path, clip, session):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    io_guard.configure([run_dir])
    ctx = Context(cfg=load_config(ROOT), run_id="r", run_dir=run_dir,
                  video_path=clip, session=session)
    return s00_ingest.run(ctx), ctx


@pytest.mark.parametrize("session", [
    {},                                                      # nothing filled in
    {"athlete_name": "A"},                                   # only a name
    {"age": 3, "gender": "unknown", "bow_type": "Longbow",   # out of range / not listed
     "draw_hand": "both", "camera_view": "", "camera_distance_m": -1,
     "slow_motion_factor": "fast", "number_of_shots_expected": 0},
])
def test_s0_never_fails_over_session_fields(tmp_path, clip, session):
    r0, ctx = _s0(tmp_path, clip, session)
    assert r0.passed, [f"{c.name}: {c.detail}" for c in r0.failures]


def test_unusable_values_are_set_aside_and_named(tmp_path, clip):
    r0, ctx = _s0(tmp_path, clip, {"athlete_name": "A", "bow_type": "Longbow", "age": 3})
    ingest = ctx.read_json("00_ingest.json")
    assert ingest["session"]["athlete_name"] == "A"
    assert ingest["session"]["bow_type"] == NOT_PROVIDED
    assert ingest["session"]["age"] == NOT_PROVIDED
    (warn,) = [c for c in r0.warnings if c.name == "session_schema_valid"]
    assert "bow_type" in warn.detail and "age" in warn.detail


def test_web_form_with_garbage_numbers_does_not_raise():
    from archery.web.app import _session_from_form

    class Form(dict):
        def getlist(self, key):
            return []

    session, _ = _session_from_form(Form(age="abc", camera_distance_m="far",
                                         number_of_shots_expected="x", declared_frame_rate=""))
    assert not {"age", "camera_distance_m", "number_of_shots_expected"} & set(session)
