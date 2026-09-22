"""Tuning a phase threshold must never force pose estimation to re-run."""
import copy
from pathlib import Path

from archery.config import load_config
from archery.context import Context
from archery.runner import compute_input_hash
from archery.runstate import RunState
from archery import io_guard

ROOT = Path(__file__).resolve().parents[1]


def _ctx(tmp_path, cfg, session):
    io_guard.configure([tmp_path])
    state = RunState.load_or_create(tmp_path, run_id="r", video_sha256="abc", config_hash="x")
    for s in ("S0", "S1", "S2", "S3"):
        state.data["steps"][s] = {"status": "DONE", "output_hash": f"out-{s}"}
    return Context(cfg=cfg, run_id="r", run_dir=tmp_path, video_path=Path("v.mov"),
                   session=session, state=state)


def test_phase_rule_change_invalidates_s4_but_not_s2(tmp_path):
    cfg = load_config(ROOT)
    session = {"draw_hand": "right", "athlete_name": "A"}
    before = {s: compute_input_hash(_ctx(tmp_path, cfg, session), s) for s in ("S1", "S2", "S3", "S4")}
    cfg2 = copy.deepcopy(cfg)
    cfg2.phase_rules["shot_detection"]["anchor_distance_enter"] = 0.5
    after = {s: compute_input_hash(_ctx(tmp_path, cfg2, session), s) for s in ("S1", "S2", "S3", "S4")}
    assert before["S1"] == after["S1"] and before["S2"] == after["S2"] and before["S3"] == after["S3"]
    assert before["S4"] != after["S4"]


def test_renaming_athlete_does_not_redo_pose(tmp_path):
    cfg = load_config(ROOT)
    a = compute_input_hash(_ctx(tmp_path, cfg, {"draw_hand": "right", "athlete_name": "A"}), "S2")
    b = compute_input_hash(_ctx(tmp_path, cfg, {"draw_hand": "right", "athlete_name": "B"}), "S2")
    assert a == b


def test_draw_hand_change_redoes_kinematics(tmp_path):
    cfg = load_config(ROOT)
    a = compute_input_hash(_ctx(tmp_path, cfg, {"draw_hand": "right"}), "S3")
    b = compute_input_hash(_ctx(tmp_path, cfg, {"draw_hand": "left"}), "S3")
    assert a != b
