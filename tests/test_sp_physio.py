"""SP physio ingest, against the real posturography export.

The fixture is the export the coach actually supplied, not a hand-written one,
because every rule in forceplate.py exists because of something in that file:
four athletes in one export, channels that are zero in every row, and trials of
different durations that must never be averaged together.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from archery.config import load_config
from archery.context import Context
from archery.forceplate import ALL_ZERO, META_COLUMNS, ForcePlateError, load
from archery.steps import sp_physio

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(__file__).parent / "data" / "body_sway_sample.csv"


def ctx_for(tmp_path: Path, session: dict) -> Context:
    from archery import io_guard
    io_guard.configure([tmp_path])
    return Context(cfg=load_config(ROOT), run_id=tmp_path.name, run_dir=tmp_path,
                   video_path=Path("synthetic.mov"), session=session)


def base_session(**over) -> dict:
    s = {"athlete_name": "Kalpana Ragar", "age": 19, "height_cm": 160.0,
         "weight_kg": 63.9, "bow_type": "Recurve", "draw_hand": "right",
         "camera_view": "side-on",
         "physio": {"heart_rate": {"rest_bpm": 62, "mean_bpm": 118, "max_bpm": None},
                    "force_plate": {"file": str(SAMPLE), "images": []}}}
    s.update(over)
    return s


# ---------------------------------------------------------------- the contract
def test_header_contract_is_checked_by_name(tmp_path):
    rows = list(csv.DictReader(open(SAMPLE, encoding="utf-8-sig")))
    dropped = META_COLUMNS["subject_name"]
    bad = tmp_path / "renamed.csv"
    fields = [f for f in rows[0] if f != dropped]
    with open(bad, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    with pytest.raises(ForcePlateError) as exc:
        load(bad, "Kalpana Ragar")
    assert dropped in str(exc.value)          # names the column, does not guess


def test_selects_one_athlete_out_of_four(tmp_path):
    export = load(SAMPLE, "Kalpana Ragar")
    assert export.n_rows_total == 20
    assert len(export.trials) == 4
    assert export.athlete.replace("  ", " ") == "Kalpana Ragar"
    assert len(export.other_athletes) == 3


def test_ambiguous_name_refuses_rather_than_guessing(tmp_path):
    # "a" is a substring of every name in the file.
    with pytest.raises(ForcePlateError) as exc:
        load(SAMPLE, "a")
    assert "more than one athlete" in str(exc.value)


def test_unknown_name_lists_the_candidates():
    with pytest.raises(ForcePlateError) as exc:
        load(SAMPLE, "Nobody At All")
    assert "Ansh" in str(exc.value)


def test_channels_that_are_zero_everywhere_are_not_measurements():
    export = load(SAMPLE, "Kalpana Ragar")
    for key in ("sway_v_left", "sway_v_right", "lr_sway_v_ratio"):
        assert export.unavailable.get(key) == ALL_ZERO
        assert all(key not in t.values for t in export.trials)


def test_comma_inside_a_notes_field_parses(tmp_path):
    rows = list(csv.DictReader(open(SAMPLE, encoding="utf-8-sig")))
    rows[0][META_COLUMNS["subject_notes"]] = "Recurve, 68 inch, 26 lb"
    out = tmp_path / "commas.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    export = load(out, "Samarth Kumar")
    assert export.subject["notes"] == "Recurve, 68 inch, 26 lb"


def test_conditions_of_different_duration_never_share_a_key(tmp_path):
    res = sp_physio.run(ctx_for(tmp_path, base_session()))
    payload = json.loads((tmp_path / "00b_physio.json").read_text())
    labels = {c["condition"] for c in payload["force_plate"]["conditions"]}
    assert "DRAW_TO_HOLD_R_10S" in labels and "FREE_50S" in labels
    keys = payload["evidence"]
    assert (keys["posture.DRAW_TO_HOLD_R_10S.sway_path_total.mean"]["value"]
            != keys["posture.FREE_50S.sway_path_total.mean"]["value"])
    assert res.passed


def test_sd_is_withheld_below_the_minimum_trial_count(tmp_path):
    ctx = ctx_for(tmp_path, base_session())
    sp_physio.run(ctx)
    payload = json.loads((tmp_path / "00b_physio.json").read_text())
    floor = payload["force_plate"]["min_trials_for_sd"]
    for cond in payload["force_plate"]["conditions"]:
        for name, m in cond["measures"].items():
            if m["n"] < floor:
                assert m["sd"] is None, f"{cond['condition']}.{name}"


# ---------------------------------------------------------------- derived values
def test_bmi_and_tanaka_estimate_come_from_cited_formulas(tmp_path):
    sp_physio.run(ctx_for(tmp_path, base_session(age=16, height_cm=170.0, weight_kg=60.0)))
    p = json.loads((tmp_path / "00b_physio.json").read_text())
    assert p["anthropometrics"]["bmi"] == pytest.approx(60.0 / 1.70 ** 2, abs=0.05)
    assert p["anthropometrics"]["bmi_source"]
    assert p["heart_rate"]["max_estimated_bpm"] == pytest.approx(208 - 0.7 * 16, abs=0.5)
    assert "Tanaka" in p["heart_rate"]["max_source"]
    # An estimate may never be presented with the confidence of a measurement.
    assert p["evidence"]["hr.max_estimated_bpm"]["confidence"] == "MEDIUM"
    assert p["evidence"]["hr.rest_bpm"]["confidence"] == "HIGH"
    zones = {z["zone"]: z for z in p["heart_rate"]["zones"]}
    assert zones["Z2"]["low_bpm"] < zones["Z4"]["low_bpm"]


def test_youth_athlete_is_flagged(tmp_path):
    res = sp_physio.run(ctx_for(tmp_path, base_session(age=16)))
    p = json.loads((tmp_path / "00b_physio.json").read_text())
    assert p["anthropometrics"]["is_youth_athlete"] is True
    assert any(c.name == "youth_athlete_flagged" for c in res.checks)
    adult = sp_physio.run(ctx_for(tmp_path, base_session(age=25)))
    assert not any(c.name == "youth_athlete_flagged" for c in adult.checks)


def test_typed_anthropometrics_disagreeing_with_the_export_are_reported(tmp_path):
    res = sp_physio.run(ctx_for(tmp_path, base_session(weight_kg=75.0)))
    p = json.loads((tmp_path / "00b_physio.json").read_text())
    assert p["force_plate"]["anthropometric_conflicts"]
    check = [c for c in res.checks if c.name == "anthropometrics_agree_with_export"][0]
    assert not check.ok and check.severity == "WARN"
    # The entered value is what the run uses, and it is not silently replaced.
    assert p["anthropometrics"]["weight_kg"] == 75.0
    assert res.passed


# ---------------------------------------------------------------- absent inputs
def test_no_inputs_at_all_yields_no_keys_and_no_zeros(tmp_path):
    session = {"athlete_name": "Kalpana", "bow_type": "Recurve", "draw_hand": "right",
               "camera_view": "side-on"}
    res = sp_physio.run(ctx_for(tmp_path, session))
    p = json.loads((tmp_path / "00b_physio.json").read_text())
    assert p["evidence"] == {}
    assert p["force_plate"] == {}
    assert all(i["status"] == "NOT PROVIDED" for i in p["inventory"])
    assert res.passed


def test_images_alone_produce_no_posture_numbers(tmp_path):
    img = tmp_path / "plate.png"
    img.write_bytes(b"\x89PNG\r\n")
    session = base_session()
    session["physio"]["force_plate"] = {"file": None, "images": [str(img)]}
    sp_physio.run(ctx_for(tmp_path, session))
    p = json.loads((tmp_path / "00b_physio.json").read_text())
    assert not [k for k in p["evidence"] if k.startswith("posture.")]
    assert p["force_plate_images"] == [str(img)]


def test_a_broken_export_fails_the_step_without_writing_evidence(tmp_path):
    bad = tmp_path / "not_a_plate.csv"
    bad.write_text("a,b\n1,2\n")
    session = base_session()
    session["physio"]["force_plate"]["file"] = str(bad)
    res = sp_physio.run(ctx_for(tmp_path, session))
    assert not res.passed
    p = json.loads((tmp_path / "00b_physio.json").read_text())
    assert not [k for k in p["evidence"] if k.startswith("posture.")]


# ---------------------------------------------------------------- cache keys
def test_editing_the_export_re_runs_sp(tmp_path):
    from archery.runner import STEP_DEPS, compute_input_hash
    from archery.runstate import RunState
    copy = tmp_path / "plate.csv"
    copy.write_bytes(SAMPLE.read_bytes())
    session = base_session()
    session["physio"]["force_plate"]["file"] = str(copy)
    ctx = ctx_for(tmp_path, session)
    ctx.state = RunState.load_or_create(tmp_path, run_id="t", video_path="v",
                                        video_sha256="0" * 64, config_hash="c")
    before = compute_input_hash(ctx, "SP")
    copy.write_text(copy.read_text(encoding="utf-8").replace("133.6", "150.0"), encoding="utf-8")
    assert compute_input_hash(ctx, "SP") != before
    assert "physio.force_plate.file" in STEP_DEPS["SP"]["files"]


def test_physio_inputs_do_not_invalidate_the_video_steps(tmp_path):
    """Changing a heart rate must not re-run pose estimation or the annotate
    step. S0 echoes the session into its output, so it must not see the block."""
    from archery.steps.s00_ingest import SP_OWNED, normalise_session
    session = base_session()
    a, _ = normalise_session(session)
    session["physio"]["heart_rate"]["rest_bpm"] = 71
    session["height_cm"] = 999 if False else 161.0
    b, _ = normalise_session(session)
    assert a == b
    assert "physio" in SP_OWNED
