"""Prompt budget, key naming and prescriptive fields.

The first real run sent ~15.9k-token prompts into a 16k window, so replies were
truncated. These tests fix the budget in place for the worst realistic case:
4 shots, 9 phases, every earlier section present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from archery.grounding import check_text
from archery.llm import FakeLLM
from archery.phase_defs import ORDER
from archery.report_spec import CALLS, build_prompt
from archery.steps import s03_kinematics, s04_phases, s05_stats
import synthetic
from helpers import make_run

LIMITS = {"max_evidence_lines": 90, "max_prior_points": 4, "max_prior_chars": 220}


@pytest.fixture(scope="module")
def four_shot(tmp_path_factory):
    df, truth = synthetic.build_multi(4)
    ctx = make_run(tmp_path_factory.mktemp("budget"), df, truth["fps"])
    for step in (s03_kinematics, s04_phases, s05_stats):
        assert step.run(ctx).passed
    m = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    detected = [p for p in ORDER
                if any(x["phase"] == p and x["detected"] for sh in m["phase_timeline"] for x in sh["phases"])]
    return ctx, m, detected


def test_every_prompt_fits_well_inside_the_context_window(four_shot):
    ctx, m, detected = four_shot
    fake, outs, sizes = FakeLLM(), {}, {}
    for sid, deps, imgs in CALLS:
        if sid == "s04_phase":
            per = {}
            for ph in detected:
                sysm, user, sch = build_prompt(sid, ctx.cfg.prompts, m, outs, deps, detected, ph, True, LIMITS)
                sizes[f"{sid}/{ph}"] = (len(sysm) + len(user)) // 4
                per[ph] = fake.chat_json(sysm, user, sch)
            outs[sid] = per
            continue
        sysm, user, sch = build_prompt(sid, ctx.cfg.prompts, m, outs, deps, detected, None, bool(imgs), LIMITS)
        sizes[sid] = (len(sysm) + len(user)) // 4
        outs[sid] = fake.chat_json(sysm, user, sch)
    worst = max(sizes.values())
    assert worst < 6000, f"prompt budget exceeded: {sorted(sizes.items(), key=lambda x: -x[1])[:3]}"
    assert len(detected) == 9 and m["session"]["n_shots"] == 4


def test_evidence_list_is_capped(four_shot):
    ctx, m, detected = four_shot
    _, user, _ = build_prompt("s09_errors", ctx.cfg.prompts, m, {}, [], detected, None, False,
                              {"max_evidence_lines": 20})
    import re as _re
    body = user.split("EVIDENCE (cite values ONLY as {{key}})\n", 1)[1]
    lines = [ln for ln in body.splitlines() if _re.match(r"^\{\{[\w.\-]+\}\} = ", ln)]
    assert len(lines) == 20, lines[-2:]
    assert "capped at 20 entries" in body


def test_timing_keys_follow_the_measure_naming(four_shot):
    _, m, _ = four_shot
    ev = m["evidence_index"]
    assert "all.AIM.duration_s.mean" in ev and "all.AIM.duration_s.sd" in ev
    assert not [k for k in ev if ".duration.mean_s" in k], "old inconsistent key names still present"


def test_dose_fields_accept_a_bare_number_but_not_a_bad_placeholder():
    assert check_text("5", {}, prescriptive=True) == []
    assert check_text("3 x 10", {}, prescriptive=True) == []
    bad = check_text("{{s15_priorities.priorities[0].sets}}", {"a.b": {}}, prescriptive=True)
    assert bad and "only keys from the EVIDENCE list" in bad[0]


def test_unknown_key_gets_a_did_you_mean_hint():
    ev = {"all.ANCHOR.duration_s.mean": {"value": 1.0, "units": "s"}}
    problems = check_text("held for {{all.ANCHOR.duration.mean_s}}", ev)
    assert problems and "Did you mean {{all.ANCHOR.duration_s.mean}}?" in problems[0]
