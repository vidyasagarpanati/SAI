"""S4 phase detection and S5 statistics against synthetic archers with known timing."""
from __future__ import annotations

import hashlib
import json

import pytest

from archery.steps import s03_kinematics, s04_phases, s05_stats
import synthetic
from helpers import make_run


@pytest.fixture(scope="module")
def single(tmp_path_factory):
    df, truth = synthetic.build()
    ctx = make_run(tmp_path_factory.mktemp("single"), df, truth["fps"])
    r3, r4, r5 = s03_kinematics.run(ctx), s04_phases.run(ctx), s05_stats.run(ctx)
    return ctx, r3, r4, r5


@pytest.fixture(scope="module")
def multi(tmp_path_factory):
    df, truth = synthetic.build_multi(3)
    ctx = make_run(tmp_path_factory.mktemp("multi"), df, truth["fps"])
    r3, r4, r5 = s03_kinematics.run(ctx), s04_phases.run(ctx), s05_stats.run(ctx)
    return ctx, truth, r4, r5


def phases_of(ctx, shot=0):
    p = json.loads((ctx.run_dir / "04_phases.json").read_text())
    return {x["phase"]: x for x in p["shots"][shot]["phases"]}, p


# ---------------------------------------------------------------- S4
def test_s4_passes_its_own_checks(single):
    _, _, r4, _ = single
    assert r4.passed, [f"{c.name}: {c.detail}" for c in r4.failures]


def test_s4_finds_exactly_one_shot(single):
    ctx, *_ = single
    _, p = phases_of(ctx)
    assert p["n_shots"] == 1


def test_s4_release_lands_inside_the_true_release(single):
    ctx, *_ = single
    _, p = phases_of(ctx)
    assert 3.48 <= p["shots"][0]["release_t_s"] <= 3.62


@pytest.mark.parametrize("phase,true_start,true_end,tol", [
    ("PRE_DRAW", 1.0, 1.6, 0.25),
    ("EXPANSION", 3.2, 3.5, 0.10),
    ("FOLLOW_THROUGH", 3.6, 4.1, 0.15),
])
def test_s4_phase_boundaries_match_known_truth(single, phase, true_start, true_end, tol):
    ctx, *_ = single
    ph, _ = phases_of(ctx)
    assert ph[phase]["detected"], ph[phase]["reason_not_detected"]
    assert abs(ph[phase]["start_t_s"] - true_start) <= tol
    assert abs(ph[phase]["end_t_s"] - true_end) <= tol


def test_s4_every_phase_detected_and_ordered(single):
    ctx, *_ = single
    ph, _ = phases_of(ctx)
    ends = -1
    for code in ["STANCE", "PRE_DRAW", "DRAW", "ANCHOR", "AIM", "EXPANSION",
                 "RELEASE", "FOLLOW_THROUGH", "RECOVERY"]:
        assert ph[code]["detected"], f"{code}: {ph[code]['reason_not_detected']}"
        assert ph[code]["start_frame"] > ends
        assert ph[code]["key_frame"] is not None
        ends = ph[code]["end_frame"]


def test_s4_multi_shot_detection(multi):
    ctx, truth, r4, _ = multi
    _, p = phases_of(ctx)
    assert p["n_shots"] == 3
    for got, want in zip([s["release_t_s"] for s in p["shots"]], truth["release_s"]):
        assert abs(got - want) <= 0.15


def test_s4_is_deterministic(single):
    from archery import io_guard
    ctx, *_ = single
    io_guard.configure([ctx.run_dir])   # fixtures share the process-wide whitelist
    a = hashlib.sha256((ctx.run_dir / "04_phases.json").read_bytes()).hexdigest()
    s04_phases.run(ctx)
    b = hashlib.sha256((ctx.run_dir / "04_phases.json").read_bytes()).hexdigest()
    assert a == b


def test_s4_undetectable_phase_is_reported_not_invented(tmp_path):
    """Flatten the bow arm so it never rises: PRE_DRAW must be detected=false with a reason."""
    df, truth = synthetic.build()
    from archery.landmarks import ID
    for j in ("LEFT_WRIST", "LEFT_ELBOW"):
        rows = df["lm"] == ID[j]
        df.loc[rows, "y"] = df.loc[rows, "y"].iloc[0] - 0.01
    ctx = make_run(tmp_path / "flat", df, truth["fps"])
    s03_kinematics.run(ctx)
    s04_phases.run(ctx)
    ph, _ = phases_of(ctx)
    assert ph["PRE_DRAW"]["detected"] is False
    assert ph["PRE_DRAW"]["reason_not_detected"]


# ---------------------------------------------------------------- S5
def test_s5_passes_its_own_checks(single):
    *_, r5 = single
    assert r5.passed, [f"{c.name}: {c.detail}" for c in r5.failures]


def test_s5_is_strict_json_with_evidence(single):
    ctx, *_ = single
    text = (ctx.run_dir / "05_metrics.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    m = json.loads(text)
    assert len(m["evidence_index"]) > 100
    assert "shot1.AIM.elbow_bow_deg.mean" in m["evidence_index"]


def test_s5_single_shot_suppresses_cross_shot_sd(single):
    ctx, *_ = single
    m = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    c = m["cross_shot"]["by_phase"]["AIM"]["elbow_bow_deg"]
    # spelled out, not "3": digits in our own status text get quoted back by the
    # model and then rejected by the grounding check as ungrounded numbers
    assert c.get("sd") is None and "requires at least three shots" in c["status"]
    assert "requires at least three shots" in m["consistency_rankings"]["status"]
    assert not any(ch.isdigit() for ch in c["status"])


def test_s5_three_shots_produce_cross_shot_stats(multi):
    ctx, _, _, r5 = multi
    assert r5.passed, [f"{c.name}: {c.detail}" for c in r5.failures]
    m = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    c = m["cross_shot"]["by_phase"]["AIM"]["elbow_bow_deg"]
    assert c["n_shots"] == 3 and c["sd"] is not None and c["cv_pct"] is not None
    assert m["consistency_rankings"]["most_consistent_phase"] in m["cross_shot"]["by_phase"]
    tilt = m["cross_shot"]["by_phase"]["AIM"]["shoulder_tilt_deg"]
    assert tilt["cv_pct"] is None, "CV must not be reported for a non-ratio-scale measure"


def test_s5_uses_no_uncited_benchmark(single):
    ctx, *_ = single
    m = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    assert m["benchmarks"]["rows"] == []
    assert m["benchmarks"]["no_benchmark"]["elbow_bow_deg"] == "Individualized assessment required"


def test_s5_bow_arm_straighter_than_draw_arm_at_full_draw(single):
    ctx, *_ = single
    m = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    aim = m["per_shot"][0]["phases"]["AIM"]["measures"]
    assert aim["elbow_bow_deg"]["mean"] > aim["elbow_draw_deg"]["mean"]
