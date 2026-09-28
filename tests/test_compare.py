"""archery compare: the ten verification steps from docs/04_COMPARE.md.

Fixtures are real runs: the synthetic archer pushed through S3, S4 and S5, so
the comparison is exercised against genuine 05_metrics.json files rather than
hand-written ones. A hand-written fixture would let the reader's shape drift
away from what S5 actually writes, which is the failure this whole pipeline is
built to avoid.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from archery import io_guard
from archery.compare import CompareError, build, load_run
from archery.compare_report import render
from archery.steps import s03_kinematics, s04_phases, s05_stats
import synthetic
from helpers import make_run

ROOT = Path(__file__).resolve().parents[1]


def make_session_run(runs_dir: Path, run_id: str, *, n_shots: int = 3,
                     stretch: tuple[int, str, float] | None = None,
                     session: dict | None = None):
    """One finished run. `stretch` slows one shot's phase by a factor, by
    editing 05_metrics.json after S5 rather than by faking landmarks: the
    comparison reads that file and nothing else, so this is its real input."""
    df, truth = synthetic.build_multi(n_shots) if n_shots > 1 else synthetic.build()
    run_dir = runs_dir / run_id
    base = {"athlete_name": "Kalpana Ragar", "camera_view": "side-on (bow-arm side)",
            "session_date": "2026-08-13", "bow_type": "Recurve", "draw_hand": "right"}
    ctx = make_run(run_dir, df, truth["fps"], session={**base, **(session or {})})
    for step in (s03_kinematics, s04_phases, s05_stats):
        assert step.run(ctx).passed, run_id
    if stretch:
        shot_i, code, factor = stretch
        path = run_dir / "05_metrics.json"
        m = json.loads(path.read_text())
        p = m["per_shot"][shot_i]["phases"][code]
        p["duration_s"] = round(p["duration_s"] * factor, 3)
        path.write_text(json.dumps(m))
    return run_dir


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    io_guard.configure([runs])
    make_session_run(runs, "wk1__aaaa")
    make_session_run(runs, "wk2__bbbb", session={"session_date": "2026-08-20"})
    return runs


# 1 -----------------------------------------------------------------
def test_identical_sessions_produce_identical_durations_and_no_flags(two_runs):
    p = build(two_runs, ["wk1__aaaa", "wk2__bbbb"], 2.0, 3)
    a = [s for s in p["shots"] if s["session_label"] == "S1"]
    b = [s for s in p["shots"] if s["session_label"] == "S2"]
    assert len(a) == len(b) == 3
    for x, y in zip(a, b):
        assert x["phases"].keys() == y["phases"].keys()
        for code in x["phases"]:
            assert x["phases"][code]["duration_s"] == y["phases"][code]["duration_s"]
    assert p["deviations"] == []


# 2 and 3 -----------------------------------------------------------
def test_one_stretched_phase_is_flagged_and_only_that_one(tmp_path):
    runs = tmp_path / "runs"
    io_guard.configure([runs])
    make_session_run(runs, "a__1")
    make_session_run(runs, "b__2", stretch=(1, "AIM", 1.5))
    p = build(runs, ["a__1", "b__2"], 2.0, 3)
    flags = p["deviations"]
    assert flags, "the stretched shot should stand out"
    assert {(f["shot"], f["phase"]) for f in flags} == {("S2 shot 2", "AIM")}
    assert flags[0]["direction"] == "slower"


def test_the_baseline_excludes_the_shot_it_judges(tmp_path):
    runs = tmp_path / "runs"
    io_guard.configure([runs])
    make_session_run(runs, "a__1")
    make_session_run(runs, "b__2", stretch=(1, "AIM", 1.5))
    p = build(runs, ["a__1", "b__2"], 2.0, 3)
    flag = p["deviations"][0]
    per_shot = p["baselines"]["AIM"]["per_shot"]
    others = [v["value"] for lab, v in per_shot.items() if lab != flag["shot"]]
    assert flag["n_others"] == len(others)
    assert flag["baseline_mean_s"] == pytest.approx(sum(others) / len(others), abs=1e-3)
    # Pooling would have dragged the mean toward the outlier and softened the z.
    pooled = [v["value"] for v in per_shot.values()]
    assert abs(flag["baseline_mean_s"] - sum(pooled) / len(pooled)) > 1e-6


# 4 -----------------------------------------------------------------
def test_too_few_shots_withholds_the_sd_and_says_so(tmp_path):
    runs = tmp_path / "runs"
    io_guard.configure([runs])
    make_session_run(runs, "a__1", n_shots=1)
    make_session_run(runs, "b__2", n_shots=1)
    p = build(runs, ["a__1", "b__2"], 2.0, 3)
    assert p["totals"]["shots"] == 2
    for code, b in p["baselines"].items():
        assert b["sd"] is None and b["sd_withheld"] is True, code
    assert p["deviations"] == []          # nothing can be flagged without an SD


# 5 -----------------------------------------------------------------
def test_a_different_camera_view_keeps_timings_and_drops_positions(tmp_path):
    runs = tmp_path / "runs"
    io_guard.configure([runs])
    make_session_run(runs, "a__1")
    make_session_run(runs, "b__2", session={"camera_view": "front-on"})
    p = build(runs, ["a__1", "b__2"], 2.0, 3)
    assert len(p["sessions"]) == 2
    assert all(len(s["phases"]) for s in p["shots"])       # durations kept
    assert any("MIXED CAMERA VIEWS" in w for w in p["warnings"])
    angle = p["positions"]["elbow_draw_deg"]
    assert angle["groups"] == {}                            # no view has two sessions
    assert len(angle["not_comparable"]) == 2
    # Anchor distance is scale-normalised, so it still travels.
    assert p["positions"]["anchor_distance_norm"]["view_free"] is True


# 6 -----------------------------------------------------------------
def test_mixed_athletes_warn_and_are_labelled(tmp_path):
    runs = tmp_path / "runs"
    io_guard.configure([runs])
    make_session_run(runs, "a__1")
    make_session_run(runs, "b__2", session={"athlete_name": "Samarth Kumar"})
    p = build(runs, ["a__1", "b__2"], 2.0, 3)
    assert any(w.startswith("MIXED ATHLETES") for w in p["warnings"])
    assert p["totals"]["athletes"] == ["Kalpana Ragar", "Samarth Kumar"]
    assert {s["athlete"] for s in p["shots"]} == {"Kalpana Ragar", "Samarth Kumar"}


# 7 -----------------------------------------------------------------
def test_order_against_dates_warns_but_nothing_is_reordered(tmp_path):
    runs = tmp_path / "runs"
    io_guard.configure([runs])
    make_session_run(runs, "later__1", session={"session_date": "2026-09-01"})
    make_session_run(runs, "earlier__2", session={"session_date": "2026-08-01"})
    p = build(runs, ["later__1", "earlier__2"], 2.0, 3)
    assert any(w.startswith("ORDER DOES NOT MATCH DATES") for w in p["warnings"])
    assert [s["run_id"] for s in p["sessions"]] == ["later__1", "earlier__2"]
    assert [s["session_date"] for s in p["sessions"]] == ["2026-09-01", "2026-08-01"]


# 8 -----------------------------------------------------------------
def test_a_missing_run_is_named_and_the_rest_still_compare(tmp_path):
    runs = tmp_path / "runs"
    io_guard.configure([runs])
    make_session_run(runs, "a__1")
    make_session_run(runs, "b__2")
    p = build(runs, ["a__1", "ghost__9", "b__2"], 2.0, 3)
    assert [s["run_id"] for s in p["skipped_runs"]] == ["ghost__9"]
    assert "05_metrics.json" in p["skipped_runs"][0]["reason"]
    assert len(p["sessions"]) == 2
    # The surviving runs keep the order they were given, minus the gap.
    assert [s["label"] for s in p["sessions"]] == ["S1", "S3"]


def test_fewer_than_two_readable_runs_refuses(tmp_path):
    runs = tmp_path / "runs"
    io_guard.configure([runs])
    make_session_run(runs, "a__1")
    with pytest.raises(CompareError, match="Fewer than two"):
        build(runs, ["a__1", "ghost__9"], 2.0, 3)
    with pytest.raises(CompareError, match="at least two"):
        build(runs, ["a__1"], 2.0, 3)
    with pytest.raises(CompareError, match="more than once"):
        build(runs, ["a__1", "a__1"], 2.0, 3)


# 9 -----------------------------------------------------------------
def test_every_number_on_a_chart_is_in_the_evidence_file(two_runs, tmp_path):
    io_guard.configure([two_runs, tmp_path])
    p = build(two_runs, ["wk1__aaaa", "wk2__bbbb"], 2.0, 3)
    target, evidence = render(p, tmp_path / "outputs", tmp_path / "ev", "t")
    ev = json.loads(evidence.read_text())
    html = target.read_text(encoding="utf-8")

    filed = set()
    for s in ev["shots"]:
        for code, ph in s["phases"].items():
            if ph["duration_s"] is not None:
                filed.add(f'{ph["duration_s"]:.3f}')
        if s["total_s"] is not None:
            filed.add(f'{s["total_s"]:.3f}')
    assert filed
    # Every duration printed in the data table came from the evidence file.
    import re
    printed = set(re.findall(r">(\d+\.\d{3})</td>", html))
    assert printed and printed <= filed, printed - filed


def test_the_report_is_self_contained_and_versioned(two_runs, tmp_path):
    io_guard.configure([two_runs, tmp_path])
    p = build(two_runs, ["wk1__aaaa", "wk2__bbbb"], 2.0, 3)
    out = tmp_path / "outputs"
    first, _ = render(p, out, tmp_path / "ev", "kalpana")
    second, _ = render(p, out, tmp_path / "ev", "kalpana")
    assert first.name.endswith("_v01.html") and second.name.endswith("_v02.html")
    html = first.read_text(encoding="utf-8")
    for forbidden in ("<script", "http://", "https://"):
        assert forbidden not in html, forbidden
    assert "<svg" in html and 'class="chart"' in html
    # A table view exists, so the charts are never the only way to read it.
    assert "Every value behind the charts" in html


# 10 ----------------------------------------------------------------
def test_compare_writes_nothing_into_a_run_and_opens_no_video(two_runs, tmp_path):
    before = {p: p.stat().st_mtime_ns for p in two_runs.rglob("*") if p.is_file()}
    io_guard.configure([two_runs, tmp_path])
    payload = build(two_runs, ["wk1__aaaa", "wk2__bbbb"], 2.0, 3)
    render(payload, tmp_path / "outputs", tmp_path / "ev", "t")
    after = {p: p.stat().st_mtime_ns for p in two_runs.rglob("*") if p.is_file()}
    assert before == after, "compare must not touch a run directory"


def test_the_reader_opens_only_the_evidence_file():
    """Guard the boundary in source: compare reads 05_metrics.json and nothing
    else, which is what keeps it honest and what makes it fast."""
    src = (ROOT / "src" / "archery" / "compare.py").read_text()
    assert "05_metrics.json" in src
    for other in ("02_landmarks", "03_kinematics", "VideoCapture", "cv2", "imread"):
        assert other not in src, other


# extra: a phase that never happened is a gap, not a zero -----------
def test_an_undetected_phase_is_absent_rather_than_zero(two_runs):
    p = build(two_runs, ["wk1__aaaa", "wk2__bbbb"], 2.0, 3)
    for s in p["shots"]:
        for ph in s["phases"].values():
            assert ph["duration_s"] is None or ph["duration_s"] > 0
    detected = {c for s in p["shots"] for c in s["phases"]}
    assert set(p["phase_order"]) == detected


def test_low_confidence_boundaries_survive_into_the_payload(two_runs):
    p = build(two_runs, ["wk1__aaaa", "wk2__bbbb"], 2.0, 3)
    confs = {ph.get("confidence") for s in p["shots"] for ph in s["phases"].values()}
    assert confs and confs <= {"HIGH", "MEDIUM", "LOW", None}
