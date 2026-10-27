"""The local web UI: the eight verification steps in docs/05_WEB_UI.md.

The app is exercised against real run directories, built by pushing the
synthetic archer through S3 to S10 with the fake model, so the library parses
the same state.json the pipeline actually writes rather than a hand-made one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient            # noqa: E402

from archery import io_guard                          # noqa: E402
from archery.llm import FakeLLM                       # noqa: E402
from archery.steps import (s03_kinematics, s04_phases, s05_stats, s06_annotate,  # noqa: E402
                           s08_narrate, s09_verify, s10_render)
from archery.web import library as lib                # noqa: E402
from archery.web.app import create_app                # noqa: E402
import synthetic                                      # noqa: E402
from helpers import make_run                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A project root with two finished runs and one that failed part-way."""
    base = tmp_path_factory.mktemp("site")
    runs, outputs = base / "runs", base / "outputs"
    for name, shots in (("kalpana__aaaa", 3), ("samarth__bbbb", 1)):
        df, truth = synthetic.build_multi(shots) if shots > 1 else synthetic.build()
        ctx = make_run(runs / name, df, truth["fps"], frames=(640, 360), outputs_dir=outputs,
                       session={"athlete_name": name.split("__")[0].title(),
                                "session_date": "2026-08-13",
                                "camera_view": "side-on (bow-arm side)"})
        from archery.runstate import RunState
        ctx.state = RunState.load_or_create(ctx.run_dir, run_id=name,
                                            video_path=f"D:/vid/{name}.mov",
                                            video_sha256="0" * 64, config_hash="c")
        # The steps that need a real video are recorded as done rather than
        # run: this fixture is about what the UI makes of state.json, not
        # about decoding frames.
        for sid in ("S0", "SP", "S1", "S2"):
            rec = ctx.state.record(sid)
            rec.update({"status": "DONE", "duration_s": 3.0, "checks": [],
                        "outputs": {}, "finished": "2026-08-13T00:00:00+00:00"})
        ctx.llm = FakeLLM()
        for mod, sid in ((s03_kinematics, "S3"), (s04_phases, "S4"), (s05_stats, "S5"),
                         (s06_annotate, "S6"), (s08_narrate, "S8"), (s09_verify, "S9"),
                         (s10_render, "S10")):
            res = mod.run(ctx)
            ctx.state.start(sid)
            ctx.state.record(sid)["duration_s"] = 4.0
            ctx.state.finish(sid, res)

    # A run that stopped at S3, which the library must still render.
    broken = runs / "broken__cccc"
    broken.mkdir(parents=True)
    (broken / "state.json").write_text(json.dumps({
        "created": "2026-08-01T00:00:00+00:00", "run_id": "broken__cccc",
        "video_path": "D:/vid/broken.mov",
        "steps": {"S0": {"status": "DONE", "duration_s": 2, "checks": [{"name": "a", "ok": True}]},
                  "S3": {"status": "FAILED", "duration_s": 1, "error": "boom",
                         "checks": [{"name": "core_measures_available", "ok": False,
                                     "severity": "FAIL", "detail": "worst NaN rate 71%"}]}}}))

    io_guard.configure([base])
    app = create_app(ROOT, base / "videos", work_dir=base)
    (base / "videos").mkdir(exist_ok=True)
    (base / "videos" / "new_one.mov").write_bytes(b"\x00" * 64)
    (base / "videos" / "plate.csv").write_text("Project,Visit\n")
    return base, TestClient(app)


# 3 -----------------------------------------------------------------
def test_the_library_lists_finished_and_failed_runs(site):
    _, client = site
    r = client.get("/")
    assert r.status_code == 200
    for expected in ("Kalpana", "Samarth", "broken__cccc", "DONE", "FAILED"):
        assert expected in r.text, expected


def test_a_failed_run_shows_the_check_that_failed(site):
    _, client = site
    r = client.get("/run/broken__cccc")
    assert r.status_code == 200
    assert "core_measures_available" in r.text and "worst NaN rate 71%" in r.text


def test_a_partial_report_is_labelled(site):
    base, client = site
    runs = lib.list_runs(base / "runs", base / "outputs")
    assert {r.run_id for r in runs} == {"kalpana__aaaa", "samarth__bbbb", "broken__cccc"}
    done = [r for r in runs if r.status == "DONE"]
    assert done and all(r.report is not None for r in done)


# 1 -----------------------------------------------------------------
@pytest.mark.parametrize("bad", ["..", "../../etc", "a/b", "a\\b", "", ".",
                                 "%2e%2e", "run id with spaces"])
def test_a_run_id_that_is_not_a_directory_name_is_refused(site, bad):
    base, _ = site
    with pytest.raises((lib.Unsafe, lib.NotFound)):
        lib.load_run(base / "runs", bad, base / "outputs")


def test_artefact_routes_refuse_traversal(site):
    _, client = site
    for bad in ("..", "../../etc/passwd", "a%2Fb"):
        assert client.get(f"/artefact/report/{bad}").status_code in (404, 400)
    assert client.get("/artefact/nonsense/kalpana__aaaa").status_code == 404
    assert client.get("/frame/kalpana__aaaa/../../state.json").status_code in (404, 400)


def test_a_recorded_path_outside_the_served_roots_is_not_served(site, tmp_path):
    """Even a path the PIPELINE recorded is refused if it does not land where
    this server is allowed to read. The recorded path is not a permission."""
    base, _ = site
    outside = tmp_path / "elsewhere.html"
    outside.write_text("secret")
    state = base / "runs" / "kalpana__aaaa" / "state.json"
    data = json.loads(state.read_text())
    keep = data["steps"]["S10"]["outputs"]["report"]
    data["steps"]["S10"]["outputs"]["report"] = str(outside)
    state.write_text(json.dumps(data))
    try:
        run = lib.load_run(base / "runs", "kalpana__aaaa", base / "outputs")
        assert run.report is None
        with pytest.raises(lib.NotFound):
            lib.artefact(base / "runs", base / "outputs", "kalpana__aaaa", "report")
    finally:
        data["steps"]["S10"]["outputs"]["report"] = keep
        state.write_text(json.dumps(data))


# 7 -----------------------------------------------------------------
def test_the_report_and_the_video_are_served(site):
    base, client = site
    run = lib.load_run(base / "runs", "kalpana__aaaa", base / "outputs")
    r = client.get("/artefact/report/kalpana__aaaa")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "Archery Biomechanics Report" in r.text
    assert client.get("/artefact/evidence/kalpana__aaaa").status_code == 200
    # S7 did not run in this fixture, so the video is honestly absent.
    assert run.video is None
    assert client.get("/artefact/video/kalpana__aaaa").status_code == 404


def test_the_pages_all_render(site):
    _, client = site
    for path in ("/", "/new", "/jobs", "/doctor", "/run/kalpana__aaaa",
                 "/?athlete=Kalpana", "/?status=FAILED"):
        r = client.get(path)
        assert r.status_code == 200, (path, r.status_code)
        assert "<html" in r.text


def test_the_api_feeds_the_progress_poll(site):
    _, client = site
    d = client.get("/api/run/kalpana__aaaa").json()
    assert d["status"] == "DONE" and d["has_report"] is True
    assert [s["step"] for s in d["steps"]][:2] == ["S0", "SP"]
    assert client.get("/api/run/ghost").status_code in (404, 400)


# 4 -----------------------------------------------------------------
def test_the_same_video_cannot_be_queued_twice(site):
    from archery.web.jobs import JobQueue
    base, _ = site
    q = JobQueue(base, python="/bin/true")
    video = base / "videos" / "new_one.mov"
    q.submit(video)
    with pytest.raises(ValueError, match="already"):
        q.submit(video)


def test_several_videos_queue_in_order(site):
    from archery.web.jobs import JobQueue
    base, _ = site
    q = JobQueue(base, python="/bin/true")
    ids = [q.submit(base / "videos" / f"v{i}.mov").id for i in range(3)]
    assert len(ids) == 3
    import time
    for _ in range(50):
        if all(q.get(i).status in ("DONE", "FAILED") for i in ids):
            break
        time.sleep(0.05)
    # /bin/true exits 0 without producing a run, which is a DONE job with no run id.
    assert all(q.get(i).status == "DONE" for i in ids)
    assert all(q.get(i).run_id is None for i in ids)


# 5 -----------------------------------------------------------------
def test_cancelling_a_queued_job_removes_it(site):
    from archery.web.jobs import JobQueue
    base, _ = site
    q = JobQueue(base, python="/bin/sleep")
    first = q.submit(base / "videos" / "a.mov")
    second = q.submit(base / "videos" / "b.mov")
    detail = q.cancel(second.id)
    assert "queue" in detail and q.get(second.id).status == "CANCELLED"
    q.cancel(first.id)
    q.shutdown()


# 6 -----------------------------------------------------------------
def test_the_form_rejects_what_the_cli_rejects(site):
    _, client = site
    r = client.post("/new", data={"videos": [], "athlete_name": "X"}, follow_redirects=False)
    assert r.status_code == 303 and "Pick+at+least+one" in r.headers["location"]

    base, _ = site
    video = str(base / "videos" / "new_one.mov")
    r = client.post("/new", data={"videos": [video], "athlete_name": "X",
                                  "profile": "Height: 17 cms"}, follow_redirects=False)
    assert r.status_code == 303
    assert "outside" in r.headers["location"], r.headers["location"]


def test_a_good_form_writes_a_session_file_and_echoes_it_back(site):
    base, client = site
    video = str(base / "videos" / "new_one.mov")
    r = client.post("/new", data={
        "videos": [video], "athlete_name": "Kalpana Ragar", "bow_type": "Recurve",
        "draw_hand": "right", "camera_view": "side-on", "session_date": "2026-09-01",
        "profile": "Height: 160 cms\nWeight: 63.9 kgs\nResting HR: 62 bpm"})
    assert r.status_code == 200
    assert "160 cm" in r.text and "62 bpm" in r.text
    assert "NOT PROVIDED" in r.text                     # Max HR was not given
    written = json.loads((base / "sessions" / "new_one.json").read_text())
    assert written["height_cm"] == 160.0
    assert written["physio"]["heart_rate"]["rest_bpm"] == 62
    assert written["session_date"] == "2026-09-01"
    # The job was queued against the real CLI; stop it before it does anything.
    from archery.web.app import create_app as _unused   # noqa: F401
    for job in client.app.state.queue.all():
        try:
            client.app.state.queue.cancel(job.id)
        except LookupError:
            pass
    client.app.state.queue.shutdown()


# 2 -----------------------------------------------------------------
def test_the_web_package_opens_nothing_for_writing_except_the_session_file():
    """Mirrors the pose-worker isolation test. Every write in the pipeline goes
    through io_guard; the web layer must not grow a path of its own."""
    web = ROOT / "src" / "archery" / "web"
    for py in web.glob("*.py"):
        src = py.read_text()
        for pattern in (r'(?<!guarded_)open\([^)]*["\']w', r'\.write_text\(',
                        r'\.write_bytes\(', r'shutil\.(copy|move|rmtree)',
                        r'os\.remove', r'\.unlink\('):
            assert not re.search(pattern, src), f"{py.name}: {pattern}"
    # The one write that exists goes through the guard.
    app_src = (web / "app.py").read_text()
    assert "guarded_open" in app_src and "guarded_path" in app_src


def test_compare_is_reachable_from_the_library(site):
    base, client = site
    r = client.post("/compare", data={"runs_selected": ["kalpana__aaaa"], "sd": "2.0"},
                    follow_redirects=False)
    assert r.status_code == 303
    name = r.headers["location"].rsplit("/", 1)[-1]
    assert name.startswith("Compare_") and name.endswith(".html")
    page = client.get(f"/compare/{name}")
    assert page.status_code == 200 and "Shot comparison" in page.text
    assert client.get("/compare/../state.json").status_code in (404, 400)


def test_compare_refusals_are_shown_not_swallowed(site):
    _, client = site
    r = client.post("/compare", data={"runs_selected": ["samarth__bbbb"], "sd": "2.0"})
    assert r.status_code == 200
    assert "refused" in r.text.lower() and "compared against" in r.text


# 8 -----------------------------------------------------------------
def test_serve_binds_loopback_only():
    src = (ROOT / "src" / "archery" / "cli.py").read_text()
    block = src[src.index("def cmd_serve"):src.index("def cmd_status")]
    assert 'host="127.0.0.1"' in block
    assert "0.0.0.0" not in block


def test_grouping_by_session_date_keeps_undated_runs_last(site):
    _, client = site
    r = client.get("/?group=date")
    assert r.status_code == 200
    assert "2026-08-13" in r.text and "No session date set" in r.text
    # The undated bucket must come after the dated one, not be sorted into it.
    assert r.text.index("2026-08-13") < r.text.index("No session date set")


def test_filters_narrow_the_library(site):
    _, client = site
    assert "Samarth" not in client.get("/?athlete=Kalpana").text.split("<table")[1]
    failed = client.get("/?status=FAILED")
    assert "broken__cccc" in failed.text and "kalpana__aaaa" not in failed.text.split("<table")[1]
