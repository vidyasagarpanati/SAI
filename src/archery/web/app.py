"""FastAPI routes for the local UI.

The rule this file keeps: it serves and it starts, it never computes. Every
number comes from an artefact the pipeline wrote, every analysis is a
subprocess of the CLI, and nothing here opens a file for writing except the
session file a new run needs, which goes through the same io_guard whitelist
as everything else.
"""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from archery import __version__, io_guard
from archery.config import load_config
from archery.runstate import STEP_NAMES, STEP_ORDER
from archery.web import library as lib
from archery.web.jobs import JobQueue


def create_app(root: Path | None = None, video_folder: Path | None = None,
               work_dir: Path | None = None) -> FastAPI:
    """`root` holds config, templates and schemas. `work_dir` overrides where
    runs, outputs and session files live, which is what lets a test drive the
    real app against a scratch directory instead of the project's own."""
    cfg = load_config(root)
    if work_dir:
        work = Path(work_dir)
        runs_root, outputs_root = work / "runs", work / "outputs"
        sessions_dir = work / "sessions"
    else:
        runs_root, outputs_root = cfg.paths.runs_dir, cfg.paths.outputs_dir
        sessions_dir = cfg.root / "sessions"
    io_guard.configure([runs_root, outputs_root, sessions_dir])
    watch = Path(video_folder).expanduser() if video_folder else None

    app = FastAPI(title="SAI archery", docs_url=None, redoc_url=None)
    app.state.cfg = cfg
    app.state.queue = JobQueue(cfg.root)
    app.state.video_folder = watch
    templates = Jinja2Templates(directory=str(cfg.root / "templates" / "web"))
    templates.env.globals.update(version=__version__, step_order=STEP_ORDER,
                                 step_names=STEP_NAMES)

    def page(request: Request, name: str, **ctx) -> HTMLResponse:
        q = app.state.queue
        return templates.TemplateResponse(
            request=request, name=name,
            context={"active_jobs": q.active(), "video_folder": app.state.video_folder, **ctx})

    def runs() -> list[lib.Run]:
        return lib.list_runs(runs_root, outputs_root)

    # ------------------------------------------------------------- library
    @app.get("/", response_class=HTMLResponse)
    def library(request: Request, athlete: str = "", status: str = "", group: str = ""):
        all_runs = runs()
        shown = [r for r in all_runs
                 if (not athlete or r.athlete == athlete)
                 and (not status or r.status == status)]
        groups = None
        if group == "date":
            # Newest session first; runs with no session_date collect at the end
            # rather than being guessed into an order.
            buckets: dict[str, list] = {}
            for r in shown:
                buckets.setdefault(r.session_date or "", []).append(r)
            groups = [(k or "No session date set", v)
                      for k, v in sorted(buckets.items(), key=lambda kv: kv[0], reverse=True)]
        return page(request, "library.html.j2", runs=shown, groups=groups, group=group,
                    athletes=sorted({r.athlete for r in all_runs}),
                    statuses=sorted({r.status for r in all_runs}),
                    athlete=athlete, status=status, total=len(all_runs))

    # ------------------------------------------------------------ run detail
    @app.get("/run/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str):
        try:
            run = lib.load_run(runs_root, run_id, outputs_root)
        except (lib.NotFound, lib.Unsafe) as exc:
            raise HTTPException(404, str(exc)) from exc
        frames = sorted((run.path / "06_frames").glob("*.jpg")) if \
            (run.path / "06_frames").is_dir() else []
        return page(request, "run.html.j2", run=run,
                    job=app.state.queue.by_run_id(run_id),
                    frames=[f.name for f in frames])

    @app.get("/api/run/{run_id}")
    def run_state(run_id: str):
        """Polled by the run page. Structure from state.json, commentary from
        the job's captured stdout."""
        try:
            run = lib.load_run(runs_root, run_id, outputs_root)
        except (lib.NotFound, lib.Unsafe) as exc:
            raise HTTPException(404, str(exc)) from exc
        job = app.state.queue.by_run_id(run_id)
        return {
            "status": run.status, "current_step": run.current_step,
            "total": run.total, "done_steps": run.done_steps,
            "steps": [{"step": s.step, "name": s.name, "status": s.status,
                       "took": s.took, "checks": f"{s.checks_ok}/{s.checks_total}"
                       if s.checks_total else "-"} for s in run.steps],
            "has_report": run.report is not None, "has_video": run.video is not None,
            "job": job.as_dict() if job else None,
        }

    # ------------------------------------------------------------- artefacts
    @app.get("/artefact/{kind}/{run_id}")
    def serve_artefact(kind: str, run_id: str, download: int = 0):
        if kind not in {"report", "video", "evidence"}:
            raise HTTPException(404, "No such artefact kind.")
        try:
            path = lib.artefact(runs_root, outputs_root, run_id, kind)
        except (lib.NotFound, lib.Unsafe) as exc:
            raise HTTPException(404, str(exc)) from exc
        media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        # FileResponse answers range requests, so the annotated video scrubs.
        return FileResponse(path, media_type=media,
                            filename=path.name if download else None)

    @app.get("/frame/{run_id}/{name}")
    def serve_frame(run_id: str, name: str):
        if "/" in name or "\\" in name or ".." in name or not name.endswith(".jpg"):
            raise HTTPException(404, "No such frame.")
        try:
            path = (lib.run_dir(runs_root, run_id) / "06_frames" / name).resolve()
        except (lib.NotFound, lib.Unsafe) as exc:
            raise HTTPException(404, str(exc)) from exc
        if not path.is_file() or runs_root.resolve() not in path.parents:
            raise HTTPException(404, "No such frame.")
        return FileResponse(path, media_type="image/jpeg")

    # --------------------------------------------------------------- new run
    @app.get("/new", response_class=HTMLResponse)
    def new_run_form(request: Request, error: str = ""):
        schema = json.loads((cfg.paths.schemas_dir / "session.schema.json")
                            .read_text(encoding="utf-8"))
        return page(request, "new.html.j2", videos=lib.videos_in(watch),
                    schema=schema, error=error,
                    plate_files=_plate_candidates(watch))

    @app.post("/new")
    async def start_runs(request: Request):
        form = await request.form()
        videos = [Path(v) for v in form.getlist("videos")]
        if not videos:
            return RedirectResponse("/new?error=Pick+at+least+one+video.", 303)
        try:
            session, echo = _session_from_form(form)
        except ValueError as exc:
            from urllib.parse import quote
            return RedirectResponse(f"/new?error={quote(str(exc)[:400])}", 303)

        started, refused = [], []
        for video in videos:
            if not video.is_file():
                refused.append(f"{video.name}: not a file any more")
                continue
            path = _write_session(sessions_dir, video, session)
            try:
                started.append(app.state.queue.submit(video, path).id)
            except ValueError as exc:
                refused.append(str(exc))
        return page(request, "started.html.j2", started=started, refused=refused,
                    echo=echo, jobs=app.state.queue.all())

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request):
        return page(request, "jobs.html.j2", jobs=app.state.queue.all())

    @app.get("/api/jobs")
    def jobs_api():
        return {"jobs": [j.as_dict() for j in app.state.queue.all()]}

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return JSONResponse({"ok": True, "detail": app.state.queue.cancel(job_id)})
        except LookupError:
            raise HTTPException(404, "No such job.") from None

    # --------------------------------------------------------------- compare
    @app.post("/compare")
    def run_compare(request: Request, runs_selected: list[str] = Form(default=[]),
                    sd: float = Form(default=2.0)):
        from archery.compare import CompareError, build
        from archery.compare_report import render
        ids = [lib.safe_run_id(r) for r in dict.fromkeys(runs_selected)]
        if not ids:
            return RedirectResponse("/?error=Tick+at+least+one+run.", 303)
        label = "_".join(i.split("__")[0] for i in ids)[:60] or "compare"
        try:
            payload = build(runs_root, ids, float(sd),
                            int(cfg.get("stats.min_shots_for_sd", 3)))
            target, evidence = render(payload, outputs_root,
                                      runs_root / "_compare" / label, label)
        except CompareError as exc:
            return page(request, "compare_error.html.j2", message=str(exc), ids=ids)
        return RedirectResponse(f"/compare/{target.name}", 303)

    @app.get("/compare/{name}")
    def serve_compare(name: str):
        if "/" in name or "\\" in name or ".." in name or not name.endswith(".html"):
            raise HTTPException(404, "No such comparison.")
        path = (outputs_root / name).resolve()
        if not path.is_file() or outputs_root.resolve() not in path.parents:
            raise HTTPException(404, "No such comparison.")
        return FileResponse(path, media_type="text/html")

    # ---------------------------------------------------------------- doctor
    @app.get("/doctor", response_class=HTMLResponse)
    def doctor(request: Request, llm: int = 0):
        from archery.cli import DOCTOR_NON_BLOCKING, doctor_rows
        rows = doctor_rows(cfg, with_llm=bool(llm))
        blocking = [n for n, ok, _ in rows if not ok and n not in DOCTOR_NON_BLOCKING]
        return page(request, "doctor.html.j2", rows=rows, blocking=blocking, llm=bool(llm))

    return app


# ------------------------------------------------------------------ helpers
def _plate_candidates(folder: Path | None) -> list[dict]:
    if not folder or not folder.is_dir():
        return []
    return [{"path": str(p), "name": p.name} for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xlsm"}]


def _session_from_form(form) -> tuple[dict, list[tuple[str, str]]]:
    """Build the session dict with the SAME parser the CLI uses, so the form and
    the command line cannot disagree about what 'Height: 170 cms' means."""
    from archery.profile_text import ProfileError, apply_to_session, parse

    session = {}
    for field in ("athlete_name", "bow_type", "discipline", "draw_hand", "camera_view",
                  "session_date", "notes"):
        value = (form.get(field) or "").strip()
        if value:
            session[field] = value
    for field in ("age", "number_of_shots_expected"):
        raw = (form.get(field) or "").strip()
        if raw:
            session[field] = int(float(raw))
    for field in ("camera_distance_m", "declared_frame_rate"):
        raw = (form.get(field) or "").strip()
        if raw:
            session[field] = float(raw)
    gender = (form.get("gender") or "").strip()
    if gender:
        session["gender"] = gender

    profile = (form.get("profile") or "").strip()
    if profile:
        try:
            session = apply_to_session(session, parse(profile))
        except ProfileError as exc:
            raise ValueError(" ".join(str(exc).split())) from exc

    plate = (form.get("force_plate") or "").strip()
    images = [i for i in form.getlist("plate_images") if i.strip()]
    if plate or images:
        physio = dict(session.get("physio") or {})
        physio["force_plate"] = {"file": plate or None, "images": images}
        session["physio"] = physio

    echo = _echo(session)
    return session, echo


NOT_PROVIDED = "NOT PROVIDED"


def _echo(session: dict) -> list[tuple[str, str]]:
    """What the CLI prints as INPUTS UNDERSTOOD, as rows for the page. Every
    absent field is named, because a silent default is how a wrong number
    reaches a report."""
    def dig(path):
        node = session
        for key in path.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        return node

    rows = []
    for label, path, unit in (
            ("Athlete", "athlete_name", ""), ("Age", "age", "years"),
            ("Height", "height_cm", "cm"), ("Weight", "weight_kg", "kg"),
            ("Bow type", "bow_type", ""), ("Draw hand", "draw_hand", ""),
            ("Camera view", "camera_view", ""), ("Session date", "session_date", ""),
            ("Resting HR", "physio.heart_rate.rest_bpm", "bpm"),
            ("Average HR", "physio.heart_rate.mean_bpm", "bpm"),
            ("Max HR", "physio.heart_rate.max_bpm", "bpm")):
        v = dig(path)
        rows.append((label, NOT_PROVIDED if v in (None, "", [])
                     else (f"{v:g} {unit}".strip() if isinstance(v, (int, float)) else str(v))))
    plate = dig("physio.force_plate.file")
    rows.append(("Force plate", Path(plate).name if plate else NOT_PROVIDED))
    imgs = dig("physio.force_plate.images") or []
    rows.append(("Plate images", ", ".join(Path(i).name for i in imgs) if imgs else NOT_PROVIDED))
    unparsed = session.get("inputs_unparsed") or []
    if unparsed:
        rows.append(("Unparsed, kept as notes", "; ".join(unparsed)))
    return rows


def _write_session(sessions_dir: Path, video: Path, session: dict) -> Path:
    """One session file per video, same location and shape the CLI expects, so
    the run is reproducible from the command line afterwards."""
    from archery.io_guard import guarded_open, guarded_path
    target = guarded_path(sessions_dir / f"{video.stem}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    merged = {**existing, **session}
    merged.setdefault("athlete_name", video.stem)
    with guarded_open(target, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    return target
