"""Reading the run library, and resolving a run id to the files it produced.

state.json is the whole data layer. Every step already records its status,
duration, output paths and every check with its detail, so the library row,
the progress table and the artefact links are reads of a file the pipeline
writes anyway. Nothing here is a second source of truth.

Path safety lives here too, in one place. The browser never sends a path: it
sends a run id, and this module resolves the path the PIPELINE recorded, then
refuses it unless it lands inside the run or output directories.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from archery.runstate import DONE, FAILED, PENDING, RUNNING, STEP_NAMES, STEP_ORDER

# A run id is "<video file name>__<hash>", and video names are whatever the
# camera or the coach called them: "Deepika (1).mov" is ordinary. So the rule
# names what is dangerous rather than what is familiar: path separators, the
# characters Windows reserves, control characters, and names the filesystem
# would quietly alter.
UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
MAX_RUN_ID = 200
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


class NotFound(LookupError):
    """No such run, or no such artefact in it."""


class Unsafe(ValueError):
    """A path that would leave the directories this server may serve."""


def safe_run_id(run_id: str) -> str:
    """A run id is a directory name, never a path. '..' and separators are the
    whole attack, so they are rejected before anything touches the disk."""
    if (not run_id or len(run_id) > MAX_RUN_ID or UNSAFE_CHARS.search(run_id)
            or run_id.startswith(".") or run_id != run_id.strip() or run_id.endswith(".")):
        raise Unsafe(f"Not a run id: {run_id!r}")
    return run_id


def run_dir(runs_root: Path, run_id: str) -> Path:
    path = (runs_root / safe_run_id(run_id)).resolve()
    if not path.is_dir() or runs_root.resolve() not in path.parents:
        raise NotFound(f"No run directory for {run_id!r}")
    return path


def _inside(path: Path, roots: list[Path]) -> bool:
    path = path.resolve()
    return any(root.resolve() == path or root.resolve() in path.parents for root in roots)


@dataclass
class StepRow:
    step: str
    name: str
    status: str
    duration_s: float | None
    checks_ok: int
    checks_total: int
    failures: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def took(self) -> str:
        d = self.duration_s
        if not d:
            return "-"
        return f"{d / 60:.1f} min" if d >= 60 else f"{d:.0f}s"


@dataclass
class Run:
    run_id: str
    path: Path
    athlete: str
    video_name: str
    session_date: str | None
    created: str | None
    status: str                 # DONE | RUNNING | FAILED | PENDING
    current_step: str | None
    steps: list[StepRow]
    total_s: float
    partial: bool
    report: Path | None
    video: Path | None
    evidence: Path | None
    n_shots: int | None

    @property
    def total(self) -> str:
        return f"{self.total_s / 60:.1f} min" if self.total_s >= 60 else f"{self.total_s:.0f}s"

    @property
    def done_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == DONE)


def _session_bits(path: Path) -> tuple[str, str | None, int | None]:
    """Athlete, session date and shot count, from whichever artefact has them.
    Preferred order is the richest file first, because a run that failed early
    still has 00_ingest.json."""
    for name, dig in (("05_metrics.json", ("session", "athlete")),
                      ("00_ingest.json", ("session",))):
        f = path / name
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        node = data
        for key in dig:
            node = (node or {}).get(key) or {}
        shots = ((data.get("session") or {}).get("n_shots")
                 if name == "05_metrics.json" else None)
        date = node.get("session_date")
        return (str(node.get("athlete_name") or "Unknown"),
                date if isinstance(date, str) and date[:4].isdigit() else None,
                shots)
    return "Unknown", None, None


def load_run(runs_root: Path, run_id: str, outputs_root: Path) -> Run:
    path = run_dir(runs_root, run_id)
    state_path = path / "state.json"
    if not state_path.is_file():
        raise NotFound(f"{run_id} has no state.json, so it is not a run.")
    data = json.loads(state_path.read_text(encoding="utf-8"))
    steps_data = data.get("steps") or {}

    steps, total = [], 0.0
    for sid in STEP_ORDER:
        rec = steps_data.get(sid, {})
        checks = rec.get("checks") or []
        total += rec.get("duration_s") or 0
        steps.append(StepRow(
            step=sid, name=STEP_NAMES[sid], status=rec.get("status", PENDING),
            duration_s=rec.get("duration_s"),
            checks_ok=sum(1 for c in checks if c.get("ok")), checks_total=len(checks),
            failures=[c for c in checks if not c.get("ok") and c.get("severity") == "FAIL"],
            warnings=[c for c in checks if not c.get("ok") and c.get("severity") == "WARN"],
            error=rec.get("error")))

    running = next((s.step for s in steps if s.status == RUNNING), None)
    terminal = next(s for s in steps if s.step == STEP_ORDER[-1])
    if running:
        status = RUNNING
    elif any(s.status == FAILED for s in steps):
        status = FAILED
    elif terminal.status == DONE:
        # The report exists. A skipped S7 does not make a finished run unfinished.
        status = DONE
    else:
        status = PENDING

    def output_of(step: str, key: str) -> Path | None:
        raw = ((steps_data.get(step) or {}).get("outputs") or {}).get(key)
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (runs_root.parent / candidate)
        # A project folder that was moved or renamed leaves a stale absolute
        # path in state.json; the file itself is still in the outputs folder.
        if not candidate.is_file():
            candidate = outputs_root / candidate.name
        # The pipeline recorded it, but the server still refuses anything that
        # does not land where this server is allowed to read.
        if not _inside(candidate, [runs_root, outputs_root]) or not candidate.is_file():
            return None
        return candidate.resolve()

    report = output_of("S10", "report")
    athlete, date, shots = _session_bits(path)
    return Run(
        run_id=run_id, path=path, athlete=athlete,
        video_name=Path(str(data.get("video_path") or "")).name or "unknown",
        session_date=date, created=data.get("created"), status=status,
        current_step=running, steps=steps, total_s=total,
        partial=bool(report and "_PARTIAL" in report.name),
        report=report, video=output_of("S7", "video"),
        evidence=(path / "05_metrics.json") if (path / "05_metrics.json").is_file() else None,
        n_shots=shots)


def list_runs(runs_root: Path, outputs_root: Path) -> list[Run]:
    """Every run directory, newest first. A directory that is not a run, or
    whose state.json is unreadable, is skipped rather than breaking the page.
    _compare and the _preview folders have no state.json, so no name is special."""
    out = []
    if not runs_root.is_dir():
        return out
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        try:
            out.append(load_run(runs_root, child.name, outputs_root))
        except (NotFound, Unsafe, ValueError, OSError):
            continue
    out.sort(key=lambda r: (r.created or "", r.run_id), reverse=True)
    return out


def artefact(runs_root: Path, outputs_root: Path, run_id: str, kind: str) -> Path:
    run = load_run(runs_root, run_id, outputs_root)
    path = {"report": run.report, "video": run.video, "evidence": run.evidence}.get(kind)
    if path is None:
        raise NotFound(f"{run_id} has no {kind}. The step that produces it has not "
                       f"completed, or its output is no longer where it was recorded.")
    return path


def videos_in(folder: Path) -> list[dict]:
    """The videos the new-run page may offer. Read only: nothing here opens,
    copies or moves a source file."""
    if not folder or not folder.is_dir():
        return []
    found = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            found.append({"path": str(p), "name": p.name,
                          "mb": round(p.stat().st_size / (1 << 20), 1)})
    return found
