"""A single-worker job queue over the CLI.

Each job is a subprocess running `python -m archery.cli run ...`, not an
in-process call. Three reasons, all learned here. MediaPipe has crashed this
project's interpreter before and must not take the server with it. Cancel
becomes a kill. And the CLI stays the single entry point, so the web UI cannot
drift away from what the command line does.

One job at a time: the pipeline is CPU-bound on a machine that is already
loaded, and two runs of the same video would write into the same run
directory and fight over state.json.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"
LOG_LINES = 400          # kept in memory per job; the full log is on disk


@dataclass
class Job:
    id: str
    video: Path
    session_path: Path | None
    label: str
    status: str = QUEUED
    run_id: str | None = None
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=LOG_LINES))
    error: str | None = None

    @property
    def elapsed_s(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    def as_dict(self) -> dict:
        return {"id": self.id, "video": self.video.name, "label": self.label,
                "status": self.status, "run_id": self.run_id,
                "elapsed_s": round(self.elapsed_s, 1), "returncode": self.returncode,
                "error": self.error, "log": list(self.log)[-40:]}


class JobQueue:
    def __init__(self, root: Path, python: str | None = None):
        self.root = Path(root)
        self.python = python or sys.executable
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._pending: deque[str] = deque()
        self._current: Job | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ---------------------------------------------------------------- public
    def submit(self, video: Path, session_path: Path | None = None,
               label: str = "") -> Job:
        video = Path(video)
        with self._lock:
            # The same video twice would collide in one run directory, so the
            # second is refused rather than queued behind a copy of itself.
            for j in self._jobs.values():
                if j.status in (QUEUED, RUNNING) and j.video == video:
                    raise ValueError(
                        f"{video.name} is already {j.status.lower()}. Wait for it, or "
                        f"cancel it first.")
            job = Job(id=uuid.uuid4().hex[:12], video=video,
                      session_path=session_path, label=label or video.stem)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._pending.append(job.id)
        self._ensure_worker()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order)]

    def active(self) -> list[Job]:
        return [j for j in self.all() if j.status in (QUEUED, RUNNING)]

    def by_run_id(self, run_id: str) -> Job | None:
        return next((j for j in self.all() if j.run_id == run_id), None)

    def cancel(self, job_id: str) -> str:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise LookupError(job_id)
            if job.status == QUEUED:
                self._pending = deque(i for i in self._pending if i != job_id)
                job.status = CANCELLED
                return "removed from the queue"
            if job.status == RUNNING and self._proc and self._proc.poll() is None:
                self._kill(self._proc)
                job.status = CANCELLED
                return ("stopped. The run is resumable: every finished step keeps its "
                        "cached output, so starting it again picks up where it stopped")
            return f"already {job.status.lower()}"

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._kill(self._proc)

    # --------------------------------------------------------------- internal
    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                proc.wait(timeout=10)
            else:
                proc.terminate()
                proc.wait(timeout=10)
        except Exception:                      # noqa: BLE001
            proc.kill()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._loop, name="archery-jobs",
                                            daemon=True)
            self._worker.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job_id = self._pending.popleft() if self._pending else None
            if job_id is None:
                return
            job = self._jobs[job_id]
            if job.status == CANCELLED:
                continue
            self._run(job)

    def _command(self, job: Job) -> list[str]:
        cmd = [self.python, "-m", "archery.cli", "--root", str(self.root),
               "run", "--video", str(job.video), "--yes"]
        if job.session_path:
            cmd += ["--session", str(job.session_path)]
        return cmd

    def _run(self, job: Job) -> None:
        job.status, job.started = RUNNING, time.time()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        src = self.root / "src"
        if src.is_dir():
            env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
        try:
            proc = subprocess.Popen(
                self._command(job), cwd=str(self.root), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, **flags)
        except OSError as exc:
            job.status, job.error, job.finished = FAILED, str(exc), time.time()
            return
        with self._lock:
            self._current, self._proc = job, proc

        for line in proc.stdout:                       # type: ignore[union-attr]
            line = line.rstrip()
            job.log.append(line)
            if job.run_id is None and line.startswith("run_id"):
                job.run_id = line.split(":", 1)[-1].strip() or None
        proc.wait()

        job.returncode, job.finished = proc.returncode, time.time()
        if job.status != CANCELLED:
            job.status = DONE if proc.returncode == 0 else FAILED
            if proc.returncode != 0:
                tail = [ln for ln in list(job.log)[-25:] if ln.strip()]
                job.error = tail[-1] if tail else f"exit code {proc.returncode}"
        with self._lock:
            self._current, self._proc = None, None
