"""Deleting runs, recoverably. The only code in this project that removes files.

The pipeline still has no delete primitive (io_guard): nothing here is imported
by a pipeline step. A person deletes, from the runs page or `archery delete`.

A deleted run is MOVED, not erased, to runs/_deleted/<run id>__<when>/:

    run/         the whole run directory, state.json and all
    outputs/     the files in outputs/ that this run produced
    deleted.json what was moved and from where, so restore puts it back exactly

What a run produced is read from what the pipeline recorded, never guessed from
a name: every output path in state.json, plus every report whose manifest names
this run (earlier report versions are not in state.json). Anything outside the
runs and outputs folders is never touched, so the source video and its session
file survive every delete. Emptying the Deleted area is the one permanent step.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from archery.io_guard import guarded_open, guarded_path
from archery.run_ids import NotFound, Unsafe, run_dir, safe_run_id

TRASH = "_deleted"
RECORD = "deleted.json"


@dataclass
class Deleted:
    trash_id: str
    run_id: str
    athlete: str
    deleted_at: str
    files: list[str]
    path: Path


def trash_root(runs_root: Path) -> Path:
    return runs_root / TRASH


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def produced_outputs(runs_root: Path, outputs_root: Path, run_id: str) -> list[Path]:
    """Files in outputs/ this run produced, from state.json and report manifests."""
    found: dict[Path, None] = {}
    state = run_dir(runs_root, run_id) / "state.json"
    data = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
    for rec in (data.get("steps") or {}).values():
        for raw in (rec.get("outputs") or {}).values():
            p = Path(str(raw))
            if not p.is_file():
                p = outputs_root / p.name           # a moved project folder
            if p.is_file() and _inside(p, outputs_root):
                found[p.resolve()] = None
    for manifest in outputs_root.glob("*.manifest.json"):
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("run_id") != run_id:
                continue
        except (OSError, ValueError):
            continue
        report = outputs_root / manifest.name.replace(".manifest.json", ".html")
        for p in (manifest, report, report.with_suffix(".html.sha256")):
            if p.is_file():
                found[p.resolve()] = None
    # The checksum beside each recorded report travels with it.
    for p in list(found):
        side = p.with_suffix(p.suffix + ".sha256")
        if side.is_file():
            found[side.resolve()] = None
    return sorted(found)


def _move(src: Path, dst: Path) -> None:
    guarded_path(src)                      # moving removes the source: both ends guarded
    guarded_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _write_record(folder: Path, record: dict) -> None:
    with guarded_open(folder / RECORD, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)


def delete(runs_root: Path, outputs_root: Path, run_id: str,
           now: dt.datetime | None = None) -> Deleted:
    src = run_dir(runs_root, run_id)
    if not (src / "state.json").is_file() or _inside(src, trash_root(runs_root)):
        raise Unsafe(f"{run_id} is not a run (no state.json), so it is not deleted.")
    when = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    trash_id = f"{run_id}__{when}"
    folder = trash_root(runs_root) / trash_id
    if folder.exists():
        raise FileExistsError(f"{folder} already exists; try again in a second.")
    outputs = produced_outputs(runs_root, outputs_root, run_id)
    athlete = _athlete(src)
    record = {"run_id": run_id, "athlete": athlete,
              "deleted_at": dt.datetime.now().isoformat(timespec="seconds"),
              "run_from": str(src), "outputs": []}
    # The record is written before anything moves and updated as things do,
    # so an interrupted delete still says where every moved file came from.
    folder.mkdir(parents=True)
    _write_record(folder, record)
    _move(src, folder / "run")
    for p in outputs:
        _move(p, folder / "outputs" / p.name)
        record["outputs"].append({"name": p.name, "from": str(p)})
        _write_record(folder, record)
    return _load(folder)


def _athlete(path: Path) -> str:
    for name, dig in (("05_metrics.json", ("session", "athlete")), ("00_ingest.json", ("session",))):
        f = path / name
        if f.is_file():
            try:
                node = json.loads(f.read_text(encoding="utf-8"))
                for key in dig:
                    node = (node or {}).get(key) or {}
                return str(node.get("athlete_name") or "Unknown")
            except (OSError, ValueError):
                continue
    return "Unknown"


def _load(folder: Path) -> Deleted:
    rec = json.loads((folder / RECORD).read_text(encoding="utf-8"))
    return Deleted(trash_id=folder.name, run_id=rec["run_id"], athlete=rec.get("athlete", "Unknown"),
                   deleted_at=rec.get("deleted_at", ""),
                   files=[o["name"] for o in rec.get("outputs", [])], path=folder)


def listing(runs_root: Path) -> list[Deleted]:
    root = trash_root(runs_root)
    if not root.is_dir():
        return []
    out = []
    for folder in root.iterdir():
        if (folder / RECORD).is_file():
            try:
                out.append(_load(folder))
            except (OSError, ValueError, KeyError):
                continue
    return sorted(out, key=lambda d: d.deleted_at, reverse=True)


def _folder(runs_root: Path, trash_id: str) -> Path:
    folder = trash_root(runs_root) / safe_run_id(trash_id)
    if not (folder / RECORD).is_file() or not _inside(folder, trash_root(runs_root)):
        raise NotFound(f"Nothing called {trash_id!r} in Deleted.")
    return folder


def restore(runs_root: Path, outputs_root: Path, trash_id: str) -> str:
    """Put a deleted run back where it was. Refuses rather than overwrite."""
    folder = _folder(runs_root, trash_id)
    rec = json.loads((folder / RECORD).read_text(encoding="utf-8"))
    target = runs_root / safe_run_id(rec["run_id"])
    clashes = [str(target)] if target.exists() else []
    clashes += [str(outputs_root / o["name"]) for o in rec.get("outputs", [])
                if (outputs_root / o["name"]).exists()]
    if clashes:
        raise FileExistsError("Not restored, because these exist again: " + ", ".join(clashes)
                              + ". Delete or rename them first.")
    if (folder / "run").is_dir():
        _move(folder / "run", target)
    for o in rec.get("outputs", []):
        moved = folder / "outputs" / o["name"]
        if moved.is_file():
            _move(moved, outputs_root / o["name"])
    shutil.rmtree(guarded_path(folder))
    return rec["run_id"]


def purge(runs_root: Path, trash_id: str) -> None:
    """Erase one deleted run for good."""
    shutil.rmtree(guarded_path(_folder(runs_root, trash_id)))


def empty(runs_root: Path) -> int:
    items = listing(runs_root)
    for d in items:
        purge(runs_root, d.trash_id)
    return len(items)
