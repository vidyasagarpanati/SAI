"""Run ids, and turning one into its directory. Shared by the web UI and the
delete command, so the rule for what is a safe id lives in one place."""
from __future__ import annotations

import re
from pathlib import Path

# A run id is "<video file name>__<hash>", and video names are whatever the
# camera or the coach called them: "Deepika (1).mov" is ordinary. So the rule
# names what is dangerous rather than what is familiar: path separators, the
# characters Windows reserves, control characters, and names the filesystem
# would quietly alter.
UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
MAX_RUN_ID = 200


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
