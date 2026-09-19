"""Step execution, skip logic and the upstream verification gate.

Both the direct CLI path and the LangGraph path go through ``execute_step`` so
their behaviour cannot diverge.
"""
from __future__ import annotations

import hashlib
import importlib
import traceback
from pathlib import Path

from archery.context import Context
from archery.contracts import StepFailed, StepResult, UpstreamFailed, dump_result
from archery.runstate import DONE, STEP_NAMES, STEP_ORDER, RunState

_MODULE_FOR = {sid: f"archery.steps.s{int(sid[1:]):02d}_{STEP_NAMES[sid]}" for sid in STEP_ORDER}


def load_step(step_id: str):
    module_name = _MODULE_FOR[step_id]
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise NotImplementedError(
                f"{step_id} ({STEP_NAMES[step_id]}) is not implemented yet "
                f"(expected module {module_name})."
            ) from exc
        raise


def _hash_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        path = Path(p)
        h.update(p.encode("utf-8"))
        if path.is_file():
            h.update(str(path.stat().st_size).encode())
            h.update(hashlib.sha256(path.read_bytes()).hexdigest().encode()
                     if path.stat().st_size < (8 << 20) else b"large")
        elif path.is_dir():
            files = sorted(path.rglob("*"))
            h.update(str(len(files)).encode())
            h.update(str(sum(f.stat().st_size for f in files if f.is_file())).encode())
    return h.hexdigest()


def compute_input_hash(state: RunState, step_id: str) -> str:
    """Invalidate a step when the config, the video, or any upstream output changes."""
    h = hashlib.sha256()
    h.update(str(state.data.get("config_hash")).encode())
    h.update(str(state.data.get("video_sha256")).encode())
    h.update(step_id.encode())
    for prior in STEP_ORDER[:STEP_ORDER.index(step_id)]:
        rec = state.data["steps"].get(prior, {})
        h.update(str(rec.get("output_hash")).encode())
    return h.hexdigest()[:16]


def execute_step(ctx: Context, step_id: str, force: bool = False) -> StepResult:
    state = ctx.state
    assert state is not None, "Context.state must be set before running steps."

    ok, why = state.upstream_ok(step_id)
    if not ok and step_id != "S0":
        raise UpstreamFailed(
            f"Refusing to run {step_id} ({STEP_NAMES[step_id]}): {why}. "
            f"Fix the upstream step, or rerun with --from that step."
        )

    input_hash = compute_input_hash(state, step_id)
    if not force and state.should_skip(step_id, input_hash):
        rec = state.record(step_id)
        result = StepResult(step=step_id, outputs=rec.get("outputs", {}),
                            stats=rec.get("stats", {}))
        result.check("skipped_unchanged", True,
                     "Inputs unchanged since the last successful run. Reusing outputs.")
        return result

    module = load_step(step_id)
    state.start(step_id, input_hash)
    try:
        result: StepResult = module.run(ctx)
    except Exception:
        state.fail(step_id, traceback.format_exc(limit=8))
        raise

    result.outputs = {k: str(v) for k, v in result.outputs.items()}
    state.record(step_id)["output_hash"] = _hash_files(result.outputs.values())
    state.finish(step_id, result)
    dump_result(result, ctx.run_dir / f"checks_{step_id}.json")

    if not result.passed:
        raise StepFailed(result)
    return result


def plan_steps(from_step: str | None, to_step: str | None, only: str | None) -> list[str]:
    if only:
        return [only.upper()]
    start = STEP_ORDER.index((from_step or "S0").upper())
    end = STEP_ORDER.index((to_step or STEP_ORDER[-1]).upper())
    if end < start:
        raise ValueError(f"--to {to_step} comes before --from {from_step}")
    return STEP_ORDER[start:end + 1]
