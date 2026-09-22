"""Step execution, skip logic and the upstream verification gate.

Both the direct CLI path and the LangGraph path go through ``execute_step`` so
their behaviour cannot diverge.
"""
from __future__ import annotations

import hashlib
import importlib
import time
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


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}h {s % 3600 // 60}m {s % 60}s" if s >= 3600 else (
        f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s")


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


# What each step actually consumes. A step's cache key is built ONLY from these,
# so editing phase_rules.yaml re-runs S4 onward but never pose estimation, and
# editing the athlete's name never re-decodes the video.
#   steps   : upstream steps whose OUTPUT hashes feed this step
#   config  : config slices (see Config.slice_hash)
#   session : session.json fields this step reads
STEP_DEPS: dict[str, dict[str, list[str]]] = {
    "S0": {"steps": [], "config": [], "session": ["*"]},
    "S1": {"steps": [], "config": ["frames"], "session": []},
    "S2": {"steps": ["S1"], "config": ["pose", "paths.pose_model_file",
                                       "quality_gates"], "session": []},
    "S3": {"steps": ["S1", "S2"], "config": ["quality_gates", "smoothing"],
           "session": ["draw_hand"]},
    "S4": {"steps": ["S1", "S3"], "config": ["phase_rules", "quality_gates"],
           "session": ["manual_phase_overrides", "number_of_shots_expected"]},
    "S5": {"steps": ["S0", "S2", "S3", "S4"], "config": ["stats", "benchmarks",
                                                        "quality_gates"], "session": []},
    "S6": {"steps": ["S1", "S3", "S4", "S5"], "config": ["render"], "session": []},
    "S7": {"steps": ["S0", "S1", "S3", "S4"], "config": ["video"], "session": []},
    "S8": {"steps": ["S5", "S6"], "config": ["llm", "prompts", "report"], "session": []},
    "S9": {"steps": ["S5", "S8"], "config": ["llm", "prompts", "benchmarks", "report"], "session": []},
    "S10": {"steps": ["S0", "S5", "S6", "S8", "S9"], "config": ["render", "report"], "session": ["*"]},
}


def compute_input_hash(ctx: Context, step_id: str) -> str:
    """Cache key for one step: the video, its config slice, the session fields it
    reads, and the output hashes of the upstream steps it consumes."""
    import json
    state = ctx.state
    deps = STEP_DEPS[step_id]
    h = hashlib.sha256()
    h.update(step_id.encode())
    h.update(str(state.data.get("video_sha256")).encode())
    h.update(ctx.cfg.slice_hash(deps["config"]).encode())
    if deps["session"] == ["*"]:
        session_part = ctx.session
    else:
        session_part = {k: ctx.session.get(k) for k in deps["session"]}
    h.update(json.dumps(session_part, sort_keys=True, default=str).encode())
    for up in deps["steps"]:
        h.update(str(state.data["steps"].get(up, {}).get("output_hash")).encode())
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

    input_hash = compute_input_hash(ctx, step_id)
    if not force and state.should_skip(step_id, input_hash):
        rec = state.record(step_id)
        result = StepResult(step=step_id, outputs=rec.get("outputs", {}),
                            stats=rec.get("stats", {}))
        result.check("skipped_unchanged", True,
                     "Inputs unchanged since the last successful run. Reusing outputs.")
        return result

    module = load_step(step_id)
    state.start(step_id, input_hash)
    print(f"\n== {step_id} {STEP_NAMES[step_id]}: started", flush=True)
    t0 = time.time()
    try:
        result: StepResult = module.run(ctx)
    except Exception:
        state.fail(step_id, traceback.format_exc(limit=8))
        print(f"== {step_id} {STEP_NAMES[step_id]}: FAILED after {_hms(time.time() - t0)}", flush=True)
        raise
    took = time.time() - t0

    result.outputs = {k: str(v) for k, v in result.outputs.items()}
    state.record(step_id)["output_hash"] = _hash_files(result.outputs.values())
    state.record(step_id)["duration_s"] = round(took, 1)
    state.finish(step_id, result)
    print(f"== {step_id} {STEP_NAMES[step_id]}: {'PASS' if result.passed else 'FAIL'} "
          f"in {_hms(took)}", flush=True)
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
