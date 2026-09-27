"""Command line interface.

    archery doctor
    archery init-session --video "D:\\archery\\kalpana_01.mp4" --ask
    archery init-session --video ... --profile "Height: 170 cms; Age: 16 years"
    archery run --video ... --force-plate "D:\\archery\\Body Sway.xlsx"
    archery run --video "D:\\archery\\kalpana_01.mp4"
    archery run --video ... --to S5            # stop after the evidence file
    archery run --video ... --only S6          # rerun one step
    archery run --video ... --from S8 --force  # redo the narrative
    archery status
    archery status <run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archery import io_guard
from archery.config import load_config, sha256_file
from archery.context import Context
from archery.contracts import StepFailed, UpstreamFailed
from archery.runstate import STEP_NAMES, STEP_ORDER, RunState


def _bootstrap(root: Path | None):
    cfg = load_config(root)
    io_guard.configure([
        cfg.paths.runs_dir,
        cfg.paths.outputs_dir,
        cfg.root / "sessions",
        cfg.paths.models_dir,
    ])
    return cfg


def _session_path(cfg, video: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return cfg.root / "sessions" / f"{video.stem}.json"


def _load_session(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"No session metadata at {path}.\n"
            f"Create one with:  archery init-session --video <path to video>\n"
            f"then fill in athlete_name, bow_type, draw_hand and camera_view."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _make_run_id(cfg, video: Path, digest: str) -> str:
    """One run directory per video. Config changes are handled per step by the
    cache keys in runner.STEP_DEPS, so tuning a threshold re-runs only the steps
    that read it, inside the same run."""
    return f"{video.stem}__{digest[:8]}"


# ------------------------------------------------------------------ input block
NOT_PROVIDED = "NOT PROVIDED"

_ASK_ORDER = [
    ("Name", "name"), ("Height", "height"), ("Weight", "weight"), ("Age", "age"),
    ("Resting HR", "resting hr"), ("Average HR", "average hr"), ("Max HR", "max hr"),
]


def add_input_args(sp, ask: bool) -> None:
    """The same input flags on init-session and on run. They only fill
    session.json; the file stays the single source of truth, because the resume
    cache keys on session fields and an input the cache cannot see would let a
    changed profile reuse a stale report."""
    g = sp.add_argument_group("athlete and physiological inputs")
    g.add_argument("--profile", metavar="TEXT",
                   help="Free-text block. Use ';' between fields, e.g. "
                        "\"Height: 170 cms; Weight: 60 Kgs; Age: 16 years; Resting HR: 60 bpm\"")
    g.add_argument("--profile-file", metavar="PATH",
                   help="The same block read from a file, one field per line. "
                        "Easier than quoting multi-line text in PowerShell.")
    g.add_argument("--force-plate", metavar="PATH",
                   help="Posturography export, .csv or .xlsx. Opened read only.")
    g.add_argument("--force-plate-image", metavar="PATH", action="append", default=[],
                   help="Plate report image, repeatable. Shown as a figure; "
                        "no number is ever read out of an image.")
    g.add_argument("--yes", "-y", action="store_true",
                   help="Skip the confirmation prompt. Use for unattended runs.")
    if ask:
        g.add_argument("--ask", action="store_true",
                       help="Prompt for each field instead of passing --profile.")


def _prompt_block() -> str:
    print("Enter the athlete inputs. Press Enter to leave a field out.\n")
    lines = []
    for label, _ in _ASK_ORDER:
        try:
            value = input(f"  {label}: ").strip()
        except EOFError:
            break
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _check_readable(path_str: str, what: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise SystemExit(f"{what} not found: {path}")
    try:
        with open(path, "rb") as fh:
            fh.read(1)
    except OSError as exc:
        raise SystemExit(f"{what} cannot be read: {path}\n  {exc}")
    return path.resolve()


def apply_input_args(session: dict, args) -> dict:
    """Fold the input flags into a copy of the session dict."""
    from archery.profile_text import ProfileError, apply_to_session, parse

    text = ""
    if getattr(args, "ask", False):
        text = _prompt_block()
    if getattr(args, "profile_file", None):
        text = _check_readable(args.profile_file, "Profile file").read_text(encoding="utf-8")
    if getattr(args, "profile", None):
        text = (text + "\n" + args.profile) if text else args.profile

    out = dict(session)
    if text.strip():
        try:
            out = apply_to_session(out, parse(text))
        except ProfileError as exc:
            raise SystemExit(str(exc))

    if getattr(args, "athlete", None):
        out["athlete_name"] = args.athlete

    if getattr(args, "force_plate", None) or getattr(args, "force_plate_image", None):
        physio = dict(out.get("physio") or {})
        plate = dict(physio.get("force_plate") or {})
        if getattr(args, "force_plate", None):
            plate["file"] = str(_check_readable(args.force_plate, "Force-plate export"))
        if getattr(args, "force_plate_image", None):
            plate["images"] = [str(_check_readable(i, "Force-plate image"))
                               for i in args.force_plate_image]
        physio["force_plate"] = plate
        out["physio"] = physio
    return out


def _dig(session: dict, path: str):
    node = session
    for key in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


_INVENTORY = [
    ("athlete_name", "athlete_name", ""),
    ("height_cm", "height_cm", "cm"),
    ("weight_kg", "weight_kg", "kg"),
    ("age_y", "age", "years"),
    ("hr_rest_bpm", "physio.heart_rate.rest_bpm", "bpm"),
    ("hr_mean_bpm", "physio.heart_rate.mean_bpm", "bpm"),
    ("hr_max_bpm", "physio.heart_rate.max_bpm", "bpm"),
]


def print_inputs(session: dict) -> None:
    """Say back exactly what was understood, and name what was not. A silent
    default is how a wrong number reaches a report."""
    print("INPUTS UNDERSTOOD")
    for label, path, unit in _INVENTORY:
        value = _dig(session, path)
        shown = NOT_PROVIDED if value in (None, "", []) else (
            f"{value:g} {unit}".strip() if isinstance(value, (int, float)) else str(value))
        print(f"  {label:<14} {shown}")

    plate = _dig(session, "physio.force_plate.file")
    print(f"  {'force_plate':<14} {Path(plate).name if plate else NOT_PROVIDED}")
    images = _dig(session, "physio.force_plate.images") or []
    print(f"  {'plate_images':<14} "
          f"{', '.join(Path(i).name for i in images) if images else NOT_PROVIDED}")

    unparsed = session.get("inputs_unparsed") or []
    if unparsed:
        print("UNPARSED, kept as notes and never used as evidence")
        for line in unparsed:
            print(f'  "{line}"')


def confirm(args) -> bool:
    if getattr(args, "yes", False) or not sys.stdin.isatty():
        return True
    try:
        return input("Proceed? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def cmd_init_session(args) -> int:
    cfg = _bootstrap(args.root)
    video = Path(args.video)
    target = _session_path(cfg, video, args.session)
    if target.exists() and not args.force:
        print(f"Already exists: {target}\nPass --force to overwrite.")
        return 1
    template = json.loads((cfg.root / "session.example.json").read_text(encoding="utf-8"))
    template["athlete_name"] = args.athlete or video.stem
    template = apply_input_args(template, args)

    print_inputs(template)
    if not confirm(args):
        print("Nothing written.")
        return 1

    with io_guard.guarded_open(target, "w", encoding="utf-8") as fh:
        json.dump(template, fh, indent=2)
    print(f"\nWrote {target}")
    print(f"Check bow_type, draw_hand and camera_view, then run:  "
          f"archery run --video \"{video}\"")
    return 0


def cmd_doctor(args) -> int:
    import shutil
    cfg = _bootstrap(args.root)
    rows: list[tuple[str, bool, str]] = []

    rows.append(("python", sys.version_info[:2] >= (3, 11) and sys.version_info[:2] < (3, 13),
                 f"{sys.version.split()[0]} (need 3.11 or 3.12 for mediapipe wheels)"))

    for mod in ["cv2", "numpy", "pandas", "pyarrow", "yaml",
                "jsonschema", "jinja2", "httpx", "langgraph", "openpyxl"]:
        try:
            __import__(mod)
            rows.append((mod, True, "installed"))
        except Exception as exc:  # noqa: BLE001
            rows.append((mod, False, f"missing: {exc}"))

    # MediaPipe runs in an isolated worker process (see archery/pose_worker.py),
    # so test it exactly that way: a fresh interpreter, nothing else loaded.
    import subprocess
    probe = subprocess.run([sys.executable, "-m", "archery.pose_worker", "--selftest"],
                           capture_output=True, text=True, timeout=180)
    if probe.returncode == 0:
        rows.append(("mediapipe worker", True, probe.stdout.strip().splitlines()[-1]))
    else:
        tail = (probe.stderr or probe.stdout).strip().splitlines()[-1:] or ["no output"]
        rows.append(("mediapipe worker", False, tail[0]))

    # Informational: does pyarrow-then-mediapipe fail on this machine? If so, the
    # isolation above is what keeps the pipeline working.
    clash = subprocess.run(
        [sys.executable, "-c", "import pyarrow; from mediapipe.tasks.python import vision"],
        capture_output=True, text=True, timeout=180)
    rows.append(("pyarrow+mediapipe", True,
                 "can share a process" if clash.returncode == 0 else
                 "CONFLICT when loaded in one process (expected on some Windows builds; "
                 "handled by running pose in an isolated worker)"))

    # Only one distribution may own site-packages/cv2.
    try:
        from importlib.metadata import distributions
        opencv_pkgs = sorted({
            d.metadata["Name"] for d in distributions()
            if (d.metadata["Name"] or "").lower().startswith("opencv")
        })
        rows.append(("opencv packages", len(opencv_pkgs) == 1,
                     f"{opencv_pkgs} (exactly one must be installed; mediapipe needs "
                     f"opencv-contrib-python)" if opencv_pkgs != ["opencv-contrib-python"]
                     else "opencv-contrib-python only, correct"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("opencv packages", False, f"could not enumerate: {exc}"))

    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.CDLL("vcruntime140_1.dll")
            rows.append(("vcruntime", True, "Visual C++ 2015-2022 x64 runtime present"))
        except OSError:
            rows.append(("vcruntime", False,
                         "vcruntime140_1.dll not loadable. Install the Microsoft Visual "
                         "C++ 2015-2022 x64 redistributable: "
                         "winget install Microsoft.VCRedist.2015+.x64"))

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    rows.append(("ffmpeg", bool(ffmpeg), ffmpeg or "not found"))
    rows.append(("ffprobe", bool(shutil.which("ffprobe")),
                 shutil.which("ffprobe") or "not found (OpenCV fallback will be used)"))

    model = cfg.paths.models_dir / cfg.get("paths.pose_model_file")
    rows.append(("pose model", model.is_file(),
                 str(model) if model.is_file() else f"missing: {model}"))

    base = cfg.get("llm.base_url")
    try:
        import httpx
        r = httpx.get(f"{base}/api/tags", timeout=5)
        tags = [m["name"] for m in r.json().get("models", [])]
        rows.append(("ollama", True, f"{base} -> {len(tags)} models: {', '.join(tags[:6])}"))
        configured = cfg.get("llm.model")
        rows.append(("llm.model", bool(configured) and configured in tags,
                     f"configured={configured!r}. Set config/pipeline.yaml llm.model to one of the tags above."))
    except Exception as exc:  # noqa: BLE001
        rows.append(("ollama", False, f"{base} unreachable: {exc}"))

    if getattr(args, "llm", False):
        # One tiny structured-output call through the same client S8 uses.
        from archery.llm import OllamaClient
        try:
            # Cache inside runs/ so the write guard allows it (a system temp dir
            # is outside the whitelist and is correctly refused).
            probe_dir = io_guard.guarded_path(cfg.paths.runs_dir / "_doctor_llm")
            probe_dir.mkdir(parents=True, exist_ok=True)
            client = OllamaClient(cfg, probe_dir)
            caps = client.capabilities()
            rows.append(("model capabilities", True, ", ".join(sorted(caps)) or "none reported"))
            rows.append(("model vision", "vision" in caps,
                         "key frames will be sent" if "vision" in caps else
                         "no vision: image-dependent items will be NOT RELIABLY ASSESSABLE"))
            import time as _t
            from archery.grounding import check_text, keys_in
            fake_ev = {"shot1.AIM.elbow_bow_deg.mean": {"value": 168.5, "units": "deg"},
                       "shot1.AIM.trunk_inclination_deg.mean": {"value": 2.1, "units": "deg"}}
            user = ("SECTION: doctor_probe\n\nINSTRUCTIONS\nWrite one sentence describing the bow "
                    "elbow and trunk at full draw, citing both values.\n\n"
                    "EVIDENCE (cite values ONLY as {{key}})\n"
                    "{{shot1.AIM.elbow_bow_deg.mean}} = 168.5 deg [HIGH]\n"
                    "{{shot1.AIM.trunk_inclination_deg.mean}} = 2.1 deg [HIGH]")
            t0 = _t.time()
            out = client.chat_json(
                cfg.prompts.get("system", "Reply with JSON only."), user,
                {"type": "object", "properties": {"sentence": {"type": "string"},
                 "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
                 "required": ["sentence", "confidence"], "additionalProperties": False})
            sentence = out.get("sentence", "")
            problems = check_text(sentence, fake_ev)
            ok = not problems and len(keys_in(sentence)) >= 1
            rows.append(("structured output", True,
                         f"valid JSON in {_t.time() - t0:.0f}s "
                         f"({client.usage['prompt_tokens']}+{client.usage['completion_tokens']} tokens)"))
            rows.append(("evidence citation", ok,
                         f"reply: {sentence!r}" + ("" if ok else f"  PROBLEMS: {problems or ['no placeholder used']}")))
        except Exception as exc:  # noqa: BLE001
            rows.append(("structured output", False, f"{type(exc).__name__}: {exc}"))

    width = max(len(name) for name, _, _ in rows)
    for name, ok, detail in rows:
        print(f"[{'ok ' if ok else 'XX '}] {name:<{width}}  {detail}")
    blocking = [n for n, ok, _ in rows if not ok and n not in ("ffprobe", "ollama", "llm.model",
                                                                   "model vision", "pyarrow+mediapipe")]
    if blocking:
        print(f"\nBlocking problems: {blocking}")
        print("Run scripts\\bootstrap.ps1 to fix the environment.")
        return 1
    print("\nEnvironment is ready for the deterministic steps (S0 to S7).")
    return 0


def cmd_run(args) -> int:
    from archery.graph import describe, run_graph
    from archery.runner import execute_step, plan_steps

    cfg = _bootstrap(args.root)
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    session_path = _session_path(cfg, video, args.session)
    session = _load_session(session_path)
    updated = apply_input_args(session, args)
    if updated != session:
        # Persist before running so the cache key and the report's provenance
        # both see exactly what was supplied on the command line.
        print_inputs(updated)
        if not confirm(args):
            print("Nothing written, nothing run.")
            return 1
        with io_guard.guarded_open(session_path, "w", encoding="utf-8") as fh:
            json.dump(updated, fh, indent=2)
        print(f"Updated {session_path}\n")
        session = updated

    digest = sha256_file(video)
    run_id = args.run_id or _make_run_id(cfg, video, digest)
    run_dir = cfg.paths.runs_dir / run_id
    io_guard.guarded_path(run_dir).mkdir(parents=True, exist_ok=True)

    state = RunState.load_or_create(
        run_dir, run_id=run_id, video_path=str(video),
        video_sha256=digest, config_hash=cfg.config_hash,
    )
    ctx = Context(cfg=cfg, run_id=run_id, run_dir=run_dir, video_path=video,
                  session=session, state=state)

    steps = plan_steps(args.from_step, args.to_step, args.only)
    print(f"run_id : {run_id}")
    print(f"video  : {video.name}  sha256 {digest[:16]}")
    print(f"config : {cfg.config_hash}")
    print(f"steps  : {describe(steps)}\n")

    import time as _t
    run_started = _t.time()

    def _summary():
        print("\n" + state.render_table())
        print(f"wall clock: {(_t.time() - run_started) / 60:.1f} min")

    if args.no_graph or args.only:
        for step_id in steps:
            try:
                result = execute_step(ctx, step_id, force=args.force)
            except (StepFailed, UpstreamFailed) as exc:
                print(f"\n{exc}")
                _summary()
                return 2
            except NotImplementedError as exc:
                print(f"\n{exc}")
                return 3
            print(result.summary())
            for c in result.warnings:
                print(f"    [warn] {c.name}: {c.detail}")
        _summary()
        print("\nDone.")
        return 0

    final = run_graph(ctx, steps, force=args.force)
    for step_id in final.get("completed", []):
        print(f"{step_id}: PASS")
    if final.get("failed"):
        print(f"\n{final['failed']} FAILED\n{final.get('error')}")
        _summary()
        return 2
    _summary()
    print("\nDone.")
    return 0


def cmd_status(args) -> int:
    cfg = _bootstrap(args.root)
    runs_dir = cfg.paths.runs_dir
    if args.run_id:
        candidates = [runs_dir / args.run_id]
    else:
        candidates = sorted((p for p in runs_dir.glob("*") if (p / "state.json").is_file()),
                            key=lambda p: (p / "state.json").stat().st_mtime, reverse=True)[:1]
    if not candidates or not (candidates[0] / "state.json").is_file():
        print(f"No runs found under {runs_dir}")
        return 1
    state = RunState.load_or_create(candidates[0])
    print(state.render_table())
    return 0


def cmd_diagnose(args) -> int:
    """Full native-dependency report. Paste the output when a DLL fails to load."""
    import subprocess
    cfg = _bootstrap(args.root)
    script = cfg.root / "scripts" / "diagnose_mediapipe.py"
    return subprocess.run([sys.executable, str(script)], cwd=cfg.root).returncode


def cmd_selftest(args) -> int:
    """Run the synthetic-archer test suite. No video and no model needed."""
    import subprocess
    cfg = _bootstrap(args.root)
    cmd = [sys.executable, "-m", "pytest", str(cfg.root / "tests"), "-q"]
    return subprocess.run(cmd, cwd=cfg.root).returncode


def cmd_steps(args) -> int:
    for s in STEP_ORDER:
        print(f"{s:<4} {STEP_NAMES[s]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="archery", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="Project root. Defaults to the repository containing this package.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="Check the environment.")
    d.add_argument("--llm", action="store_true",
                   help="Also make one small structured-output call to the configured Ollama model.")
    d.set_defaults(func=cmd_doctor)

    i = sub.add_parser("init-session", help="Write a session.json template for a video.")
    i.add_argument("--video", required=True)
    i.add_argument("--session")
    i.add_argument("--athlete")
    i.add_argument("--force", action="store_true")
    add_input_args(i, ask=True)
    i.set_defaults(func=cmd_init_session)

    r = sub.add_parser("run", help="Run the pipeline.")
    r.add_argument("--video", required=True)
    r.add_argument("--session", help="Path to session.json. Defaults to sessions/<video stem>.json")
    r.add_argument("--run-id")
    r.add_argument("--from", dest="from_step", choices=STEP_ORDER)
    r.add_argument("--to", dest="to_step", choices=STEP_ORDER)
    r.add_argument("--only", choices=STEP_ORDER, help="Run exactly one step.")
    r.add_argument("--force", action="store_true", help="Ignore the resume cache.")
    r.add_argument("--no-graph", action="store_true", help="Run steps directly, bypassing LangGraph.")
    r.add_argument("--athlete", help="Override the athlete name in the session file.")
    add_input_args(r, ask=False)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="Show the state of a run.")
    s.add_argument("run_id", nargs="?")
    s.set_defaults(func=cmd_status)

    g = sub.add_parser("diagnose", help="Full native-dependency report for DLL failures.")
    g.set_defaults(func=cmd_diagnose)

    t = sub.add_parser("selftest", help="Run the synthetic-archer tests. No video needed.")
    t.set_defaults(func=cmd_selftest)

    l = sub.add_parser("steps", help="List the pipeline steps.")
    l.set_defaults(func=cmd_steps)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
