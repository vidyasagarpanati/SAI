"""S10 render.

IN     00_ingest.json, 05_metrics.json, 06 key frames, 08_narrative,
       09_verification.json (must be PASS)
DO     substitute every evidence placeholder with its measured value, build the
       computed sections, re-render key frames with the verified coaching text,
       and write one self-contained HTML file (inline CSS, base64 images)
OUT    outputs/Archery_Report_<Athlete>_<YYYYMMDD>_vNN.html, .sha256, .manifest.json
VERIFY S9 passed; no external references; no unresolved placeholders; every
       section present in order; one image per detected phase per shot and one
       "[ANNOTATED FRAME NOT AVAILABLE]" box per undetected phase; HTML parses;
       the file is a new version, nothing overwritten
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re
from html.parser import HTMLParser

import cv2

from archery import __version__
from archery.context import Context
from archery.contracts import StepResult
from archery.framedata import FrameData
from archery.grounding import format_value, substitute, substitute_all
from archery.io_guard import guarded_open, guarded_path
from archery.phase_defs import BADGE, COLOR_BGR, DISPLAY, ORDER
from archery.report_spec import CATEGORIES, CORE, LOWER, report_order
from archery.steps.s06_annotate import render_key_frame
from archery.versioning import next_versioned, safe

NP = "NOT PROVIDED - CANNOT BE CONFIRMED"
NICE = {"elbow_bow_deg": "Bow elbow angle", "elbow_draw_deg": "Draw elbow angle",
        "shoulder_bow_deg": "Bow shoulder angle", "shoulder_draw_deg": "Draw shoulder angle",
        "wrist_bow_deg": "Bow wrist angle", "wrist_draw_deg": "Draw wrist angle",
        "trunk_inclination_deg": "Trunk inclination", "neck_inclination_deg": "Neck inclination",
        "head_tilt_deg": "Head tilt", "shoulder_tilt_deg": "Shoulder-girdle tilt",
        "pelvic_tilt_deg": "Pelvic tilt", "shoulder_hip_separation_deg": "Shoulder-hip separation",
        "hip_left_deg": "Hip angle (left)", "hip_right_deg": "Hip angle (right)",
        "knee_left_deg": "Knee angle (left)", "knee_right_deg": "Knee angle (right)",
        "ankle_left_deg": "Ankle angle (left)", "ankle_right_deg": "Ankle angle (right)",
        "stance_width_norm": "Stance width", "anchor_distance_norm": "Anchor distance"}
UNIT = {"deg": " deg", "shoulder widths": " SW", "shoulder widths/s": " SW/s"}


def _v(x, unit):
    return "—" if x is None else f"{x}{UNIT.get(unit, '')}"


def _b64(img, max_w: int, q: int) -> str:
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def build_view(ctx: Context, version_label: str) -> tuple[dict, int, int]:
    ingest = ctx.read_json("00_ingest.json")
    m = ctx.read_json("05_metrics.json")
    ver = ctx.read_json("09_verification.json")
    narr_raw = json.loads((ctx.narrative_dir / "all_sections.json").read_text(encoding="utf-8"))
    ev = m["evidence_index"]
    n = substitute_all(narr_raw, ev)
    sess, athlete = m["session"], m["session"]["athlete"]
    units = m["measure_units"]

    def a(key):
        val = athlete.get(key)
        return NP if val in (None, "", NP) else str(val)

    info = [("Athlete", a("athlete_name")), ("Archery Discipline", a("discipline")),
            ("Bow Type", a("bow_type")), ("Video Duration", f"{sess['duration_s']} s" if sess["duration_s"] else None),
            ("Video Frame Rate", f"{sess['measured_fps']} fps (measured); analysed at {sess['analysis_fps']:g} fps"),
            ("Camera View", a("camera_view")), ("Number of Shots", str(sess["n_shots"])),
            ("Analysis Method", sess["analysis_method"]), ("Pose Estimation Method", sess["pose_estimation"])]
    declared = athlete.get("declared_frame_rate")
    pre_basic = [("Name", a("athlete_name")), ("Age", a("age")), ("Gender (Male / Female)", a("gender")),
                 ("Bow Type (Recurve / Compound)", a("bow_type")), ("Number of Videos Provided", "Single Video"),
                 ("Any Additional Data Available?", "Yes" if athlete.get("additional_data") else "No")]
    pre_video = [("Video Frame Rate", f"{sess['measured_fps']} fps (measured)"
                  + (f"; declared {declared} fps" if declared else "")),
                 ("Number of Shots Captured", str(sess["n_shots"]))]

    q = m["data_quality"]
    quality_rows = [("Resolution", sess["resolution"]), ("Frame rate", f"{sess['measured_fps']} fps measured"),
                    ("Codec / metadata source", f"{sess['codec']} / {sess['metadata_source']}"),
                    ("Pose-detection quality", f"{q['pose_detection_rate_pct']}% of frames with the archer detected"),
                    ("Joint visibility", f"{q['mean_visible_landmarks']} of 33 landmarks visible on average; "
                                         f"{q['usable_frame_share_pct']}% of frames usable"),
                    ("Measurement reliability", f"worst core measure missing in {q['worst_core_measure_missing_pct']}% of frames"),
                    ("Draw-hand geometry check", (q.get("draw_hand_check") or {}).get("detail"))]

    fd = FrameData.load(ctx.run_dir)
    track = float(ctx.cfg.get("quality_gates.landmark_track_confidence", 0.5))
    max_w = int(ctx.cfg.get("render.max_embedded_frame_width", 1280))
    qual = int(ctx.cfg.get("render.jpeg_quality_embed", 85))
    s04 = n.get("s04_phase", {})
    phases, frame_rows, n_img, n_missing = [], [], 0, 0
    for code in ORDER:
        occurrences = [(sh, p) for sh in m["phase_timeline"] for p in sh["phases"] if p["phase"] == code]
        detected = [(sh, p) for sh, p in occurrences if p["detected"]]
        col = COLOR_BGR[code]
        entry = {"code": code, "badge": BADGE[code], "name": DISPLAY[code], "detected": bool(detected),
                 "color": f"rgb({col[2]},{col[1]},{col[0]})"}
        if not detected:
            entry["reason"] = next((p["reason_not_detected"] for _, p in occurrences), "not present in video")
            n_missing += 1
            phases.append(entry)
            continue
        out = s04.get(code, {})
        entry.update(analysis=out.get("analysis", []), coaching=out.get("coaching_implication"),
                     criteria=out.get("criteria_groups", []),
                     timing="; ".join(f"shot {sh['shot']}: {p['start_t_s']}-{p['end_t_s']} s "
                                      f"({p['duration_s']} s, {p['boundary_confidence']} boundary)"
                                      for sh, p in detected))
        frames = []
        for sh, p in detected:
            if p["key_frame"] is None:
                continue
            canvas, _, kf = render_key_frame(fd, ctx.draw_hand, sh["shot"], code,
                                             coaching=entry["coaching"], track_conf=track)
            frames.append({"shot": sh["shot"], "t": p["key_frame_t_s"], "frame": kf, "b64": _b64(canvas, max_w, qual)})
            n_img += 1
        entry["frames"] = frames
        phases.append(entry)
        rows_by_shot = {r["shot"]: r for r in out.get("frame_rows", [])}
        for sh, p in detected:
            r = rows_by_shot.get(sh["shot"], {})
            meas = m["per_shot"][sh["shot"] - 1]["phases"][code]["measures"]
            bits = [f"{lab} {_v(meas.get(k, {}).get('at_key_frame'), units.get(k))}"
                    for k, lab in (("elbow_bow_deg", "bow elbow"), ("elbow_draw_deg", "draw elbow"),
                                   ("trunk_inclination_deg", "trunk"))]
            frame_rows.append({"shot": sh["shot"], "ts": f"{p['start_t_s']}-{p['end_t_s']} s",
                               "phase": DISPLAY[code], "observation": r.get("observation"),
                               "measurement": "; ".join(bits), "interpretation": r.get("interpretation"),
                               "confidence": r.get("confidence"), "level": r.get("evidence_level")})
    frame_rows.sort(key=lambda r: (r["shot"], ORDER.index(next(c for c in ORDER if DISPLAY[c] == r["phase"]))))

    # Section 5: full-draw statistics
    stats_rows = []
    aim_cross = m["cross_shot"]["by_phase"].get("AIM", {})
    for k in CORE + LOWER:
        d = m["per_shot"][0]["phases"].get("AIM", {}).get("measures", {}).get(k)
        if not d:
            continue
        c = aim_cross.get(k, {})
        stats_rows.append({"name": NICE.get(k, k), "min": _v(d.get("min"), units[k]), "max": _v(d.get("max"), units[k]),
                           "mean": _v(d.get("mean"), units[k]), "range": _v(d.get("range"), units[k]),
                           "sd": _v(d.get("sd"), units[k]),
                           "xsd": _v(c.get("sd"), units[k]) if c.get("sd") is not None else (c.get("status") or "—"),
                           "conf": d.get("confidence")})

    timing_rows = []
    for code in ORDER:
        t = m["timing"].get(code)
        if not t:
            continue
        timing_rows.append({"phase": DISPLAY[code], "n": t["n_shots"], "values": ", ".join(map(str, t["values_s"])),
                            "mean": t.get("mean_s", "—"), "sd": t.get("sd_s", "—"),
                            "cv": f"{t['cv_pct']}%" if t.get("cv_pct") is not None else (t.get("status") and "n/a" or "—")})
    r = m["consistency_rankings"]
    if r.get("status"):
        rankings = [("Most / least consistent phase, largest variations", r["status"])]
    else:
        def lv(x, key, unit):
            return f"{DISPLAY.get(x['phase'], x['phase'])}: {NICE.get(x.get('measure'), x.get('measure', ''))} {x[key]}{unit}" if x else "—"
        rankings = [("Most consistent phase", DISPLAY.get(r.get("most_consistent_phase"), "—")),
                    ("Least consistent phase", DISPLAY.get(r.get("least_consistent_phase"), "—")),
                    ("Largest angular variation", lv(r.get("largest_angular_variation"), "sd_deg", " deg SD")),
                    ("Largest positional variation", lv(r.get("largest_positional_variation"), "sd_shoulder_widths", " SW SD")),
                    ("Largest timing variation", (f"{DISPLAY.get(r['largest_timing_variation']['phase'])}: "
                                                  f"{r['largest_timing_variation']['sd_s']} s SD")
                     if r.get("largest_timing_variation") else "—")]

    # Section 7: benchmarks (cited only)
    rows_by_measure = {row["measure"]: row for row in m["benchmarks"]["rows"]}
    bench_rows = []
    for k in CORE:
        d = m["per_shot"][0]["phases"].get("AIM", {}).get("measures", {}).get(k, {})
        c = aim_cross.get(k, {})
        athlete_val = c.get("mean") if c.get("mean") is not None else d.get("mean")
        row = rows_by_measure.get(k)
        if row:
            bench_rows.append({"variable": NICE.get(k, k) + " (full draw)", "athlete": _v(athlete_val, units[k]),
                               "reference": f"{row['target']} ({row['source']})", "diff": row["difference"],
                               "interp": f"Evidence class: {row['evidence_class']}", "conf": row["confidence"]})
        else:
            bench_rows.append({"variable": NICE.get(k, k) + " (full draw)", "athlete": _v(athlete_val, units[k]),
                               "reference": "Individualized assessment required", "diff": "—",
                               "interp": "Individualized assessment required", "conf": d.get("confidence")})
    bench_rejected = [f"{x['id']} ({x['reason']})" for x in m["benchmarks"]["rejected_entries"]]

    # Section 11 and 12 display names; Section 9/15 timestamps from keys
    cat_name = dict(CATEGORIES)
    framework = [dict(c, label=f"{c['category']}. {cat_name[c['category']]}")
                 for c in sorted(n["s11_framework"]["categories"], key=lambda c: c["category"])]
    for p in n["s12_scorecard"]["phases"]:
        p["phase_name"] = DISPLAY.get(p["phase"], p["phase"])
    for e in n["s09_errors"]["errors"]:
        e["timestamp"] = format_value(ev[e["timestamp_key"]]) if e["timestamp_key"] in ev else "—"
    for p in n["s15_priorities"]["priorities"]:
        p["timestamp"] = format_value(ev[p["timestamp_key"]]) if p["timestamp_key"] in ev else "—"
        p["shot_phase"] = DISPLAY.get(p["shot_phase"], p["shot_phase"])

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for c in re.findall(r'"confidence":\s*"(HIGH|MEDIUM|LOW)"', json.dumps(narr_raw)):
        counts[c] += 1

    physio = {}
    ph = athlete.get("physio") or {}
    if isinstance(ph, dict):
        if ph.get("heart_rate_file"):
            physio["p1"] = (f"Heart-rate file supplied ({ph['heart_rate_file']}) but no parser is configured "
                            f"for its format, so it was NOT ASSESSED. No values are reported.")
        if ph.get("force_plate_file"):
            physio["p2"] = (f"Force-plate file supplied ({ph['force_plate_file']}) but no parser is configured "
                            f"for its format, so it was NOT ASSESSED. No values are reported.")

    usage_path = ctx.narrative_dir / "usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.is_file() else {}
    provenance = [("Run id", ctx.run_id), ("Source video", sess["video_file"]),
                  ("Video SHA-256", sess["video_sha256"]), ("Config hash", m["config_hash"]),
                  ("Pipeline version", __version__), ("Narrative model", str(ctx.cfg.get("llm.model"))),
                  ("Model settings", f"temperature {ctx.cfg.get('llm.temperature')}, seed {ctx.cfg.get('llm.seed')}"),
                  ("Model calls / tokens", f"{usage.get('calls', 0)} calls ({usage.get('cached', 0)} cached), "
                                           f"{usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion tokens"),
                  ("Key frames sent to model", "yes" if usage.get("vision_used") else "no")]

    date = athlete.get("session_date") if isinstance(athlete.get("session_date"), str) and \
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", athlete.get("session_date") or "") else dt.date.today().isoformat()
    view = {
        "athlete_name": a("athlete_name"), "session_date": date, "version_label": version_label,
        "run_id": ctx.run_id, "order": report_order(ctx.cfg.get("report.physio_placement", "inline")),
        "info": info, "pre_basic": pre_basic, "pre_video": pre_video,
        "additional": athlete.get("additional_data") or [],
        "n": n, "overall_score": n["s12_scorecard"]["overall_technique"]["score"],
        "quality_rows": quality_rows, "overall_conf": q["overall_analysis_confidence"],
        "overall_rule": q["overall_confidence_rule"],
        "frame_rows": frame_rows, "phases": phases, "stats_rows": stats_rows,
        "timing_rows": timing_rows, "rankings": rankings, "bench_rows": bench_rows,
        "bench_rejected": bench_rejected, "framework": framework,
        "n_strengths": len(n["s13_strengths"]["items"]), "conf_counts": counts,
        "checklist": ver["checklist"], "physio": physio, "provenance": provenance,
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return view, n_img, n_missing


class _Parse(HTMLParser):
    def error(self, message):  # pragma: no cover
        raise ValueError(message)


def run(ctx: Context) -> StepResult:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    res = StepResult(step="S10")
    ver = ctx.read_json("09_verification.json")
    res.check("verification_passed", bool(ver.get("passed")),
              "S9 passed." if ver.get("passed") else "S9 did not pass; refusing to publish an unverified report.")
    if not ver.get("passed"):
        return res

    session = ctx.read_json("00_ingest.json")["session"]
    date = session.get("session_date") if isinstance(session.get("session_date"), str) and \
        len(session.get("session_date")) == 10 else dt.date.today().isoformat()
    stem = f"Archery_Report_{safe(session.get('athlete_name', 'Athlete'))}_{date.replace('-', '')}"
    target = next_versioned(guarded_path(ctx.cfg.paths.outputs_dir), stem, ".html")
    label = re.search(r"_v(\d+)\.html$", target.name).group(0)[1:-5]

    view, n_img, n_missing = build_view(ctx, label)
    env = Environment(loader=FileSystemLoader(str(ctx.cfg.paths.templates_dir)),
                      autoescape=select_autoescape(["html", "j2"]))
    html = env.get_template("report.html.j2").render(v=view)

    existed = target.exists()
    with guarded_open(target, "w", encoding="utf-8") as fh:
        fh.write(html)
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    with guarded_open(target.with_suffix(".html.sha256"), "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  {target.name}\n")
    manifest = {"report": target.name, "sha256": digest, "run_id": ctx.run_id,
                "video_sha256": ctx.read_json("00_ingest.json")["video_sha256"],
                "config_hash": ctx.cfg.config_hash, "model": ctx.cfg.get("llm.model"),
                "images": n_img, "frames_not_available": n_missing}
    with guarded_open(target.with_suffix(".manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    external = re.findall(r'(?:src|href)\s*=\s*["\'](?:https?:)?//|url\(\s*["\']?https?:|@import', html)
    res.check("self_contained", not external, f"External references found: {external[:5]}" if external
              else "No external CSS, scripts, fonts or images.")
    res.check("no_unresolved_placeholders", "{{" not in html, "All evidence placeholders substituted.")
    order_found = re.findall(r'<section id="sec-(\w+)"', html)
    expected = ["info"] + [s["key"] for s in view["order"]]
    res.check("sections_present_in_order", order_found == expected,
              f"found {order_found}" if order_found != expected else f"{len(order_found)} sections in the fixed order.")
    res.check("one_image_per_detected_phase_per_shot", html.count("data:image/jpeg;base64,") == n_img,
              f"{n_img} annotated frames embedded")
    res.check("placeholder_box_per_undetected_phase", html.count("[ANNOTATED FRAME NOT AVAILABLE]") == n_missing,
              f"{n_missing} phase(s) without a frame")
    try:
        _Parse().feed(html)
        parsed = True
    except Exception:  # noqa: BLE001
        parsed = False
    res.check("html_parses", parsed, "HTML parsed cleanly.")
    res.check("new_version_not_overwrite", not existed, f"Wrote {target.name}")

    res.outputs["report"] = str(target)
    res.outputs["sha256"] = str(target.with_suffix(".html.sha256"))
    res.stats = {"report": target.name, "size_mb": round(len(html) / 1e6, 2), "images": n_img}
    return res
