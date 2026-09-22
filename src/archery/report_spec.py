"""Report structure, section schemas, context builders and section self-checks.

The report's shape is fixed here, in code, so it cannot drift between runs:
section order, names, which sections are computed and which are written by the
model, what each model call may see, and the rules each output must satisfy.

Token economy: the model never receives the video, the landmark table or the
whole metrics file. Each call gets the shared rules, one section's instructions,
its schema, and an EVIDENCE list of only the keys that section needs (a few KB).
Later sections receive the earlier sections' JSON instead of raw metrics, which
also keeps the report internally consistent.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from archery.grounding import format_value, keys_in, walk_strings
from archery.phase_defs import DISPLAY, ORDER

NA = "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO"
LEVELS = ["OBSERVED", "MEASURED", "INTERPRETED", "INFERRED"]
CONF = ["HIGH", "MEDIUM", "LOW"]
SEV = ["CRITICAL", "MODERATE", "MINOR"]
CATEGORIES = [("A", "Alignment"), ("B", "Stability"), ("C", "Mobility"), ("D", "Strength"),
              ("E", "Coordination"), ("F", "Timing"), ("G", "Force Production"),
              ("H", "Force Transfer"), ("I", "Movement Efficiency"), ("J", "Repeatability"),
              ("K", "Equipment Interaction"), ("L", "Postural Control"), ("M", "Shot-Cycle Consistency")]

# ------------------------------------------------------------------ report order
MASTER = [("s01", "EXECUTIVE SUMMARY"), ("s02", "VIDEO & DATA QUALITY"),
          ("s03", "FRAME-BY-FRAME SHOT ANALYSIS"), ("s04", "SKILL-PHASE ANALYSIS (Extracted Phases)"),
          ("s05", "BIOMECHANICAL ANALYSIS"), ("s06", "CROSS-PHASE CONSISTENCY ANALYSIS (Extracted Phases)"),
          ("s07", "BIOMECHANICAL BENCHMARKS"), ("s08", "EQUIPMENT OBSERVATIONS"),
          ("s09", "TECHNICAL ERROR DETECTION"), ("s10", "INJURY RISK-FACTOR ASSESSMENT"),
          ("s11", "ARCHERY BIOMECHANICS ASSESSMENT FRAMEWORK"), ("s12", "PERFORMANCE SCORECARD"),
          ("s13", "TOP 10 STRENGTHS"), ("s14", "TOP 10 WEAKNESSES"),
          ("s15", "PRIORITIZED IMPROVEMENT PLAN"), ("s16", "TRAINING RECOMMENDATIONS"),
          ("s17", "COACHING PRIORITIES"), ("s18", "PERFORMANCE PROJECTION"),
          ("s19", "FINAL COACHING SUMMARY"), ("s20", "EVIDENCE & CONFIDENCE")]
PHYSIO = [("p1", "PHYSIOLOGICAL INTERVENTION & HEART RATE ANALYSIS"),
          ("p2", "INTEGRATED INTERVENTION (FORCE PLATE + HEART RATE)")]


def report_order(placement: str) -> list[dict]:
    """Numbered section list. 'inline' follows decision D10 (physio after 13)."""
    out = []
    if placement == "appendix":
        for i, (k, t) in enumerate(MASTER, 1):
            out.append({"key": k, "number": str(i), "title": t})
        for letter, (k, t) in zip("AB", PHYSIO):
            out.append({"key": k, "number": f"Appendix {letter}", "title": t})
        return out
    seq = MASTER[:13] + PHYSIO + MASTER[13:]
    return [{"key": k, "number": str(i), "title": t} for i, (k, t) in enumerate(seq, 1)]


# ------------------------------------------------------------------ schema helpers
STR = {"type": "string", "maxLength": 900}      # guard against run-on fields
INT = {"type": "integer"}
KEY_STR = {"type": "string", "pattern": "^[A-Za-z0-9_.\\-]+$", "maxLength": 120}


def enum(vals):
    return {"type": "string", "enum": list(vals)}


def arr(item, mn=0, mx=None):
    a = {"type": "array", "items": item, "minItems": mn}
    if mx is not None:
        a["maxItems"] = mx
    return a


def obj(**props):
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


KEYS = arr(KEY_STR, 0, 6)
SCORE = {"type": ["integer", "null"], "minimum": 0, "maximum": 10}
POINT = obj(point=STR, evidence_level=enum(LEVELS), confidence=enum(CONF), evidence_keys=KEYS)

# Only fields that hold ids, evidence keys or closed enums are exempt from
# grounding. Any free-text field is grounded, however short: a test showed a
# fabricated number slipping through a free-text "topic" field that had been
# exempted by mistake.
SKIP_FIELDS = {"evidence_level", "confidence", "severity", "classification", "phase", "category",
               "id", "ref", "weakness_ref", "priority", "evidence_keys", "addresses",
               "related_errors", "shot_phase", "timestamp_key"}
PRESCRIPTIVE = {"sets", "repetitions", "frequency", "progression", "technical_drill", "exercise",
                "coaching_cue", "success_metric", "recommendation", "corrective_strategy"}


def valid_ids(outputs: dict) -> dict[str, list[str]]:
    """Ids defined by earlier sections. Reference fields are restricted to these
    through schema enums, so the model cannot invent an id."""
    return {
        "strengths": [i["id"] for i in outputs.get("s13_strengths", {}).get("items", [])],
        "weaknesses": [i["id"] for i in outputs.get("s14_weaknesses", {}).get("items", [])],
        "errors": [e["id"] for e in outputs.get("s09_errors", {}).get("errors", [])],
    }


NOT_ESTABLISHED = "NOT ESTABLISHED"


def _ref_enum(values: list[str]) -> dict:
    return enum(list(dict.fromkeys(values + [NOT_ESTABLISHED])))


def schema_for(sid: str, ctx: dict) -> dict:
    phases = ctx.get("detected_phases", ORDER)
    ids = ctx.get("ids") or {"strengths": [], "weaknesses": [], "errors": []}
    # evidence_keys may only name keys that were in this call's EVIDENCE list:
    # a bare key, no braces. Enforced by Ollama's grammar through the enum.
    call_keys = ctx.get("evidence_keys")
    KEYS = arr(enum(call_keys), 0, 6) if call_keys else arr(KEY_STR, 0, 6)
    POINT = obj(point=STR, evidence_level=enum(LEVELS), confidence=enum(CONF), evidence_keys=KEYS)
    S_ENUM = _ref_enum(ids["strengths"])
    WE_ENUM = _ref_enum(ids["weaknesses"] + ids["errors"])
    E_ITEMS = arr(enum(ids["errors"])) if ids["errors"] else arr(STR, 0, 0)
    if sid == "s02_quality":
        return obj(camera_angle=STR, lighting=STR, athlete_visibility=STR, occlusion=STR,
                   motion_blur=STR, clothing_interference=STR, limitations=arr(STR, 1, 8))
    if sid == "s04_phase":
        return obj(phase=enum([ctx["phase"]]),
                   criteria_groups=arr(enum(list("ABCDEFGH")), 1, 8),
                   analysis=arr(POINT, 2, 12),
                   frame_rows=arr(obj(shot=INT, observation=STR, interpretation=STR,
                                      evidence_level=enum(LEVELS), confidence=enum(CONF)), 1, 20),
                   coaching_implication=STR)
    if sid == "s05_biomech":
        return obj(findings=arr(obj(topic=STR, finding=STR, evidence_level=enum(LEVELS),
                                    confidence=enum(CONF), evidence_keys=KEYS), 4, 18))
    if sid == "s06_consistency":
        return obj(effects=arr(obj(aspect=enum(["Accuracy", "Grouping", "Arrow flight", "Repeatability",
                                                "Pressure performance"]),
                                   explanation=STR, confidence=enum(CONF)), 5, 5),
                   notes=STR)
    if sid == "s08_equipment":
        aspects = ["Bow fit", "Draw length", "Arrow length", "Arrow spine indications",
                   "Stabilizer behavior", "String alignment", "Nocking-point clues",
                   "Grip configuration", "Bow torque", "Equipment-induced movement"]
        return obj(items=arr(obj(aspect=enum(aspects), classification=enum(["VISIBLE", "INFERRED", "NOT ASSESSABLE"]),
                                 observation=STR, confidence=enum(CONF)), 10, 10))
    if sid == "s09_errors":
        return obj(errors=arr(obj(id=STR, rank=INT, error=STR, phase=enum(phases), timestamp_key=STR,
                                  evidence=STR, cause=STR, biomechanical_effect=STR,
                                  performance_effect=STR, severity=enum(SEV), correction=STR,
                                  confidence=enum(CONF), evidence_keys=KEYS), 0, 12))
    if sid == "s10_injury":
        return obj(risks=arr(obj(factor=STR, evidence=STR, possible_concern=STR, severity=enum(SEV),
                                 corrective_strategy=STR, confidence=enum(CONF), evidence_keys=KEYS), 0, 12))
    if sid == "s11_framework":
        return obj(categories=arr(obj(category=enum([c for c, _ in CATEGORIES]), score=SCORE,
                                      evidence=STR, main_limitation=STR, recommendation=STR,
                                      confidence=enum(CONF), evidence_keys=KEYS), 13, 13))
    if sid == "s12_scorecard":
        sc = obj(score=SCORE, justification=STR, confidence=enum(CONF), evidence_keys=KEYS)
        return obj(phases=arr(obj(phase=enum(phases), score=SCORE, justification=STR,
                                  confidence=enum(CONF), evidence_keys=KEYS), len(phases), len(phases)),
                   shot_consistency=sc, biomechanical_efficiency=sc, movement_stability=sc,
                   overall_technique=sc)
    if sid == "s13_strengths":
        return obj(items=arr(obj(id=STR, strength=STR, evidence=STR, biomechanical_reason=STR,
                                 performance_benefit=STR, maintenance_strategy=STR,
                                 confidence=enum(CONF), evidence_keys=KEYS), 0, 10))
    if sid == "s14_weaknesses":
        return obj(items=arr(obj(id=STR, rank=INT, weakness=STR, evidence=STR, likely_cause=STR,
                                 performance_consequence=STR, correction=STR, priority=enum(CONF),
                                 confidence=enum(CONF), evidence_keys=KEYS, related_errors=E_ITEMS), 0, 10))
    if sid == "s15_priorities":
        return obj(priorities=arr(obj(rank=INT, weakness_ref=WE_ENUM, issue=STR, current_problem=STR,
                                      evidence=STR, timestamp_key=STR, shot_phase=enum(phases),
                                      why_it_matters=STR, biomechanical_consequence=STR,
                                      performance_consequence=STR, corrective_strategy=STR,
                                      technical_drill=STR, sets=STR, repetitions=STR, frequency=STR,
                                      progression=STR, success_metric=STR, confidence=enum(CONF)), 5, 5))
    if sid == "s16_training_coaching":
        rec = obj(purpose=STR, exercise=STR, sets=STR, repetitions=STR, frequency=STR,
                  coaching_cue=STR, progression=STR, addresses=arr(WE_ENUM, 1))
        cp = obj(recommendation=STR, addresses=arr(WE_ENUM, 1))
        return obj(technical=arr(rec, 0, 8), strength_conditioning=arr(rec, 0, 8),
                   mental=arr(rec, 0, 6), warm_up=arr(rec, 0, 6),
                   immediate=arr(cp, 1, 6), short_term=arr(cp, 1, 6), long_term=arr(cp, 1, 6))
    if sid == "s18_projection_final":
        s_item, w_item = obj(text=STR, ref=S_ENUM), obj(text=STR, ref=WE_ENUM)
        return obj(projection=arr(obj(aspect=enum(["Consistency", "Grouping", "Stability", "Repeatability",
                                                   "Movement efficiency"]),
                                      projection=STR, confidence=enum(CONF)), 5, 5),
                   does_well=arr(s_item, 3, 3), fix_first=arr(w_item, 3, 3),
                   single_correction=w_item, key_cue=STR, final_assessment=STR)
    if sid == "s01_executive":
        s_item, w_item = obj(text=STR, ref=S_ENUM), obj(text=STR, ref=WE_ENUM)
        return obj(technical_level=STR, strongest_characteristic=s_item,
                   most_important_weakness=w_item, main_consistency_limitation=STR,
                   main_biomechanical_limitation=STR, main_injury_risk_factor=STR,
                   most_important_correction=w_item, expected_improvement_opportunity=STR)
    raise KeyError(sid)


# ------------------------------------------------------------------ output template
def _example(node: dict, depth: int = 0):
    t = node.get("type")
    if "enum" in node:
        vals = node["enum"]
        return vals[0] if len(vals) == 1 else " | ".join(map(str, vals[:12])) + (" | ..." if len(vals) > 12 else "")
    if t == "object":
        return {k: _example(v, depth + 1) for k, v in node["properties"].items()}
    if t == "array":
        return [_example(node["items"], depth + 1)]
    if t == "integer" or t == ["integer", "null"]:
        rng = f"{node.get('minimum', '')}-{node.get('maximum', '')}".strip("-")
        return f"<integer{' ' + rng if rng else ''}{' or null' if isinstance(t, list) else ''}>"
    if node.get("pattern"):
        return "<evidence key, exactly as listed, WITHOUT braces>"
    return "<text>"


def _counts(node: dict, path: str = "") -> list[str]:
    out = []
    if node.get("type") == "object":
        for k, v in node["properties"].items():
            out += _counts(v, f"{path}.{k}" if path else k)
    elif node.get("type") == "array":
        mn, mx = node.get("minItems", 0), node.get("maxItems")
        if mn == mx:
            out.append(f"{path}: exactly {mn} item(s)")
        elif mn or mx is not None:
            out.append(f"{path}: {mn} to {mx if mx is not None else 'any'} items")
        out += _counts(node["items"], f"{path}[]")
    return out


def output_format(schema: dict) -> str:
    """Human-readable template of the required JSON, derived from the schema so
    the two can never disagree."""
    template = json.dumps(_example(schema), indent=1, ensure_ascii=False)
    lines = [
        "OUTPUT FORMAT",
        "Return ONE JSON object with exactly these fields and nothing else. No markdown,",
        "no comments, no text before or after it. Values separated by | are the only",
        "allowed choices; pick one.",
        template,
        "Item counts:",
        *[f"- {c}" for c in _counts(schema)],
        "Be concise: one or two sentences per text field. In text fields cite values as",
        "{{KEY}}. In evidence_keys arrays list bare keys WITHOUT braces, at most 6.",
        "Finish the JSON completely; a cut-off answer is rejected.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ call plan
# (section id, depends on earlier outputs, image phases to attach)
CALLS: list[tuple[str, list[str], list[str]]] = [
    ("s02_quality", [], ["STANCE", "AIM"]),
    ("s04_phase", [], []),                     # expanded: one call per detected phase
    ("s05_biomech", ["s04_phase"], []),
    ("s06_consistency", ["s04_phase"], []),
    ("s08_equipment", [], ["STANCE", "AIM", "RELEASE"]),
    ("s09_errors", ["s04_phase", "s05_biomech", "s06_consistency"], []),
    ("s10_injury", ["s04_phase", "s05_biomech", "s09_errors"], []),
    ("s11_framework", ["s05_biomech", "s06_consistency", "s09_errors", "s08_equipment"], []),
    ("s12_scorecard", ["s04_phase", "s09_errors", "s11_framework"], []),
    ("s13_strengths", ["s04_phase", "s05_biomech", "s11_framework", "s12_scorecard"], []),
    ("s14_weaknesses", ["s09_errors", "s10_injury", "s11_framework", "s12_scorecard"], []),
    ("s15_priorities", ["s09_errors", "s14_weaknesses"], []),
    ("s16_training_coaching", ["s09_errors", "s14_weaknesses", "s15_priorities"], []),
    ("s18_projection_final", ["s12_scorecard", "s13_strengths", "s14_weaknesses", "s15_priorities"], []),
    ("s01_executive", ["s06_consistency", "s10_injury", "s12_scorecard", "s13_strengths",
                       "s14_weaknesses", "s15_priorities", "s18_projection_final"], []),
]
USES_RUBRIC = {"s11_framework", "s12_scorecard"}


# ------------------------------------------------------------------ evidence selection
CORE = ["elbow_bow_deg", "elbow_draw_deg", "shoulder_bow_deg", "shoulder_draw_deg", "wrist_bow_deg",
        "wrist_draw_deg", "trunk_inclination_deg", "neck_inclination_deg", "head_tilt_deg",
        "shoulder_tilt_deg", "pelvic_tilt_deg", "shoulder_hip_separation_deg"]
LOWER = ["hip_left_deg", "hip_right_deg", "knee_left_deg", "knee_right_deg", "ankle_left_deg",
         "ankle_right_deg", "stance_width_norm"]
PHASE_MEASURES = {
    "STANCE": CORE + LOWER, "PRE_DRAW": CORE, "DRAW": CORE + ["anchor_distance_norm"],
    "ANCHOR": CORE + ["anchor_distance_norm"], "AIM": CORE + ["anchor_distance_norm", "com_speed_norm_s"],
    "EXPANSION": CORE + ["anchor_distance_norm"], "RELEASE": CORE + ["draw_wrist_speed_norm_s"],
    "FOLLOW_THROUGH": CORE + ["draw_wrist_speed_norm_s"], "RECOVERY": CORE,
}


def evidence_lines(evidence: dict, keys: list[str], max_lines: int = 90) -> str:
    """The EVIDENCE block, capped. Keys arrive in priority order (this section's
    own measures first, then keys cited by earlier sections), so the cap drops
    the least relevant. An over-long block was what starved the reply of context
    room on the first real run."""
    seen, lines = set(), []
    for k in keys:
        if k in evidence and k not in seen:
            seen.add(k)
            e = evidence[k]
            conf = f" [{e['confidence']}]" if e.get("confidence") else ""
            lines.append(f"{{{{{k}}}}} = {format_value(e)}{conf}")
        if len(lines) >= max_lines:
            break
    if len(lines) >= max_lines:
        lines.append(f"(evidence list capped at {max_lines} entries; the most relevant are shown)")
    return "\n".join(lines) if lines else "(no measured evidence available for this section)"


def phase_keys(metrics: dict, phase: str) -> list[str]:
    ev = metrics["evidence_index"]
    n = metrics["session"]["n_shots"]
    out = []
    for sh in range(1, n + 1):
        b = f"shot{sh}.{phase}"
        out += [f"{b}.start_t_s", f"{b}.end_t_s", f"{b}.duration_s"]
        if phase == "AIM":
            out += [f"{b}.com_sway_x.range", f"{b}.com_sway_y.range"]
    for m in PHASE_MEASURES.get(phase, CORE):
        if n >= metrics["cross_shot"]["min_shots_for_sd"]:
            out += [f"all.{phase}.{m}.mean", f"all.{phase}.{m}.sd", f"all.{phase}.{m}.cv_pct",
                    f"shot1.{phase}.{m}.at_key_frame"]
        else:
            for sh in range(1, n + 1):
                out += [f"shot{sh}.{phase}.{m}.at_key_frame", f"shot{sh}.{phase}.{m}.mean",
                        f"shot{sh}.{phase}.{m}.range"]
    return [k for k in out if k in ev]


def quality_keys(metrics: dict) -> list[str]:
    return [k for k in metrics["evidence_index"] if k.startswith(("session.", "quality."))]


def section_base_keys(sid: str, metrics: dict, detected: list[str]) -> list[str]:
    ev = metrics["evidence_index"]
    if sid == "s02_quality":
        return quality_keys(metrics)
    if sid == "s05_biomech":
        return quality_keys(metrics) + phase_keys(metrics, "AIM") + phase_keys(metrics, "STANCE")
    if sid == "s06_consistency":
        ks = [k for k in ev if k.startswith("all.") and (".duration_s." in k or any(
            f".{m}." in k for m in ("elbow_bow_deg", "elbow_draw_deg", "anchor_distance_norm",
                                    "trunk_inclination_deg")))]
        return ["session.n_shots"] + ks + [k for k in ev if k.endswith(".duration_s")]
    if sid == "s08_equipment":
        return [k for k in phase_keys(metrics, "AIM") + phase_keys(metrics, "RELEASE")
                if "elbow_bow" in k or "wrist_bow" in k or "draw_wrist_speed" in k]
    if sid in ("s09_errors", "s10_injury", "s11_framework"):
        return quality_keys(metrics) + phase_keys(metrics, "AIM")
    return quality_keys(metrics)


def _shrink(val, max_chars: int):
    """Earlier sections are passed on for consistency, not re-analysis: keep ids,
    enums and headline text, drop evidence_keys and clip long prose."""
    if isinstance(val, dict):
        return {k: _shrink(v, max_chars) for k, v in val.items() if k != "evidence_keys"}
    if isinstance(val, list):
        return [_shrink(v, max_chars) for v in val]
    if isinstance(val, str) and len(val) > max_chars:
        return val[:max_chars].rsplit(" ", 1)[0] + " ..."
    return val


def digest(outputs: dict, deps: list[str], max_items: int = 4,
           max_chars: int = 220) -> tuple[str, list[str]]:
    """Earlier sections as compact JSON, plus every evidence key they cite."""
    parts, keys = [], []
    for d in deps:
        val = outputs.get(d)
        if val is None:
            continue
        if d == "s04_phase":
            val = {ph: {"analysis": [a["point"] for a in o.get("analysis", [])[:max_items]],
                        "coaching_implication": o.get("coaching_implication")}
                   for ph, o in val.items()}
        val = _shrink(val, max_chars)
        parts.append(f"### {d}\n{json.dumps(val, separators=(',', ':'), ensure_ascii=False)}")
        for _, s in walk_strings(val):
            keys += keys_in(s)
            if re.fullmatch(r"(shot\d+|all|session|quality|benchmark)\.[\w.\-]+", s):
                keys.append(s)
    return "\n\n".join(parts), keys


def build_prompt(sid: str, cfg_prompts: dict, metrics: dict, outputs: dict, deps: list[str],
                 detected: list[str], phase: str | None, has_images: bool,
                 limits: dict | None = None) -> tuple[str, str, dict]:
    limits = limits or {}
    ev = metrics["evidence_index"]
    ctx = {"detected_phases": detected, "phase": phase, "ids": valid_ids(outputs)}
    base = phase_keys(metrics, phase) if sid == "s04_phase" else section_base_keys(sid, metrics, detected)
    prior, prior_keys = digest(outputs, deps, max_items=int(limits.get("max_prior_points", 4)),
                               max_chars=int(limits.get("max_prior_chars", 220)))
    call_keys = [k for k in dict.fromkeys(base + prior_keys) if k in ev][
        :int(limits.get("max_evidence_lines", 90))]
    ctx["evidence_keys"] = call_keys
    rules = cfg_prompts["sections"][sid]
    system = cfg_prompts["system"]
    if sid in USES_RUBRIC:
        system += "\n\n" + cfg_prompts["scoring_rubric"]
    header = [f"SECTION: {sid}" + (f" / PHASE: {phase}" if phase else "")]
    session = metrics["session"]
    brief = {
        "athlete": {k: session["athlete"].get(k) for k in ("athlete_name", "bow_type", "draw_hand", "camera_view")},
        "shots_analysed": session["n_shots"],
        "overall_analysis_confidence": metrics["data_quality"]["overall_analysis_confidence"],
        "detected_phases": detected,
        "phases_not_detected": sorted({f"{p['phase']}: {p['reason_not_detected']}" for s in metrics["phase_timeline"]
                                       for p in s["phases"] if not p["detected"]}),
        "cross_shot_status": metrics["cross_shot"].get("status") or "available",
    }
    if phase:
        detected_shots = [s["shot"] for s in metrics["phase_timeline"]
                          for p in s["phases"] if p["phase"] == phase and p["detected"]]
        brief["phase"] = {"code": phase, "name": DISPLAY[phase], "shots_with_phase": detected_shots}
    if sid == "s06_consistency":
        brief["consistency_rankings"] = metrics["consistency_rankings"]
    if any(i for i in ctx["ids"].values()):
        brief["valid_ids"] = ctx["ids"]
    user = "\n\n".join(filter(None, [
        "\n".join(header),
        "INSTRUCTIONS\n" + rules.strip(),
        "CONTEXT\n" + json.dumps(brief, ensure_ascii=False),
        ("EARLIER SECTIONS (cite their ids; do not contradict them)\n" + prior) if prior else "",
        "EVIDENCE (cite values ONLY as {{key}})\n"
        + evidence_lines(ev, call_keys, int(limits.get("max_evidence_lines", 90))),
        ("IMAGE: the attached annotated key frame(s) may be used only for OBSERVED qualitative "
         "points (grip, string contact, finger relaxation, occlusion, visible equipment). Numbers "
         "printed on the image must still be cited by key." if has_images else
         "NO IMAGE is available: anything that needs visual inspection is NOT RELIABLY ASSESSABLE "
         "FROM AVAILABLE VIDEO."),
    ]))
    schema = schema_for(sid, ctx)
    user += "\n\n" + output_format(schema)
    return system, user, schema


# ------------------------------------------------------------------ self-checks
def _ids(items, key="id"):
    return [i.get(key) for i in items]


def local_checks(sid: str, out: dict, detected: list[str], phase: str | None,
                 outputs: dict, shots_for_phase: list[int] | None = None,
                 banned: list[str] | None = None) -> list[str]:
    """Rules one section must satisfy on its own. Run by S8 before accepting an
    output, and again by S9."""
    v: list[str] = []

    def need_keys(items, label, score_field=None):
        for i, it in enumerate(items):
            if score_field and it.get(score_field) is None:
                continue
            if not it.get("evidence_keys"):
                v.append(f"{label}[{i}] has no evidence_keys")

    if sid == "s04_phase":
        got = sorted({r["shot"] for r in out["frame_rows"]})
        if shots_for_phase is not None and got != sorted(shots_for_phase):
            v.append(f"frame_rows cover shots {got}, expected {sorted(shots_for_phase)}")
        for i, p in enumerate(out["analysis"]):
            if p["evidence_level"] == "MEASURED" and not p["evidence_keys"]:
                v.append(f"analysis[{i}] is MEASURED but cites no evidence key")
    elif sid == "s09_errors":
        ranks = sorted(e["rank"] for e in out["errors"])
        if ranks != list(range(1, len(ranks) + 1)):
            v.append(f"error ranks must be 1..n with no gaps, got {ranks}")
        ids = _ids(out["errors"])
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r"E\d+", i or "") for i in ids):
            v.append(f"error ids must be unique E1, E2, ...; got {ids}")
        need_keys(out["errors"], "errors")
    elif sid == "s10_injury":
        need_keys(out["risks"], "risks")
        for path, text in walk_strings(out):
            low = text.lower()
            for b in banned or []:
                if re.search(rf"\b{re.escape(b)}\b", low):
                    v.append(f"{path}: diagnostic language '{b}' is not allowed")
    elif sid == "s11_framework":
        cats = sorted(c["category"] for c in out["categories"])
        if cats != [c for c, _ in CATEGORIES]:
            v.append(f"all 13 categories A-M exactly once; got {cats}")
        need_keys(out["categories"], "categories", "score")
    elif sid == "s12_scorecard":
        got = sorted(p["phase"] for p in out["phases"])
        if got != sorted(detected):
            v.append(f"scorecard must score exactly the detected phases {sorted(detected)}; got {got}")
        need_keys(out["phases"], "phases", "score")
        for k in ("shot_consistency", "biomechanical_efficiency", "movement_stability", "overall_technique"):
            if out[k]["score"] is not None and not out[k]["evidence_keys"]:
                v.append(f"{k} has a score but no evidence_keys")
    elif sid == "s13_strengths":
        ids = _ids(out["items"])
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r"S\d+", i or "") for i in ids):
            v.append(f"strength ids must be unique S1, S2, ...; got {ids}")
        need_keys(out["items"], "items")
    elif sid == "s14_weaknesses":
        ids = _ids(out["items"])
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r"W\d+", i or "") for i in ids):
            v.append(f"weakness ids must be unique W1, W2, ...; got {ids}")
        ranks = sorted(w["rank"] for w in out["items"])
        if ranks != list(range(1, len(ranks) + 1)):
            v.append(f"weakness ranks must be 1..n, got {ranks}")
        errs = set(_ids(outputs.get("s09_errors", {}).get("errors", [])))
        for w in out["items"]:
            bad = [e for e in w["related_errors"] if e not in errs]
            if bad:
                v.append(f"{w['id']} references unknown errors {bad}")
        need_keys(out["items"], "items")
    elif sid == "s15_priorities":
        ranks = sorted(p["rank"] for p in out["priorities"])
        if ranks != [1, 2, 3, 4, 5]:
            v.append(f"exactly five priorities ranked 1-5; got {ranks}")
        known = set(_ids(outputs.get("s14_weaknesses", {}).get("items", []))) | \
            set(_ids(outputs.get("s09_errors", {}).get("errors", [])))
        for p in out["priorities"]:
            if p["weakness_ref"] not in known:
                v.append(f"priority {p['rank']} references unknown id {p['weakness_ref']}")
    elif sid == "s16_training_coaching":
        known = set(_ids(outputs.get("s14_weaknesses", {}).get("items", []))) | \
            set(_ids(outputs.get("s09_errors", {}).get("errors", [])))
        for grp in ("technical", "strength_conditioning", "mental", "warm_up",
                    "immediate", "short_term", "long_term"):
            for i, r in enumerate(out[grp]):
                bad = [a for a in r["addresses"] if a not in known]
                if bad:
                    v.append(f"{grp}[{i}] addresses unknown ids {bad}")
    elif sid == "s18_projection_final":
        S = set(_ids(outputs.get("s13_strengths", {}).get("items", [])))
        WE = set(_ids(outputs.get("s14_weaknesses", {}).get("items", []))) | \
            set(_ids(outputs.get("s09_errors", {}).get("errors", [])))
        for i, d in enumerate(out["does_well"]):
            if d["ref"] == NOT_ESTABLISHED and len(S) < 3:
                continue
            if d["ref"] not in S:
                v.append(f"does_well[{i}] ref {d['ref']} is not a strength id")
        for i, d in enumerate(out["fix_first"]):
            if d["ref"] == NOT_ESTABLISHED and len(WE) < 3:
                continue
            if d["ref"] not in WE:
                v.append(f"fix_first[{i}] ref {d['ref']} is not a weakness/error id")
        if out["single_correction"]["ref"] not in WE:
            v.append("single_correction ref is not a weakness/error id")
    elif sid == "s01_executive":
        S = set(_ids(outputs.get("s13_strengths", {}).get("items", [])))
        WE = set(_ids(outputs.get("s14_weaknesses", {}).get("items", []))) | \
            set(_ids(outputs.get("s09_errors", {}).get("errors", [])))
        if out["strongest_characteristic"]["ref"] not in S:
            v.append("strongest_characteristic ref is not a strength id")
        for k in ("most_important_weakness", "most_important_correction"):
            if out[k]["ref"] not in WE:
                v.append(f"{k} ref is not a weakness/error id")
    return v
