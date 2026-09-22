"""S9 verify: the guardrail agent.

IN     05_metrics.json, 08_narrative/all_sections.json
DO     the master prompt's CONSISTENCY CONTROL checklist, executed mechanically,
       plus evidence grounding of every sentence and cross-section consistency.
       Sections that fail are sent back to the narrator with the violations,
       up to llm.max_retries_per_section rounds, then re-verified.
OUT    09_verification.json (checklist, per-section results, repairs made)
VERIFY every checklist item PASS. A failure blocks S10: no report is written
       from an unverified narrative.
"""
from __future__ import annotations

import jsonschema

from archery import grounding
from archery.context import Context
from archery.contracts import StepResult
from archery.narrator import Narrator
from archery.report_spec import (CALLS, MASTER, PRESCRIPTIVE, SKIP_FIELDS, local_checks,
                                 report_order, schema_for)


def _section_problems(nar: Narrator) -> dict[str, list[str]]:
    probs: dict[str, list[str]] = {}
    for sid, deps, _ in CALLS:
        if sid == "s04_phase":
            got = sorted(nar.outputs.get(sid, {}))
            if got != sorted(nar.detected):
                probs["s04_phase"] = [f"phases analysed {got} != detected {sorted(nar.detected)}"]
            for ph, out in nar.outputs.get(sid, {}).items():
                p = nar.check(sid, out, nar.schema(sid, ph), ph)
                if p:
                    probs[f"s04_phase/{ph}"] = p
            continue
        out = nar.outputs.get(sid)
        if out is None:
            probs[sid] = ["section missing"]
            continue
        p = nar.check(sid, out, nar.schema(sid), None)
        if p:
            probs[sid] = p
    return probs


def _cross_problems(nar: Narrator) -> dict[str, list[str]]:
    o = nar.outputs
    probs: dict[str, list[str]] = {}
    pri = sorted(o.get("s15_priorities", {}).get("priorities", []), key=lambda p: p["rank"])
    pri_refs = [p["weakness_ref"] for p in pri]
    top_w = [w["id"] for w in sorted(o.get("s14_weaknesses", {}).get("items", []), key=lambda w: w["rank"])]
    final = o.get("s18_projection_final", {})
    exe = o.get("s01_executive", {})
    if pri_refs and final:
        bad = [f["ref"] for f in final.get("fix_first", []) if f["ref"] not in pri_refs]
        if bad:
            probs.setdefault("s18_projection_final", []).append(
                f"3 THINGS TO FIX FIRST must come from the prioritized plan {pri_refs}; not in plan: {bad}")
        if final.get("single_correction", {}).get("ref") != pri_refs[0]:
            probs.setdefault("s18_projection_final", []).append(
                f"SINGLE MOST IMPORTANT CORRECTION must be priority #1 ({pri_refs[0]})")
    if exe and pri_refs:
        if exe["most_important_correction"]["ref"] != pri_refs[0]:
            probs.setdefault("s01_executive", []).append(
                f"most_important_correction must reference priority #1 ({pri_refs[0]})")
        allowed = {pri_refs[0]} | ({top_w[0]} if top_w else set())
        if exe["most_important_weakness"]["ref"] not in allowed:
            probs.setdefault("s01_executive", []).append(
                f"most_important_weakness must be the rank-1 weakness or priority #1 ({sorted(allowed)})")
    if top_w and pri_refs:
        missing_top = [w for w in top_w[:1] if w not in pri_refs]
        if missing_top:
            probs.setdefault("s15_priorities", []).append(
                f"the rank-1 weakness {missing_top[0]} must appear in the prioritized plan")
    return probs


def _checklist(nar: Narrator, sec: dict, cross: dict) -> list[dict]:
    m, o = nar.metrics, nar.outputs
    order = [s["key"] for s in report_order(nar.ctx.cfg.get("report.physio_placement", "inline"))]
    master_in_order = [k for k in order if k.startswith("s")] == [k for k, _ in MASTER]
    grounded = not any("evidence key" in p or "typed outside" in p for ps in sec.values() for p in ps)
    scored = [c.get("score") for c in o.get("s11_framework", {}).get("categories", [])] + \
             [p.get("score") for p in o.get("s12_scorecard", {}).get("phases", [])]
    injury_ok = not any("diagnostic language" in p for p in sec.get("s10_injury", []))
    rows = [
        ("All required sections are present.", all(k in o for k, _, _ in CALLS)),
        ("Section order is unchanged.", master_in_order),
        ("All extracted skill phases were assessed.",
         sorted(o.get("s04_phase", {})) == sorted(nar.detected) and "s12_scorecard" not in sec),
        ("Measurements are supported by visible/valid data.", grounded),
        ("No measurement has been fabricated.", grounded),
        ("All major findings have confidence ratings.", not any("schema" in p for ps in sec.values() for p in ps)),
        ("Technical errors are ranked.", "s09_errors" not in sec),
        ("Exactly five improvement priorities are provided.", "s15_priorities" not in sec),
        ("Scores use the 0-10 system.", all(s is None or 0 <= s <= 10 for s in scored)),
        ("Injury assessment is non-diagnostic.", injury_ok),
        ("Recommendations correspond to identified weaknesses.", "s16_training_coaching" not in sec),
        ("Equipment observations are labeled VISIBLE/INFERRED.", "s08_equipment" not in sec),
        ("Unsupported conclusions are marked NOT ASSESSABLE.", grounded),
        ("Executive summary matches the detailed findings.", "s01_executive" not in sec and "s01_executive" not in cross),
        ("Final coaching priorities match the ranked weaknesses.",
         "s18_projection_final" not in cross and "s15_priorities" not in cross),
        ("No required section has been omitted.", all(k in o for k, _, _ in CALLS)),
        ("Benchmarks: only cited entries used.", all(r.get("source") for r in m["benchmarks"]["rows"])),
    ]
    return [{"check": c, "status": "PASS" if ok else "FAIL"} for c, ok in rows]


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S9")
    nar = Narrator(ctx, llm=ctx.llm)
    nar.load_saved()
    repairs = []
    rounds = int(ctx.cfg.get("llm.max_retries_per_section", 2))
    deps_of = {sid: (deps, imgs) for sid, deps, imgs in CALLS}

    for rnd in range(rounds + 1):
        sec = _section_problems(nar)
        cross = _cross_problems(nar)
        combined: dict[str, list[str]] = {}
        for d in (sec, cross):
            for k, v in d.items():
                combined.setdefault(k, []).extend(v)
        if not combined or rnd == rounds:
            break
        for key, problems in combined.items():
            sid, _, ph = key.partition("/")
            deps, imgs = deps_of[sid]
            if sid == "s04_phase" and not ph:
                continue
            out, left, _ = nar.generate(sid, deps, imgs, phase=ph or None, extra_feedback=problems)
            if ph:
                nar.outputs[sid][ph] = out
                nar.save(f"s04_phase_{ph}", out)
            else:
                nar.outputs[sid] = out
                nar.save(sid, out)
            repairs.append({"round": rnd + 1, "section": key, "problems": problems[:10],
                            "resolved": not left})
        nar.save("all_sections", nar.outputs)

    checklist = _checklist(nar, sec, cross)
    payload = {"passed": not combined and all(r["status"] == "PASS" for r in checklist),
               "checklist": checklist, "section_problems": sec, "cross_section_problems": cross,
               "repairs": repairs, "detected_phases": nar.detected}
    out = ctx.write_json("09_verification.json", payload)

    for row in checklist:
        res.check(row["check"], row["status"] == "PASS",
                  "" if row["status"] == "PASS" else "See 09_verification.json for the offending text.")
    for key, problems in combined.items():
        res.check(f"section_{key}", False, "; ".join(problems[:5]))
    res.outputs["verification"] = str(out)
    res.outputs["narrative"] = str(nar.dir / "all_sections.json")
    res.stats = {"passed": payload["passed"], "repairs": len(repairs),
                 "checklist_pass": sum(r["status"] == "PASS" for r in checklist),
                 "checklist_total": len(checklist)}
    return res
