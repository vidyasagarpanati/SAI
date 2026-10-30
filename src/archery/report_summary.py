"""The coach's layer of the report: a Coach's Page and one short summary per section.

Two sources, kept apart on purpose. Scores, counts, pills, tables and bars are
read straight from the verified narrative and metrics, so they cannot disagree
with the Full analysis beneath them. The plain-language wording comes from the
sum_* model calls, which pass the same grounding checks as every other section.
When a summary call was rejected, its sections fall back to text built from the
data, so a coach never loses a section to a wording failure.
"""
from __future__ import annotations

import re

from archery.report_spec import NOT_ESTABLISHED, SUMMARY_SECTIONS

DEG = re.compile(r"(-?\d+(?:\.\d+)?) ?deg\b")
SECONDS = re.compile(r"(?<![\w.])(\d+\.\d+) s\b")
WIDTHS = re.compile(r"(-?\d+\.\d+) SW\b")
CONF_WORDS = {"HIGH": "well", "MEDIUM": "fairly well", "LOW": "only a little"}


def plain_numbers(text: str) -> str:
    """Numbers as a coach says them. Exact values stay in the Full analysis."""
    def seconds(m):
        v = float(m.group(1))
        return f"{v:.0f} seconds" if v >= 10 else f"{max(round(v, 1), 0.1):g} seconds"
    def degrees(m):
        v = float(m.group(1))
        return f"{int(abs(v) + 0.5) * (1 if v >= 0 else -1)}°"    # 164.5 is 165, not 164
    text = DEG.sub(degrees, text)
    text = SECONDS.sub(seconds, text)
    return WIDTHS.sub(lambda m: f"{float(m.group(1)):.2f} shoulder widths", text)


def _first_sentence(text: str | None) -> str:
    return re.split(r"(?<=[a-z)])\. ", (text or "").strip())[0].rstrip(".") + "." if text else ""


def _chip(text: str, cls: str = "") -> dict:
    return {"text": text, "cls": cls}


def _ref(ref: str | None) -> dict | str:
    return {"ref": ref} if ref and ref != NOT_ESTABLISHED else ""


def _sev(s: str) -> dict:
    return _chip(s, f"sev-{s}")


def _conf(c: str | None) -> dict:
    return _chip(c or "", f"c-{c}")


def _entry(head: str, points=(), pill=None, **mini) -> dict:
    return {"head": head, "points": [p for p in points if p], "pill": pill, **mini}


def _wording(n: dict) -> tuple[dict, dict, dict]:
    """Merge the sum_* outputs: section text by key, phase takes, Coach's Page."""
    sections, takes, coach = {}, {}, {}
    for sid in SUMMARY_SECTIONS:
        out = n.get(sid) or {}
        for s in out.get("sections", []):
            sections[s["key"]] = {"head": plain_numbers(s["headline"]),
                                  "points": [plain_numbers(p) for p in s["points"]]}
        for p in out.get("phases", []):
            takes[p["phase"]] = plain_numbers(p["take"])
        for k, v in (out.get("coach") or {}).items():
            coach[k] = [plain_numbers(x) for x in v] if isinstance(v, list) else plain_numbers(v)
    return sections, takes, coach


def low_confidence_ids(n: dict, ev: dict) -> set[str]:
    """Findings resting on a LOW-confidence measurement. The finding's own rating
    can read MEDIUM while the number under it is LOW, and a coach acts on the number."""
    items = (n.get("s14_weaknesses", {}).get("items", [])
             + n.get("s09_errors", {}).get("errors", []))
    return {i["id"] for i in items
            if any(ev.get(k, {}).get("confidence") == "LOW" for k in i.get("evidence_keys", []))}


def _scored_phases(v: dict) -> list[dict]:
    return [p for p in v["n"].get("s12_scorecard", {}).get("phases", [])
            if p.get("score") is not None]


def _sections(v: dict, ev: dict, flagged: set[str], takes: dict) -> dict:
    n, out = v["n"], {}
    care = ("This rests on a low-confidence reading. Check it on the video before acting on it."
            if flagged else None)
    score = v.get("overall_score")

    out["info"] = _entry("", tiles=[(val, k) for k, val in v["info"] if k in
                                    ("Athlete", "Bow Type", "Number of Shots", "Video Duration",
                                     "Camera View") and val and "NOT PROVIDED" not in val])

    e = n.get("s01_executive", {})
    out["s01"] = _entry(_first_sentence((n.get("s18_projection_final") or {}).get("final_assessment")),
                        [e.get("strongest_characteristic", {}).get("text"),
                         e.get("most_important_weakness", {}).get("text")],
                        pill=(f"{score}/10", "") if score is not None else None)

    out["s02"] = _entry(f"Analysis confidence: {v['overall_conf']}.",
                        [dict(v["quality_rows"]).get("Pose-detection quality"),
                         dict(v["quality_rows"]).get("Measurement reliability")],
                        pill=(v["overall_conf"], f"conf-{v['overall_conf']}"))

    conf = [r["confidence"] for r in v["frame_rows"] if r.get("confidence")]
    tally = [(conf.count(c), CONF_WORDS[c]) for c in ("HIGH", "MEDIUM", "LOW") if conf.count(c)]
    if len(tally) == 1:
        trust = f"all of them {tally[0][1]}"
    else:
        trust = ", ".join(f"{count} of them {word}" for count, word in tally) or "none of them"
    steps = sum(1 for p in v["phases"] if p["detected"])
    out["s03"] = _entry(f"We split the shot into {steps} steps. "
                        f"We can trust {trust}.",
                        pill=(f"{steps} steps", ""), fixed_head=True)

    scored = _scored_phases(v)
    grid = []
    by_code = {p["phase"]: p for p in scored}
    for p in v["phases"]:
        sc = by_code.get(p["code"], {})
        grid.append({"code": p["code"], "name": p["name"], "detected": p["detected"],
                     "score": sc.get("score"), "thumb": p.get("thumb"),
                     "take": takes.get(p["code"]) or _first_sentence(sc.get("justification"))
                     or "Not found in this video."})
    head, pill = "", None
    if scored:
        low, high = min(scored, key=lambda p: p["score"]), max(scored, key=lambda p: p["score"])
        head = (f"Every step scored {low['score']}/10." if low["score"] == high["score"] else
                f"Best step: {high['phase_name']} ({high['score']}/10). "
                f"Weakest step: {low['phase_name']} ({low['score']}/10).")
        pill = (f"Lowest: {low['phase_name']} {low['score']}/10", "")
    out["s04"] = _entry(head, pill=pill, grid=grid, fixed_head=True)

    hard = [r["name"].replace(" angle", "") for r in v["angle_space_rows"] if r.get("flag")]
    out["s05"] = _entry("Measurements taken at full draw.", pill=(f"{len(v['stats_rows'])} measures", ""),
                        care=("Hard to read from this camera angle, so treat these with care: "
                              + ", ".join(hard) + ".") if hard else None)

    too_few = "NOT RELIABLY" in str(v["rankings"][0][1])
    out["s06"] = _entry("We need at least three shots to compare one shot with another."
                        if too_few else "How much the shot changes from one arrow to the next.",
                        pill=("Not assessable", "warn") if too_few else None)

    cited = sum(1 for r in v["bench_rows"] if r["diff"] != "—")
    out["s07"] = _entry(f"{cited} of {len(v['bench_rows'])} measures have a published reference to "
                        f"compare with." + (" These numbers are this athlete's own starting point."
                                            if not cited else ""),
                        pill=(f"{cited} of {len(v['bench_rows'])} cited", ""), fixed_head=True)

    items = n.get("s08_equipment", {}).get("items", [])
    by = {c: [i["aspect"] for i in items if i["classification"] == c]
          for c in ("VISIBLE", "INFERRED", "NOT ASSESSABLE")}
    out["s08"] = _entry("What we could and could not see of the equipment.",
                        [f"Seen in the video: {', '.join(by['VISIBLE'])}." if by["VISIBLE"] else "",
                         f"Worked out indirectly: {', '.join(by['INFERRED'])}." if by["INFERRED"] else "",
                         f"Cannot tell from this view: {', '.join(by['NOT ASSESSABLE'])}."
                         if by["NOT ASSESSABLE"] else ""],
                        pill=(f"{len(items) - len(by['NOT ASSESSABLE'])} of {len(items)} checked", "")
                        if items else None)

    errors = sorted(n.get("s09_errors", {}).get("errors", []), key=lambda x: x["rank"])
    sev_count = [f"{sum(x['severity'] == s for x in errors)} {s.lower()}"
                 for s in ("CRITICAL", "MODERATE", "MINOR") if any(x["severity"] == s for x in errors)]
    out["s09"] = _entry(f"{len(errors)} technique error{'s' if len(errors) != 1 else ''} found"
                        + (f": {', '.join(sev_count)}." if sev_count else "."),
                        pill=(f"{len(errors)} error{'s' if len(errors) != 1 else ''}", ""), fixed_head=True,
                        table={"head": ["", "Error", "Severity"],
                               "rows": [[_ref(x["id"]), x["error"], _sev(x["severity"])] for x in errors]},
                        care=care if {x["id"] for x in errors} & flagged else None)

    risks = n.get("s10_injury", {}).get("risks", [])
    out["s10"] = _entry("Movement habits to keep an eye on. This is not a medical diagnosis.",
                        pill=(f"{len(risks)} to watch", ""),
                        table={"head": ["What to watch", "Severity", "Confidence"],
                               "rows": [[r["factor"], _sev(r["severity"]), _conf(r["confidence"])]
                                        for r in risks]})

    done = [c for c in v["framework"] if c.get("score") is not None]
    skipped = [c["label"].split(". ", 1)[-1].lower() for c in v["framework"] if c.get("score") is None]
    out["s11"] = _entry(f"{len(done)} of {len(v['framework'])} areas could be scored from this video.",
                        pill=(f"{len(done)} of {len(v['framework'])} scored", ""), fixed_head=True,
                        bars=[{"label": c["label"].split(". ", 1)[-1], "score": c["score"]}
                              for c in sorted(done, key=lambda c: -c["score"])],
                        muted=f"Not scored: {', '.join(skipped)}." if skipped else None)

    sc = n.get("s12_scorecard", {})
    out["s12"] = _entry(f"Overall technique: {score}/10." if score is not None
                        else "No overall score could be given.",
                        pill=(f"{score}/10", "") if score is not None else None, fixed_head=True,
                        bars=[{"label": label, "score": (sc.get(key) or {}).get("score")}
                              for key, label in (("overall_technique", "Overall technique"),
                                                 ("biomechanical_efficiency", "Efficiency"),
                                                 ("movement_stability", "Stability"),
                                                 ("shot_consistency", "Shot consistency"))],
                        muted="Scores for each step are in the Skill-Phase section.")

    strengths = n.get("s13_strengths", {}).get("items", [])
    out["s13"] = _entry("What is working. Protect these while fixing the rest.",
                        pill=(f"{len(strengths)} found", ""),
                        table={"head": ["", "Strength", "Confidence"],
                               "rows": [[_ref(i["id"]), i["strength"], _conf(i["confidence"])]
                                        for i in strengths]})

    ph = v["physio"]
    out["p1"] = _entry("" if ph["p1"] else "No heart-rate information was given for this session.",
                       pill=None if ph["p1"] else ("No data", "mute"), dim=not ph["p1"])
    out["p2"] = _entry("" if ph["p2"] else "No force-plate information was given for this session.",
                       pill=None if ph["p2"] else ("No data", "mute"), dim=not ph["p2"])

    weak = sorted(n.get("s14_weaknesses", {}).get("items", []), key=lambda x: x["rank"])
    out["s14"] = _entry("What to work on, most important first.", pill=(f"{len(weak)} ranked", ""),
                        table={"head": ["", "Weakness", "Priority", "Confidence"],
                               "rows": [[_ref(i["id"]), i["weakness"], i["priority"], _conf(i["confidence"])]
                                        for i in weak]},
                        care=care if {i["id"] for i in weak} & flagged else None)

    plan = sorted(n.get("s15_priorities", {}).get("priorities", []), key=lambda x: x["rank"])
    out["s15"] = _entry("The drills, how much to do, and how to know it worked.",
                        pill=(f"{len(plan)} priorities", ""),
                        table={"head": ["#", "Drill", "How much", "Goal"],
                               "rows": [[str(p["rank"]), p["technical_drill"].split(":")[0],
                                         f"{p['sets']} × {p['repetitions']}, {p['frequency']}",
                                         plain_numbers(p["success_metric"])] for p in plan]})

    t = n.get("s16_training_coaching", {})
    groups = [("technical", "Technique"), ("strength_conditioning", "Strength"),
              ("mental", "Mental"), ("warm_up", "Warm-up")]
    total = sum(len(t.get(g, [])) for g, _ in groups)
    out["s16"] = _entry("Exercises that support the plan.", pill=(f"{total} exercises", ""),
                        fixed_points=[f"{label}: {', '.join(r['exercise'] for r in t[g])}."
                                      for g, label in groups if t.get(g)])

    out["s17"] = _entry("What to do now, soon and later.",
                        [t[g][0]["recommendation"] for g in ("immediate", "short_term", "long_term")
                         if t.get(g)])

    final = n.get("s18_projection_final", {})
    out["s18"] = _entry("What should improve if the plan is followed. No score gain is promised.")
    out["s19"] = _entry(_first_sentence(final.get("final_assessment")),
                        muted="The coaching cue and first fixes are on the Coach's Page.")

    counts = v["conf_counts"]
    passed = sum(c["status"] == "PASS" for c in v["checklist"])
    out["s20"] = _entry(f"{passed} of {len(v['checklist'])} automatic checks passed.",
                        pill=(f"{passed}/{len(v['checklist'])} checks",
                              "ok" if passed == len(v["checklist"]) else "warn"),
                        fixed_head=True, stack=[("HIGH", counts["HIGH"]), ("MEDIUM", counts["MEDIUM"]),
                                                ("LOW", counts["LOW"])])
    return out


def _coach_page(v: dict, flagged: set[str], plain: dict) -> dict | None:
    final = v["n"].get("s18_projection_final")
    if not final:
        return None

    def lines(field, wording):
        words = plain.get(wording) or []
        return [{"text": words[i] if i < len(words) else item["text"],
                 "ref": item["ref"] if item["ref"] != NOT_ESTABLISHED else None,
                 "low": item["ref"] in flagged} for i, item in enumerate(final[field])]

    limits = []
    shots = v["n_shots"]
    if shots < 3:
        limits.append(f"Based on {shots} shot{'s' if shots != 1 else ''}, so we cannot yet say "
                      f"how repeatable the shot is.")
    hard = [r["name"].replace(" angle", "") for r in v["angle_space_rows"] if r.get("flag")]
    if hard:
        limits.append("Hard to read from this camera angle: " + ", ".join(hard) + ".")
    scored = _scored_phases(v)
    low = min((p["score"] for p in scored), default=None)
    return {"score": v.get("overall_score"), "cue": final["key_cue"], "limits": limits,
            "does_well": lines("does_well", "does_well"), "fix_first": lines("fix_first", "fix_first"),
            "main_fix": {"text": plain.get("main_fix") or final["single_correction"]["text"],
                         "ref": final["single_correction"]["ref"]},
            "bars": [{"label": p["phase_name"], "score": p["score"], "low": p["score"] == low}
                     for p in scored],
            "confidence": v["overall_conf"]}


def build(v: dict, ev: dict) -> tuple[dict, dict | None]:
    """Section summaries by key, and the Coach's Page."""
    wording, takes, plain = _wording(v["n"])
    flagged = low_confidence_ids(v["n"], ev)
    sections = _sections(v, ev, flagged, takes)
    for key, entry in sections.items():
        said = wording.get(key)
        if key in v["failed"]:
            sections[key] = _entry("This section could not be produced. Open it to see why.",
                                   pill=("Not available", "warn"))
        elif said:
            if not entry.get("fixed_head"):
                entry["head"] = said["head"]
            entry["points"] = said["points"] or entry["points"]
        if entry.get("fixed_points"):
            entry["points"] = entry["fixed_points"]
    return sections, _coach_page(v, flagged, plain)
