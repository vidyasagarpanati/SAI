"""SP  physio ingest.

IN     session.json: height_cm, weight_kg, age, physio.heart_rate.*,
       physio.force_plate.file and .images
DO     parse the posturography export for THIS athlete, compute BMI and the
       age-predicted heart-rate zones from cited formulas, and build the
       posture.*, hr.* and athlete.* evidence keys
OUT    00b_physio.json
VERIFY every emitted key has a value and a unit; every absent input is named;
       no measure the instrument did not record is emitted as a number

Why this is its own step. Supplying a corrected export or a fixed typo must not
re-run pose estimation or the annotated video, which cost minutes. SP sits
between S0 and S1 with its own cache key, so a changed input re-runs SP, S5,
S8, S9 and S10 and nothing else.

Nothing here is inferred. An absent input produces no key at all, so the report
prints NOT PROVIDED rather than reasoning from a default.
"""
from __future__ import annotations

from pathlib import Path

from archery.context import Context
from archery.contracts import FAIL, WARN, StepResult
from archery.forceplate import ForcePlateError, MEASURES, load

NOT_PROVIDED = "NOT PROVIDED"


def _r(value, places: int):
    return None if value is None else round(float(value), places)


def _dig(node, path: str):
    for key in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sd(xs: list[float]) -> float:
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def run(ctx: Context) -> StepResult:
    res = StepResult(step="SP")
    session = ctx.session
    bench = (ctx.cfg.benchmarks or {}).get("physiological", {})

    evidence: dict[str, dict] = {}
    inventory: list[dict] = []

    def ev(key: str, value, units: str, confidence: str, n: int | None = None):
        if value is None:
            return
        evidence[key] = {"value": value, "units": units, "confidence": confidence, "n": n}

    def note(label: str, supplied: bool, detail: str = ""):
        inventory.append({"input": label, "status": "SUPPLIED" if supplied else NOT_PROVIDED,
                          "detail": detail})

    # ---- anthropometrics ---------------------------------------------------
    height_cm = session.get("height_cm")
    weight_kg = session.get("weight_kg")
    age = session.get("age")
    age = age if isinstance(age, (int, float)) else None

    note("height", height_cm is not None, f"{height_cm} cm" if height_cm else "")
    note("weight", weight_kg is not None, f"{weight_kg} kg" if weight_kg else "")
    note("age", age is not None, f"{age} years" if age else "")

    ev("athlete.height_cm", _r(height_cm, 1), "cm", "HIGH")
    ev("athlete.weight_kg", _r(weight_kg, 1), "kg", "HIGH")
    ev("athlete.age_y", age, "years", "HIGH")

    bmi = None
    bmi_spec = bench.get("bmi", {})
    if height_cm and weight_kg:
        bmi = _r(weight_kg / ((height_cm / 100.0) ** 2), 1)
        ev("athlete.bmi", bmi, "kg/m^2", "HIGH")

    youth_spec = bench.get("youth_age_threshold_years", {})
    youth_threshold = youth_spec.get("value")
    is_youth = bool(age is not None and youth_threshold is not None and age < youth_threshold)

    # ---- heart rate --------------------------------------------------------
    hr = _dig(session, "physio.heart_rate") or {}
    hr_rest, hr_mean, hr_max_obs = hr.get("rest_bpm"), hr.get("mean_bpm"), hr.get("max_bpm")
    note("resting heart rate", hr_rest is not None, f"{hr_rest} bpm" if hr_rest else "")
    note("average heart rate", hr_mean is not None, f"{hr_mean} bpm" if hr_mean else "")
    note("maximum heart rate observed", hr_max_obs is not None,
         f"{hr_max_obs} bpm" if hr_max_obs else "")

    ev("hr.rest_bpm", _r(hr_rest, 0), "bpm", "HIGH")
    ev("hr.mean_bpm", _r(hr_mean, 0), "bpm", "HIGH")
    ev("hr.max_observed_bpm", _r(hr_max_obs, 0), "bpm", "HIGH")

    zones: list[dict] = []
    hr_max_est = None
    max_spec = bench.get("max_heart_rate", {})
    zone_spec = bench.get("zones_pct_hrmax", {})
    if age is not None and max_spec.get("formula"):
        hr_max_est = _r(208.0 - 0.7 * float(age), 0)
        # MEDIUM, never HIGH: a population formula, not this athlete's measurement.
        ev("hr.max_estimated_bpm", hr_max_est, "bpm", "MEDIUM")
        for band in zone_spec.get("bands", []):
            low = _r(hr_max_est * band["low_pct"] / 100.0, 0)
            high = _r(hr_max_est * band["high_pct"] / 100.0, 0)
            zones.append({**band, "low_bpm": low, "high_bpm": high})
            ev(f"hr.zone.{band['zone']}.low_bpm", low, "bpm", "MEDIUM")
            ev(f"hr.zone.{band['zone']}.high_bpm", high, "bpm", "MEDIUM")
        if hr_mean is not None:
            in_zone = [z["zone"] for z in zones if z["low_bpm"] <= hr_mean < z["high_bpm"]]
            if in_zone:
                ev("hr.mean_zone", in_zone[0], "", "MEDIUM")
    if hr_max_obs is not None and hr_max_est is not None and hr_max_obs > hr_max_est:
        res.check("hr_observed_within_estimate", True,
                  f"Observed maximum {hr_max_obs:g} bpm exceeds the {hr_max_est:g} bpm "
                  f"age-predicted estimate. Expected for a population formula; "
                  f"the report says so rather than treating either as a limit.",
                  severity=WARN)

    # ---- force plate -------------------------------------------------------
    plate_file = _dig(session, "physio.force_plate.file")
    images = _dig(session, "physio.force_plate.images") or []
    plate_payload: dict = {}
    min_for_sd = int(ctx.cfg.get("stats.min_shots_for_sd", 3))

    note("force-plate images", bool(images),
         ", ".join(Path(i).name for i in images) + " (figures only, no numbers read)"
         if images else "")

    if not plate_file:
        note("force-plate export", False)
        res.check("force_plate_parsed", True,
                  "No force-plate export supplied. Section p2 will render NOT PROVIDED.")
    else:
        path = Path(plate_file)
        athlete_name = session.get("athlete_name") or ""
        try:
            export = load(path, athlete_name)
        except ForcePlateError as exc:
            res.check("force_plate_parsed", False, str(exc))
            ctx.write_json("00b_physio.json", {"schema_version": 1, "error": str(exc),
                                               "inventory": inventory, "evidence": evidence})
            return res

        by_condition: dict[str, list] = {}
        for trial in export.trials:
            by_condition.setdefault(trial.condition, []).append(trial)

        conditions = []
        for label, trials in sorted(by_condition.items()):
            row = {"condition": label, "n_trials": len(trials),
                   "definition": trials[0].condition_full, "measures": {}}
            for key, (units, _column) in MEASURES.items():
                xs = [t.values[key] for t in trials if key in t.values]
                if not xs:
                    continue
                mean = _r(_mean(xs), 3)
                sd = _r(_sd(xs), 3) if len(xs) >= min_for_sd else None
                # A single trial is a reading, not a repeatable measure.
                conf = "HIGH" if len(xs) >= 2 else "MEDIUM"
                row["measures"][key] = {"mean": mean, "sd": sd, "n": len(xs), "units": units}
                ev(f"posture.{label}.{key}.mean", mean, units, conf, len(xs))
                ev(f"posture.{label}.{key}.sd", sd, units, conf, len(xs))
            conditions.append(row)

        plate_payload = {
            "file": path.name, "sha256": export.sha256,
            "athlete_matched": export.athlete,
            "rows_in_file": export.n_rows_total,
            "rows_for_this_athlete": len(export.trials),
            "other_athletes_in_file": len(export.other_athletes),
            "subject_block": export.subject,
            "conditions": conditions,
            "unavailable_measures": export.unavailable,
            "min_trials_for_sd": min_for_sd,
        }
        note("force-plate export", True,
             f"{path.name} -> \"{export.athlete}\", {len(export.trials)} trials, "
             f"{len(conditions)} conditions")

        res.check("force_plate_parsed", True,
                  f"{len(export.trials)} trials for \"{export.athlete}\" out of "
                  f"{export.n_rows_total} rows, {len(conditions)} condition(s).")
        res.check("force_plate_conditions_kept_separate", True,
                  "Conditions: " + ", ".join(
                      f"{c['condition']} (n={c['n_trials']}, {c['definition']['duration_s']}s)"
                      for c in conditions))
        res.check("unrecorded_measures_dropped", True,
                  f"Dropped as not recorded: {sorted(export.unavailable)}"
                  if export.unavailable else "Every whitelisted measure was recorded.",
                  severity=WARN if export.unavailable else FAIL)

        # Cross-check what the operator typed against the instrument's own
        # subject block. A disagreement is reported, never silently resolved.
        conflicts = []
        for field, typed in (("height_cm", height_cm), ("weight_kg", weight_kg)):
            from_file = export.subject.get(field)
            if typed is not None and from_file is not None and abs(typed - from_file) > 0.5:
                conflicts.append(f"{field}: you entered {typed:g}, the export says {from_file:g}")
        plate_payload["anthropometric_conflicts"] = conflicts
        res.check("anthropometrics_agree_with_export", not conflicts,
                  "; ".join(conflicts) + ". The entered value is used for this run and the "
                  "report states the disagreement." if conflicts else
                  "Entered height and weight agree with the export's subject block.",
                  severity=WARN if conflicts else FAIL)

    # ---- payload -----------------------------------------------------------
    payload = {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "inventory": inventory,
        "anthropometrics": {
            "height_cm": _r(height_cm, 1), "weight_kg": _r(weight_kg, 1), "age_y": age,
            "bmi": bmi, "bmi_formula": bmi_spec.get("formula"),
            "bmi_source": bmi_spec.get("source"),
            "is_youth_athlete": is_youth,
            "youth_threshold_y": youth_threshold,
            "youth_source": youth_spec.get("source"),
        },
        "heart_rate": {
            "rest_bpm": _r(hr_rest, 0), "mean_bpm": _r(hr_mean, 0),
            "max_observed_bpm": _r(hr_max_obs, 0),
            "max_estimated_bpm": hr_max_est,
            "max_formula": max_spec.get("formula"),
            "max_label": max_spec.get("label"),
            "max_source": max_spec.get("source"),
            "zones": zones,
            "zones_source": zone_spec.get("source"),
        },
        "force_plate": plate_payload,
        "force_plate_images": [str(i) for i in images],
        "evidence": evidence,
    }
    out = ctx.write_json("00b_physio.json", payload)
    res.outputs["physio"] = str(out)

    supplied = [i["input"] for i in inventory if i["status"] == "SUPPLIED"]
    absent = [i["input"] for i in inventory if i["status"] != "SUPPLIED"]
    res.check("inputs_inventoried", True,
              f"supplied: {supplied or 'none'} | not provided: {absent or 'none'}")
    res.check("every_key_has_units_and_confidence",
              all(e.get("confidence") and (e.get("units") is not None) for e in evidence.values()),
              f"{len(evidence)} physiological evidence key(s)")
    if is_youth:
        res.check("youth_athlete_flagged", True,
                  f"Age {age} is below {youth_threshold}. Training-load guidance will state "
                  f"that volume and progression follow youth guidelines and need coach "
                  f"sign-off.", severity=WARN)
    res.stats = {"evidence_keys": len(evidence),
                 "force_plate_trials": plate_payload.get("rows_for_this_athlete", 0),
                 "conditions": len(plate_payload.get("conditions", []))}
    return res
