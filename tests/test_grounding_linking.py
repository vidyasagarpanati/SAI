"""Auto-linking of restated evidence values, using the exact sentences
qwen3.6:35b produced on the Kalapna run (2026-09-23)."""
from archery.grounding import check_text, link_text
from archery.report_spec import schema_for

EV = {
    "all.AIM.elbow_draw_deg.mean": {"value": 164.3, "units": "deg"},
    "all.AIM.elbow_draw_deg.sd": {"value": 1.8, "units": "deg"},
    "all.AIM.elbow_draw_deg.cv_pct": {"value": 2.3, "units": "%"},
    "shot1.AIM.elbow_draw_deg.at_key_frame": {"value": 164.3, "units": "deg"},
    "shot1.DRAW.duration_s": {"value": 0.3, "units": "s"},
    "shot2.DRAW.duration_s": {"value": 0.067, "units": "s"},
    "all.DRAW.anchor_distance_norm.cv_pct": {"value": 2.3, "units": "%"},
    "all.AIM.trunk_inclination_deg.mean": {"value": 10.3, "units": "deg"},
}


def fixed(text, subset=EV):
    out, log = link_text(text, subset, EV)
    return out, log, check_text(out, EV)


def test_restated_mean_sd_cv_are_linked_by_stat_word():
    out, log, problems = fixed("with a mean of 164.3 deg (SD 1.8 deg), and a CV of 2.3 %.",
                               {k: v for k, v in EV.items() if "elbow_draw" in k})
    assert not problems, (out, problems)
    assert "{{all.AIM.elbow_draw_deg.mean}}" in out and "{{all.AIM.elbow_draw_deg.sd}}" in out


def test_copied_evidence_lines_are_linked():
    out, _, problems = fixed("shot1.DRAW.duration_s = 0.3 s; shot2.DRAW.duration_s = 0.067 s")
    assert not problems and out.count("{{") == 2


def test_backticked_line_is_linked():
    out, _, problems = fixed("see `all.DRAW.anchor_distance_norm.cv_pct = 2.3 %`")
    assert not problems and "{{all.DRAW.anchor_distance_norm.cv_pct}}" in out


def test_measure_words_disambiguate():
    out, _, problems = fixed("trunk inclination mean of 10.3 deg")
    assert not problems and "{{all.AIM.trunk_inclination_deg.mean}}" in out


def test_invented_threshold_is_not_linked():
    _, _, problems = fixed("consistent across shots (CV% < 3 %)")
    assert problems, "a threshold must stay a violation, never be linked"


def test_approximation_is_not_linked():
    _, _, problems = fixed("a mean of ~ 164 deg")
    assert problems


def test_ambiguous_value_without_hint_is_not_linked():
    subset = {k: v for k, v in EV.items() if k in ("all.AIM.elbow_draw_deg.mean",
                                                   "shot1.AIM.elbow_draw_deg.at_key_frame")}
    out, log, problems = fixed("held at 164.3 deg", subset)
    assert problems and not log


def test_drill_beats_are_allowed():
    assert not check_text("Metronome at 60 bpm. On beat 1, begin draw. On beat 2, complete "
                          "expansion. On beat 3, hold anchor.", {}, prescriptive=True)
    assert not check_text("On beat 1, begin draw. On beat 2, expand. Step 3: hold.", {})


def test_reference_fields_are_restricted_to_real_ids():
    ids = {"strengths": ["S1", "S2"], "weaknesses": ["W1"], "errors": ["E1"]}
    sch = schema_for("s18_projection_final", {"detected_phases": ["AIM"], "ids": ids})
    ref = sch["properties"]["does_well"]["items"]["properties"]["ref"]
    assert set(ref["enum"]) == {"S1", "S2", "NOT ESTABLISHED"}
    assert "recovery_stability" not in ref["enum"]
