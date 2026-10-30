"""S8 narrate, S9 verify and S10 render, end to end with a deterministic fake model."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from archery import io_guard
from archery.llm import FakeLLM
from archery.steps import (s03_kinematics, s04_phases, s05_stats, s06_annotate,
                           s08_narrate, s09_verify, s10_render)
import synthetic
from helpers import make_run


def _prepare(base, name):
    df, truth = synthetic.build()
    ctx = make_run(base / name, df, truth["fps"], frames=(640, 360), outputs_dir=base / "outputs")
    for step in (s03_kinematics, s04_phases, s05_stats, s06_annotate):
        assert step.run(ctx).passed
    return ctx


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    base = tmp_path_factory.mktemp("s810")
    ctx = _prepare(base, "run")
    ctx.llm = FakeLLM()
    r8 = s08_narrate.run(ctx)
    r9 = s09_verify.run(ctx)
    r10 = s10_render.run(ctx)
    return ctx, base, r8, r9, r10


def test_s8_passes_and_writes_every_section(full):
    ctx, _, r8, *_ = full
    assert r8.passed, [f"{c.name}: {c.detail}" for c in r8.failures]
    saved = json.loads((ctx.narrative_dir / "all_sections.json").read_text())
    assert len(saved["s04_phase"]) == 9
    for sid in ("s02_quality", "s05_biomech", "s09_errors", "s15_priorities", "s01_executive"):
        assert sid in saved


def test_s8_calls_are_per_section_not_one_giant_prompt(full):
    ctx, *_ = full
    assert len(ctx.llm.calls) == 9 + 14 + 4  # 9 phases, 14 other sections, 4 coach summaries


def test_s9_checklist_all_pass(full):
    ctx, _, _, r9, _ = full
    assert r9.passed, [f"{c.name}: {c.detail}" for c in r9.failures]
    v = json.loads((ctx.run_dir / "09_verification.json").read_text())
    assert all(c["status"] == "PASS" for c in v["checklist"])
    assert len(v["checklist"]) >= 16


def test_s10_report_is_self_contained_versioned_and_complete(full):
    ctx, base, _, _, r10 = full
    assert r10.passed, [f"{c.name}: {c.detail}" for c in r10.failures]
    reports = sorted((base / "outputs").glob("Archery_Report_*_v*.html"))
    assert reports and reports[0].name.endswith("_v01.html")
    html = reports[0].read_text()
    assert "{{" not in html
    assert not re.search(r'(src|href)\s*=\s*["\']https?:', html)
    assert html.count('alt="Annotated key frame') == 9
    assert (reports[0].with_suffix(".html.sha256")).is_file()


def test_s10_substitutes_measured_values(full):
    ctx, base, *_ = full
    html = sorted((base / "outputs").glob("Archery_Report_*_v*.html"))[0].read_text()
    metrics = json.loads((ctx.run_dir / "05_metrics.json").read_text())
    elbow = metrics["per_shot"][0]["phases"]["AIM"]["measures"]["elbow_bow_deg"]["mean"]
    assert f"{elbow:g} deg" in html


def test_s10_follows_the_fixed_section_order_with_physio_inline(full):
    ctx, base, *_ = full
    html = sorted((base / "outputs").glob("Archery_Report_*_v*.html"))[0].read_text()
    ids = re.findall(r'<section id="sec-(\w+)"', html)
    assert ids[:15] == ["info", "s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08",
                        "s09", "s10", "s11", "s12", "s13", "p1"]
    assert ids[-1] == "s20"
    assert "Individualized assessment required" in html
    assert "NOT PROVIDED - CANNOT BE CONFIRMED" in html


def test_s10_second_render_is_a_new_version(full):
    ctx, base, *_ = full
    io_guard.configure([ctx.run_dir, base / "outputs"])
    assert s10_render.run(ctx).passed
    names = sorted(p.name for p in (base / "outputs").glob("Archery_Report_*.html"))
    assert names[-1].endswith("_v02.html") and len(names) == 2


def test_model_html_is_escaped(tmp_path):
    """Text from the model must never be injected as markup."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    from helpers import ROOT
    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=select_autoescape(["html", "j2"]))
    t = env.from_string("{{ x }}")
    assert "<script>" not in t.render(x="<script>alert(1)</script>")


# ---------------------------------------------------------------- adversarial
def test_fabricated_number_is_caught_and_regenerated(tmp_path):
    ctx = _prepare(tmp_path, "fab")
    ctx.llm = FakeLLM(fabricate_in="s05_biomech")      # types "172.4 degrees" once
    r8 = s08_narrate.run(ctx)
    assert r8.passed
    log = json.loads((ctx.narrative_dir / "generation_log.json").read_text())
    first = next(x for x in log if x["section"] == "s05_biomech" and x["attempt"] == 0)
    assert any("172.4" in p for p in first["problems"])
    final = json.loads((ctx.narrative_dir / "s05_biomech.json").read_text())
    assert "172.4" not in json.dumps(final)


class _AlwaysFabricates(FakeLLM):
    def chat_json(self, system, user, schema, images=None, attempt=0):
        out = super().chat_json(system, user, schema, images, attempt)
        if "SECTION: s13_strengths" in user:
            out["items"][0]["evidence"] = "Bow elbow held at 171.9 degrees throughout."
        return out


def test_persistent_fabrication_degrades_to_a_partial_report(tmp_path):
    """With report.allow_partial (the default) a section that cannot be produced
    is dropped and marked, instead of killing the run."""
    ctx = _prepare(tmp_path, "block")
    ctx.llm = _AlwaysFabricates()
    r8 = s08_narrate.run(ctx)
    assert r8.passed, [f"{c.name}: {c.detail}" for c in r8.failures]
    warn = [c for c in r8.warnings if c.name == "all_sections_pass_self_checks"]
    assert warn and "171.9" in warn[0].detail
    status = json.loads((ctx.narrative_dir / "section_status.json").read_text())
    assert status["s13_strengths"]["status"] == "FAILED"
    assert status["s13_strengths"]["classes"] == ["GROUNDING"]
    saved = json.loads((ctx.narrative_dir / "all_sections.json").read_text())
    assert "s13_strengths" not in saved


def test_partial_report_is_published_and_marked(tmp_path):
    ctx = _prepare(tmp_path, "partial")
    ctx.llm = _AlwaysFabricates()
    assert s08_narrate.run(ctx).passed
    s09_verify.run(ctx)
    r10 = s10_render.run(ctx)
    assert r10.passed, [f"{c.name}: {c.detail}" for c in r10.failures]
    report = Path(r10.outputs["report"])
    assert "_PARTIAL_v" in report.name
    html = report.read_text()
    assert "PARTIAL REPORT." in html
    assert "NOT AVAILABLE (section failed)" in html
    assert "GROUNDING" in html          # the reason is shown, not hidden
    assert "{{" not in html


def test_retry_prompts_differ_so_a_cached_answer_is_not_reused(tmp_path):
    """Retries used to repeat the identical prompt, hit the response cache and
    return the identical rejected answer three times."""
    from archery.narrator import Narrator
    ctx = _prepare(tmp_path, "retry")
    seen = []

    class _Recorder(FakeLLM):
        def chat_json(self, system, user, schema, images=None, attempt=0):
            seen.append((attempt, user))
            out = super().chat_json(system, user, schema, images, attempt)
            if "SECTION: s05_biomech" in user and attempt < 2:
                out["findings"][0]["finding"] = "Bow elbow at 123.4 degrees."
            return out

    ctx.llm = _Recorder()
    nar = Narrator(ctx, llm=ctx.llm)
    out, problems, attempts = nar.generate("s05_biomech", [], [])
    assert attempts == 2 and not problems
    s05 = [(a, u) for a, u in seen if "SECTION: s05_biomech" in u]
    assert len({u for _, u in s05}) == len(s05), "each retry must send a different prompt"
    assert "attempt 1 of" in s05[1][1] and "FINAL attempt" not in s05[1][1]


def test_no_vision_model_gets_no_images(tmp_path):
    ctx = _prepare(tmp_path, "novision")
    ctx.llm = FakeLLM(vision=False)
    r8 = s08_narrate.run(ctx)
    assert r8.passed
    assert any(c.name == "vision_available" and not c.ok for c in r8.checks)
    log = json.loads((ctx.narrative_dir / "generation_log.json").read_text())
    assert all(x["images"] == 0 for x in log)


# ---------------------------------------------------------------- output format
def test_every_prompt_carries_an_explicit_json_template(tmp_path):
    ctx = _prepare(tmp_path, "fmt")
    ctx.llm = FakeLLM()
    assert s08_narrate.run(ctx).passed
    user = ctx.llm.last_user
    assert "OUTPUT FORMAT" in user and "Return ONE JSON object" in user
    assert '"main_fix"' in user                 # last call was sum_action
    assert "WITHOUT braces" in user


def test_evidence_keys_are_restricted_to_the_calls_own_keys(tmp_path):
    ctx = _prepare(tmp_path, "keys")
    ctx.llm = FakeLLM()
    assert s08_narrate.run(ctx).passed
    import re as _re
    listed = set(_re.findall(r"^\{\{([\w.\-]+)\}\} = ", ctx.llm.last_user, flags=_re.M))
    sch = ctx.llm.last_schema
    # s01_executive has no evidence_keys; check an earlier schema instead via spec
    from archery.report_spec import schema_for
    s = schema_for("s09_errors", {"detected_phases": ["AIM"], "evidence_keys": ["a.b", "c.d"]})
    assert s["properties"]["errors"]["items"]["properties"]["evidence_keys"]["items"]["enum"] == ["a.b", "c.d"]


def test_truncated_reply_is_retried_not_fatal(tmp_path):
    ctx = _prepare(tmp_path, "trunc")
    ctx.llm = FakeLLM(truncate_in="s05_biomech")
    r8 = s08_narrate.run(ctx)
    assert r8.passed, [f"{c.name}: {c.detail}" for c in r8.failures]
    log = json.loads((ctx.narrative_dir / "generation_log.json").read_text())
    cut = [x for x in log if x["section"] == "s05_biomech" and x.get("done_reason") == "length"]
    assert cut and "shorter text fields" in cut[0]["problems"][0]


def test_braced_evidence_keys_are_normalised():
    from archery.narrator import _normalise_keys
    out = _normalise_keys({"analysis": [{"evidence_keys": ["{{all.STANCE.stance_width_norm.mean}}", "`x.y`"]}]})
    assert out["analysis"][0]["evidence_keys"] == ["all.STANCE.stance_width_norm.mean", "x.y"]


# ------------------------------------------- summary, then full analysis
def _report(base):
    return sorted((base / "outputs").glob("Archery_Report_*_v*.html"))[-1].read_text()


def test_every_section_opens_with_a_summary_above_its_full_analysis(full):
    _, base, *_ = full
    html = _report(base)
    sections = re.findall(r'<section id="sec-(\w+)">(.*?)</section>', html, re.S)
    assert len(sections) >= 21
    for key, body in sections:
        assert body.index('class="sum"') < body.index("<details"), key
    # Physio is empty in this run, so it says so at the summary level.
    assert "No heart-rate information was given" in html


def test_the_coachs_page_comes_first_and_links_to_its_evidence(full):
    ctx, base, *_ = full
    html = _report(base)
    assert html.index('id="coach"') < html.index('id="sec-info"')
    assert "Key coaching cue" in html and "Elbow behind the arrow" in html
    assert 'href="#W1"' in html and 'id="W1"' in html
    assert 'href="#ph-AIM"' in html and 'id="ph-AIM"' in html
    assert "Plain fix." in html                     # the model's plain wording is used
    assert html.count('alt="Key frame') == 9


def test_section_timing_table_shows_durations_not_a_python_repr(full):
    _, base, *_ = full
    assert "built-in method" not in _report(base)


def test_substitution_drops_units_and_values_the_model_repeated():
    from archery.grounding import substitute
    ev = {"a": {"value": 76.6, "units": "deg"}, "b": {"value": 0.071, "units": "shoulder widths"}}
    assert substitute("at {{a}} = {{a}} deg now", ev) == "at 76.6 deg now"
    assert substitute("{{b}} SW wide", ev) == "0.071 SW wide"
    assert substitute("{{a}} to {{b}}", ev) == "76.6 deg to 0.071 SW"


def test_summary_numbers_are_said_the_way_a_coach_says_them():
    from archery.report_summary import plain_numbers
    assert plain_numbers("held 164.5 deg for 39.467 s") == "held 165° for 39 seconds"
    assert plain_numbers("trunk -4.1 deg") == "trunk -4°"
    assert plain_numbers("only 0.167 s") == "only 0.2 seconds"
    assert plain_numbers("sway 0.071 SW") == "sway 0.07 shoulder widths"


def test_a_fabricated_number_in_a_summary_never_reaches_the_report(tmp_path):
    """Summaries are grounded like every section, but losing the wording must
    not lose the report: the data-built text takes over and nothing is PARTIAL."""
    class AlwaysFabricates(FakeLLM):
        def _build(self, sid, *rest):
            out = super()._build(sid, *rest)
            if sid == "sum_action":
                out["sections"][0]["headline"] = "The elbow sat at 172.4 degrees."
            return out
    ctx = _prepare(tmp_path, "sumfab")
    ctx.llm = AlwaysFabricates()
    r8 = s08_narrate.run(ctx)
    assert "sum_action" not in json.loads((ctx.narrative_dir / "all_sections.json").read_text())
    r9 = s09_verify.run(ctx)
    r10 = s10_render.run(ctx)
    assert r8.passed and r9.passed and r10.passed
    html = _report(tmp_path)
    assert "172.4" not in html


def test_s9_keeps_a_summary_that_covers_the_physio_sections():
    from archery.report_spec import schema_for
    import jsonschema
    out = {"sections": [{"key": k, "headline": "Plain.", "points": []}
                        for k in ("s09", "s10", "s11", "s12", "s13", "p1", "p2")]}
    jsonschema.validate(out, schema_for("sum_diagnosis", {"detected_phases": ["AIM"]}))


def test_jargon_in_a_summary_is_rejected_and_sent_back():
    from archery.report_spec import local_checks
    out = {"sections": [{"key": k, "headline": "Scapular control is weak.", "points": []}
                        for k in ("s01", "s02", "s03")],
           "phases": []}
    assert any("scapula" in p for p in local_checks("sum_overview", out, [], None, {}))
    out["sections"][0]["headline"] = out["sections"][1]["headline"] = "Pull the shoulder blade back."
    out["sections"][2]["headline"] = "Good."
    assert local_checks("sum_overview", out, [], None, {}) == []


def test_a_rejected_summary_call_falls_back_to_the_data_and_stays_complete(tmp_path):
    ctx = _prepare(tmp_path, "sumfail")
    ctx.llm = FakeLLM()
    assert s08_narrate.run(ctx).passed
    assert s09_verify.run(ctx).passed
    # as if every summary had been rejected
    path = ctx.narrative_dir / "all_sections.json"
    saved = json.loads(path.read_text())
    path.write_text(json.dumps({k: v for k, v in saved.items() if not k.startswith("sum_")}))
    r10 = s10_render.run(ctx)
    assert r10.passed
    html = _report(tmp_path)
    assert "PARTIAL REPORT" not in html
    assert "Plain headline" not in html and "technique error" in html
    assert 'class="sum"' in html
