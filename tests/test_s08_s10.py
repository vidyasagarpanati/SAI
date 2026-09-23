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
    assert len(ctx.llm.calls) == 9 + 14      # 9 phase calls + 14 other sections


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
    assert html.count("data:image/jpeg;base64,") == 9
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
    assert '"technical_level"' in user          # last call was s01_executive
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
