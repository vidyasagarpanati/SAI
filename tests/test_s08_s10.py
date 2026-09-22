"""S8 narrate, S9 verify and S10 render, end to end with a deterministic fake model."""
from __future__ import annotations

import json
import re

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
    def chat_json(self, system, user, schema, images=None):
        out = super().chat_json(system, user, schema, images)
        if "SECTION: s13_strengths" in user:
            out["items"][0]["evidence"] = "Bow elbow held at 171.9 degrees throughout."
        return out


def test_persistent_fabrication_blocks_the_report(tmp_path):
    ctx = _prepare(tmp_path, "block")
    ctx.llm = _AlwaysFabricates()
    r8 = s08_narrate.run(ctx)
    assert not r8.passed
    assert any("171.9" in c.detail for c in r8.failures)


def test_no_vision_model_gets_no_images(tmp_path):
    ctx = _prepare(tmp_path, "novision")
    ctx.llm = FakeLLM(vision=False)
    r8 = s08_narrate.run(ctx)
    assert r8.passed
    assert any(c.name == "vision_available" and not c.ok for c in r8.checks)
    log = json.loads((ctx.narrative_dir / "generation_log.json").read_text())
    assert all(x["images"] == 0 for x in log)
