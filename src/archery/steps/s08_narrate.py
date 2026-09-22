"""S8 narrate (the only step that uses the language model).

IN     05_metrics.json (sliced per section), 06_manifest.json (key frames)
DO     one Ollama call per report section (one per detected phase for Section
       4), temperature 0, fixed seed, schema-enforced JSON, cached by request
       hash. Each output is self-checked (schema, evidence grounding, section
       rules) and regenerated with the violations listed when it fails.
OUT    08_narrative/<section>.json, all_sections.json, generation_log.json,
       usage.json (token counts)
VERIFY every section generated and passing its own checks
"""
from __future__ import annotations

from archery.context import Context
from archery.contracts import WARN, StepResult
from archery.narrator import Narrator


def run(ctx: Context) -> StepResult:
    res = StepResult(step="S8")
    nar = Narrator(ctx, llm=ctx.llm)
    summary = nar.run_all()
    usage = summary["usage"]

    res.check("vision_available", nar.vision,
              "Key frames sent to the model." if nar.vision else
              "Model lacks vision (or disabled): image-dependent observations are marked "
              "NOT RELIABLY ASSESSABLE.", severity=WARN)
    res.check("all_sections_pass_self_checks", not summary["failed"],
              "All sections passed schema, grounding and section rules." if not summary["failed"] else
              "; ".join(f"{k}: {v[:3]}" for k, v in summary["failed"].items()))
    res.check("few_retries", summary["retries"] <= 3,
              f"{summary['retries']} regeneration(s) were needed.", severity=WARN)
    res.outputs["narrative"] = str(nar.dir / "all_sections.json")
    res.outputs["usage"] = str(nar.dir / "usage.json")
    res.stats = {"sections": len(nar.outputs), "calls": usage.get("calls"),
                 "cached": usage.get("cached"), "prompt_tokens": usage.get("prompt_tokens"),
                 "completion_tokens": usage.get("completion_tokens"),
                 "retries": summary["retries"]}
    return res
