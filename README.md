# SAI: Archery Biomechanics Analysis Pipeline

Turns a recorded archery video into a fixed-format, evidence-based biomechanics
report, reproducibly.

## The one rule

> **Python measures. The model only writes prose over a frozen evidence file.**
> Any number that is not already in `05_metrics.json` cannot appear in the report.

This is the entire reason the pipeline exists. A single large prompt asking a
model to perceive, measure, judge and typeset in one pass produces a different
report every time. Splitting measurement (deterministic) from narration
(generative), and gating the narration against a measured evidence file, makes
the output repeatable.

## Pipeline

| Step | Name | Owner | Key output |
|---|---|---|---|
| S0 | ingest | ffprobe + schema | video hash, measured fps, validated session |
| S1 | frames | ffmpeg | single decode pass to JPEG |
| S2 | pose | MediaPipe | 33 landmarks per frame, image and world space |
| S3 | kinematics | NumPy | joint angles, reference frames, reference lines |
| S4 | phases | rules | deterministic shot and phase boundaries |
| S5 | stats | NumPy | **`05_metrics.json`, the evidence file** |
| S6 | annotate | OpenCV | one fully overlaid key frame per phase |
| S7 | video | ffmpeg | annotated MP4 plus per-phase clips |
| S8 | narrate | Qwen via Ollama | one JSON call per report section |
| S9 | verify | guardrail | consistency checklist + numeric grounding |
| S10 | render | Jinja2 | single self-contained versioned HTML |

Every step declares INPUT, OUTPUT and VERIFY. A step refuses to start if any
upstream check failed, so a bad measurement never reaches the report.

## Getting started

Windows: see [`docs/02_RUNBOOK.md`](docs/02_RUNBOOK.md).
Design and rationale: [`docs/00_PLAN.md`](docs/00_PLAN.md).

```powershell
.\scripts\bootstrap.ps1
archery init-session --video "D:\archery\shot.mp4"
archery run --video "D:\archery\shot.mp4" --to S5
```

## Status

| Steps | State |
|---|---|
| S0 ingest, S1 frames, S2 pose, S3 kinematics | implemented, tested |
| S4 phases, S5 stats (evidence file) | implemented, tested |
| S6 annotate, S7 annotated video | implemented, tested, visually checked |
| S8 narrate, S9 verify, S10 render | next |

`archery selftest` runs the whole S3 measurement layer against a synthetic
archer with known geometry. No video, no model, about one second.
