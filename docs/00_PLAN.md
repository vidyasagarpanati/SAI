# Archery Biomechanics Analysis Pipeline (SAI)
## Plan v0.1 — 2026-09-19
Status: AWAITING FACTUAL INPUTS (see Section 9). No code written yet.

---

## 1. Diagnosis: why the report does not replicate

The current setup asks one LLM to do four incompatible jobs at once:
perceive the video, measure angles, judge technique, and emit a 20-section
HTML document. Three of those four are non-deterministic in an LLM.

Root causes of non-replication:

| Cause | Effect |
|---|---|
| The model is asked to "measure" | Numbers are sampled, not computed. Different every run. |
| One giant prompt, one giant output | No checkpoint. Any failure loses the whole run. |
| Long context near the window limit | Later sections degrade or get dropped. |
| No grounding file | Nothing to check the output against, so drift is invisible. |
| Vision-driven phase detection | Phase boundaries move run to run, so every downstream stat moves. |

**Core design rule for this rebuild:**
> Python measures. The model only writes prose over a frozen evidence file.
> Any number that is not already in `05_metrics.json` cannot appear in the report.

---

## 2. Confirmed decisions (from requirements session, 2026-09-19)

| # | Decision | Choice |
|---|---|---|
| D1 | Report scope | Full 20-section master-prompt spec |
| D2 | Orchestration | LangGraph (state graph + SQLite checkpoint), every node also a standalone CLI |
| D3 | LLM invocation | Python calls Ollama HTTP API, temperature 0, fixed seed. No chat UI in the loop. |
| D4 | Annotated frames in HTML | One key frame per detected phase per shot (~9-15 per shot) |
| D5 | Model input | `05_metrics.json` slice (authoritative numbers) + annotated key frames (qualitative only) |
| D6 | Benchmarks | Curated, citation-carrying `benchmarks.json`. No entry, no benchmark. |
| D7 | Video output | Full annotated MP4 + per-phase clips |
| D8 | Metadata input | `session.json` sidecar per video |
| D9 | Phase detection | Rule-based from pose kinematics, thresholds in config |
| D10 | Physiological sections | Sections 14 and 15 inserted mid-report as in the Kalpana example (see open item O1) |
| D11 | Project location | `~/Documents/Claude/Projects/SAI`. Source videos read-only, never copied over or modified. |

---

## 3. Pipeline: 11 steps, each an agent with a contract

Each step declares: INPUT, OUTPUT, VERIFY. A step refuses to start if its
upstream `verify` block has any FAIL. Each step writes its own status into
`runs/<run_id>/state.json` so a rerun skips completed work.

### S0 `ingest`
- IN: video path, `session.json`
- DO: SHA-256 the video; `ffprobe` for real fps, resolution, duration, codec, rotation; validate session.json against schema
- OUT: `00_ingest.json`
- VERIFY: file readable; hash recorded; fps > 0; every required session field present or explicitly `NOT PROVIDED - CANNOT BE CONFIRMED`; source directory is NOT on the writable whitelist

### S1 `frames`
- IN: `00_ingest.json`
- DO: single ffmpeg decode pass to JPEG at `analysis_fps` (config; default min(native, 60))
- OUT: `frames/*.jpg`, `01_frames.json`
- VERIFY: frame count matches expected within 1; no zero-byte frames; first and last frame decodable

### S2 `pose`
- IN: `frames/`
- DO: MediaPipe Pose Landmarker, `model_complexity=1`, `running_mode=VIDEO`, `num_poses=1`, `min_pose_detection_confidence=0.5`, `min_pose_tracking_confidence=0.5`. 33 landmarks with x, y, z, visibility, presence.
- OUT: `02_landmarks.parquet`, `02_pose_quality.json`
- VERIFY: >= 1 row per frame; per-frame count of landmarks with confidence >= 0.5; frames with < 20 visible landmarks flagged `INSUFFICIENT_LANDMARKS`; detection-rate summary
- NOTE: heaviest step. Checkpointed. Never reruns unless the frame hash changes.

### S3 `kinematics`
- IN: `02_landmarks.parquet`
- DO: pure NumPy. All 8 anatomical reference frames, reference lines A-J / C1-C9, joint angles (shoulder, elbow, wrist, hip, knee, ankle, trunk, neck, head), alignment chains, COM proxy, sway.
- Confidence gates per the master prompt: angle computed only if both proximal and distal landmarks have confidence >= 0.4, and >= 20/33 landmarks visible. Otherwise `NaN` with a reason code.
- OUT: `03_kinematics.parquet`, `03_quality.json`
- VERIFY: no angle outside anatomical plausibility bounds without a flag; NaN rate per angle reported; units and sign conventions asserted against a fixture

### S4 `phases`
- IN: `03_kinematics.parquet`
- DO: deterministic segmentation. Signals: draw-hand-to-face distance, draw-elbow angle and its velocity, draw-wrist displacement spike (release), bow-arm stability, COM velocity. Shot cycles first, then phases within each shot.
- OUT: `04_phases.json` (shot, phase, start/end frame, start/end timestamp, key frame, boundary confidence)
- VERIFY: phases non-overlapping and contiguous; every phase has a key frame with adequate landmark quality; release detected exactly once per shot; boundary confidence recorded; same input always yields identical output (hash-checked)

### S5 `stats`
- IN: `03_kinematics.parquet`, `04_phases.json`, `config/benchmarks.json`
- DO: per phase per shot: min, max, mean, range, SD. Cross-phase and shot-to-shot: CV, angular deviation, timing variation. Benchmark comparison only where a cited benchmark exists.
- OUT: **`05_metrics.json` — the single evidence file. Every number in the report traces to a key here.**
- VERIFY: no value derived from NaN-gated data; SD requires n >= 3 shots or is suppressed; every benchmark row carries a `citation_id` present in benchmarks.json

### S6 `annotate`
- IN: `frames/`, `03_kinematics.parquet`, `04_phases.json`
- DO: render one annotated key frame per phase per shot with the full overlay set from the master prompt: 33 landmarks with ID labels, region colouring (red face / green upper limb / blue lower limb), MediaPipe topology, low-confidence points dimmed with confidence shown, reference-frame origin markers, reference lines C1-C9 with measured angles, joint-angle labels with leader lines and semi-transparent backgrounds, phase badge, timestamp, frame number, shot number, measurement summary overlay, observation callout, coaching implication box.
- OUT: `06_frames/*.jpg`, `06_manifest.json`
- VERIFY: one frame per extracted phase, none missing; every required overlay element present (render-time assertion, not a visual guess); no label overlaps a joint landmark; file size within budget

### S7 `video`
- IN: source video (read-only), `03_kinematics.parquet`, `04_phases.json`
- DO: same renderer as S6 applied per frame, piped to ffmpeg. Full annotated MP4 plus one clip per phase.
- OUT: `07_video/annotated_full.mp4`, `07_video/phase_*.mp4`
- VERIFY: output duration matches source within tolerance; frame count matches; clips align to phase boundaries

### S8 `narrate` (the only LLM step)
- IN: section-specific slice of `05_metrics.json` + relevant annotated key frames
- DO: one Ollama call per report section. `temperature=0`, fixed `seed`, `format=json`, per-section JSON schema. Prompt carries the master prompt's rules for that section only, not the whole document.
- The model is told explicitly: numbers come from the supplied JSON, images are for qualitative observation only, unavailable items get `NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO`.
- Response cached by hash of (section, prompt, input slice, model, seed). Reruns are free, and a single section can be regenerated alone.
- OUT: `08_narrative/sec_NN.json`
- VERIFY: valid JSON against section schema; every numeric literal present in the input slice; evidence tags [OBSERVED] / [MEASURED] / [INTERPRETED] / [INFERRED] used; confidence rating present on every major claim

### S9 `verify` (guardrail agent)
Mechanical execution of the master prompt's Consistency Control checklist, plus:
- numeric grounding: every number in the narrative exists in `05_metrics.json`
- no uncited benchmark anywhere
- all 20 sections present, in order, unrenamed
- exactly 5 improvement priorities
- all scores within 0-10
- injury language screened against a banned-phrase list (non-diagnostic enforcement)
- executive summary claims reappear in the detailed sections
- every extracted phase has an annotated frame and a scorecard entry
- OUT: `09_verification.json` (PASS/FAIL per check, with the offending text)
- ON FAIL: returns only the failing section to S8 with the specific violation. Max 2 retries per section, then the run hard-fails with a readable reason. It never silently publishes a bad report.

### S10 `render`
- IN: `08_narrative/`, `06_frames/`, `05_metrics.json`, `09_verification.json` (must be all-PASS)
- DO: Jinja2 template to a single self-contained HTML. Inline CSS, base64 images, zero external dependencies.
- OUT: `outputs/Archery_Report_<athlete>_<YYYYMMDD>_v<NN>.html` plus `.sha256` and a manifest linking report to run_id to video hash.
- VERIFY: opens standalone; no external URL references; version number strictly increments; no existing report overwritten

---

## 4. Guardrails

1. **Source video is read-only.** Opened `O_RDONLY`. Its directory is never on the writable whitelist. All writes route through `io_guard.py`, which permits only `runs/` and `outputs/`.
2. **No delete path.** The codebase contains no `rm`, `unlink`, `rmtree`, or `shutil.move` against anything outside `runs/<run_id>/tmp/`.
3. **Versioned output.** Reports are never overwritten. `_v01`, `_v02`, and so on.
4. **Evidence lock.** S9 blocks publication of any number not traceable to `05_metrics.json`.
5. **No invented benchmarks.** Section 7 renders `Individualized assessment required` unless a cited entry exists.
6. **Non-diagnostic language.** Banned-phrase screen on Section 10.
7. **Closed environment.** Ollama on localhost only. No outbound network call at runtime. Models and MediaPipe weights pinned and cached locally.
8. **Reproducibility contract.** Same video hash + same config hash + same model tag + same seed must produce a byte-identical `05_metrics.json`. Asserted by a regression test.

---

## 5. Token budget

| What | Today | After |
|---|---|---|
| Video or frames to the model | Entire video | None in the measurement path |
| Per-section input | Whole master prompt + all context | ~1-2 KB JSON slice + section rules |
| Images to the model | Many | 1-2 annotated key frames, only for qualitative sections |
| Failure cost | Whole run | One section, and it is cached |
| Estimated total | 300k+ tokens | 15-30k tokens |

---

## 6. Repository layout

```
SAI/
  docs/            00_PLAN.md, 01_DECISIONS.md, 02_RUNBOOK.md
  config/          pipeline.yaml, phase_rules.yaml, benchmarks.json,
                   report_sections.yaml, banned_phrases.txt
  schemas/         session.schema.json, metrics.schema.json,
                   section_*.schema.json
  src/archery/     io_guard.py, ingest.py, frames.py, pose.py,
                   kinematics.py, phases.py, stats.py, annotate.py,
                   video.py, narrate.py, verify.py, render.py, graph.py
  templates/       report.html.j2, partials/
  tests/           fixtures + reproducibility regression tests
  runs/<run_id>/   00_ingest.json ... 09_verification.json, state.json
  outputs/         Archery_Report_<athlete>_<date>_v<NN>.html
```

Run with: `python -m archery run --video <path> [--resume] [--from S8] [--only S6]`
Monitor with: `python -m archery status <run_id>`

---

## 7. Dependencies

mediapipe, opencv-python-headless, numpy, pandas, pyarrow, jinja2,
langgraph, langgraph-checkpoint-sqlite, httpx, pydantic, jsonschema, ffmpeg.
No model weights beyond the MediaPipe pose task file and the Ollama model
already present.

---

## 8. Open items

- **O1** Section numbering with physio sections. D10 places 14 and 15 mid-report as in the Kalpana example, which pushes the master prompt's 14-20 to 16-22 and breaks numeric comparability with the protocol. Recommended alternative: keep 1-20 fixed and render physio as Appendix A and B. Needs a decision before templating.
- **O2** Qwen vision capability must be confirmed before D5 is implementable.
- **O3** True frame rate. The master prompt states ~1000 fps, which is implausible for the source. The report must state the measured rate from ffprobe.

## 9. Blocking inputs needed from Vidya

1. Path to the archery video(s). Nothing matching was found in Downloads.
2. Output of `ollama list` (exact model tag; "Qwen 3.6" is not a published tag).
3. Mac chip, RAM, free disk.
4. Draw hand and camera view per athlete or as a default.
5. The original Kalpana video, to validate the rebuild against the known-good report.
6. Python environment in use (version, venv or conda) and whether installs are permitted.
7. Force-plate and heart-rate source formats for sections 14 and 15.

---

## 10. Addendum (2026-09-19): target is a remote Windows box

Decisions added after the environment was confirmed:

| # | Decision | Choice |
|---|---|---|
| D12 | Execution host | Remote Windows machine, reached over RustDesk. Everything runs there. |
| D13 | Code delivery | GitHub repo `vidyasagarpanati/SAI`. Built and maintained in the Mac working copy, pushed from the Mac terminal (which holds the SSH key), pulled on Windows. |
| D14 | Network | Full internet on the Windows box, so `pip` and `winget` bootstrap normally. No offline wheelhouse needed. |
| D15 | Python | 3.11 or 3.12 only. MediaPipe ships no 3.13 wheels. The bootstrap script refuses to continue otherwise. |
| D16 | GPU | RTX 3060 12GB reserved entirely for Ollama. MediaPipe pose runs on CPU by deliberate choice, so nothing contends for VRAM mid-run. |
| D17 | ffmpeg | `winget install Gyan.FFmpeg`, with the `imageio-ffmpeg` bundled binary as fallback. ffprobe absent degrades to OpenCV metadata, which is recorded in the report. |

Windows-specific implementation notes:

- All paths go through `pathlib`. No separator is ever hardcoded.
- `archery.bat` wraps the venv so the CLI works without activating it.
- `io_guard.open_source_readonly` passes `O_BINARY` on Windows so the read-only
  guarantee is kernel-enforced, not conventional.
- The run id is `<video stem>__<video sha8>__<config sha8>`, so rerunning the
  same video with the same config resumes rather than starting over. Changing
  any config value starts a new run and leaves the old one intact.

### Build status

| Steps | State |
|---|---|
| Scaffold, config, schemas, io_guard, run state, CLI, LangGraph, bootstrap | done |
| S0 ingest, S1 frames | done |
| S2 pose (isolated worker), S3 kinematics | done |
| S4 phases, S5 stats | done |
| S6 annotate, S7 video | done |
| S8 narrate, S9 verify, S10 render | queued |

### Change log (2026-09-22)

- Cache keys are now per step (`runner.STEP_DEPS`): each step hashes only the
  config slice, session fields and upstream outputs it reads. The run id is
  `<video stem>__<video sha8>`. Tuning `phase_rules.yaml` re-runs S4 onward and
  never pose estimation; renaming the athlete never re-decodes the video.
- S2 runs MediaPipe in an isolated worker process (pyarrow/MediaPipe DLL clash
  on Windows).
- phase_rules.yaml v2: every key is read by S4, nothing decorative.
- Phase badges follow the master prompt: STANCE 1, PRE_DRAW 2, DRAW 3,
  ANCHOR 4, AIM 5, EXPANSION 6, RELEASE 7, FOLLOW_THROUGH 8, RECOVERY 9.

### Change log (2026-09-23)

- S6/S7 share one renderer (`overlay.py`), so key frames and video cannot disagree.
  Side panel holds the measurement table, observation and coaching boxes, so text
  never covers the athlete. The renderer reports every overlay it drew or skipped
  with a reason; S6 fails if a required overlay is neither.
- Weight distribution is labelled NOT ASSESSABLE FROM VIDEO (needs force data);
  COM ground projection is labelled [INFERRED].
- Fix found by visual inspection: speeds were differentiated from raw landmark
  positions, so jitter read as motion (held aim ~1.16 SW/s). Now computed from
  smoothed positions (~0.21 SW/s); regression test added.
