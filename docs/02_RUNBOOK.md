# Runbook: remote Windows box (RTX 3060 12GB)

Target machine reached over RustDesk. Everything below runs in PowerShell on that box.

---

## One-time setup

```powershell
# 1. Clone
cd D:\                       # or wherever you keep projects
git clone https://github.com/vidyasagarpanati/SAI.git
cd SAI

# 2. Bootstrap: venv, pinned deps, ffmpeg, MediaPipe pose model, env check
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\bootstrap.ps1

# 3. Tell the pipeline which Ollama model to use
ollama list                  # copy the exact tag
notepad config\pipeline.yaml # set llm.model to that tag
```

`bootstrap.ps1` refuses to continue on Python 3.13, because MediaPipe publishes
no wheels for it. Use 3.11 or 3.12.

---

## Per-video run

```powershell
.\.venv\Scripts\Activate.ps1

# 1. Create the metadata sidecar
archery init-session --video "D:\archery\kalpana_shot01.mp4"

# 2. Fill it in. draw_hand and camera_view are mandatory and change the analysis.
notepad sessions\kalpana_shot01.json

# 3. Check the environment is sane
archery doctor

# 4. Run to the evidence file, inspect the numbers before spending GPU time
archery run --video "D:\archery\kalpana_shot01.mp4" --to S5

# 5. Full run
archery run --video "D:\archery\kalpana_shot01.mp4"
```

Output lands in `outputs\Archery_Report_<athlete>_<date>_v<NN>.html`.
Version numbers increment. No report is ever overwritten.

---

## Tuning phase detection (S4)

Thresholds live in `config/phase_rules.yaml`. Tune them from data, not by eye:

1. `archery run --video ... --to S5`
2. Open `runs\<run_id>\04_phases.json` and read `signal_summary`: the
   percentiles of every signal S4 uses, measured on your footage.
3. If no shot was detected, compare `anchor_distance_norm.p1` (how close the
   draw hand actually gets to the face) with `anchor_distance_enter`.
4. Edit the threshold and rerun with `--from S4`. Pose estimation is not
   repeated: each step's cache key covers only the config it reads.

A phase the signals cannot support is written as `detected: false` with a
reason, never with a guessed boundary. `checks_S4.json` lists every one.

## Annotated outputs (S6, S7)

| Output | Where |
|---|---|
| One annotated key frame per detected phase | `runs\<run_id>\06_frames\shot<k>_p<badge>_<PHASE>.jpg` |
| What each frame drew or skipped, and why | `runs\<run_id>\06_manifest.json` |
| Full annotated video with live joint angles | `runs\<run_id>\07_video\annotated_full.mp4` |
| One clip per phase | `runs\<run_id>\07_video\shot<k>_p<badge>_<PHASE>.mp4` |
| Published copy, versioned, never overwritten | `outputs\<Athlete>_Annotated_<date>_vNN.mp4` |

Numbers on key frames are the rounded values from `05_metrics.json`, so the
images and the report agree exactly. The coaching box on key frames reads
PENDING until S10 re-renders the frames with the verified narrative. Video
rendering runs on CPU and prints progress; expect roughly real time or slower.

## The report (S8, S9, S10)

Before the first full run, check that the model returns structured JSON:

```powershell
archery doctor --llm
```

`structured output` must be `[ok ]`. `model vision` tells you whether key frames
will be sent to the model; without vision, anything that needs looking at the
image (grip, string contact, finger relaxation) is reported as NOT RELIABLY
ASSESSABLE instead of guessed.

How the narrative is kept honest:

1. **S8** makes one call per section (one per detected phase for Section 4),
   temperature 0, fixed seed, JSON schema enforced. The model never types a
   measured number: it writes evidence keys like
   `{{shot1.AIM.elbow_bow_deg.mean}}` and the renderer substitutes the value.
   Each output is checked before it is accepted (schema, grounding, section
   rules) and regenerated with the exact violations listed if it fails.
2. **S9** runs the master prompt's consistency checklist mechanically, plus
   cross-section checks (executive summary and final summary must match the
   ranked weaknesses and priority #1). Failing sections go back to the model.
   If S9 still fails, **no report is written**.
3. **S10** writes `outputs\Archery_Report_<Athlete>_<date>_vNN.html` with a
   `.sha256` and `.manifest.json`. Self-contained, versioned, never overwritten.

Expect about 23 model calls per video. With `qwen3.6:35b` partly offloaded
from the 12GB card, allow several minutes per call on the first run. Every call
is cached, so re-running is fast, and editing one section's instructions in
`config/prompts/sections.yaml` regenerates only what changed.

Token usage per run is in `runs\<run_id>\08_narrative\usage.json` and in the
report's Section 20 provenance table.

## What a run costs

Measured on the Kalpana video (4 shots, 9 phases, RTX 3060 12GB, qwen3.6:35b):

| Step | Time |
|---|---|
| S0 ingest, S1 frames | seconds |
| S2 pose | about 2.5 min |
| S3 kinematics, S4 phases, S5 stats | seconds |
| S6 annotate | about 16 min |
| S7 video | about 3.5 min |
| S8 narrate | about 12.5 min (23 calls, 25-50 s each) |
| S9 verify, S10 render | seconds, plus any repair calls |

Every step prints its start, result and duration, and S8 prints one line per
model call with tokens and a running ETA. `archery status` shows the same table
afterwards. Reruns reuse cached model replies, so an unchanged section costs
nothing.

If a step is skipped as unchanged it prints nothing and shows `-` for its
duration in the table. That is the resume cache doing its job.

## Day-to-day commands

| Need | Command |
|---|---|
| Where did the run get to | `archery status` |
| State of a specific run | `archery status <run_id>` |
| Rerun one step only | `archery run --video ... --only S4` |
| Redo the narrative after editing a prompt | `archery run --video ... --from S8 --force` |
| Ignore the resume cache entirely | `archery run --video ... --force` |
| Bypass LangGraph while debugging | `archery run --video ... --no-graph` |
| List the steps | `archery steps` |

The expensive step is S2 (pose). It is cached against a hash of the config and
the video, so changing a phase threshold and rerunning costs seconds, not
minutes. Only editing something that S2 actually depends on forces a redo.

---

## Pull updates

```powershell
cd D:\SAI
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # only if requirements changed
```

If a config default changed, your existing runs keep their own `config_hash`, so
old runs stay reproducible and the next run starts fresh under a new run id.

---

## What runs where

| Component | Device | Note |
|---|---|---|
| ffmpeg decode | CPU | one pass per video, cached |
| MediaPipe pose | CPU | deliberate. Keeps all 12GB of VRAM free for Ollama. |
| Kinematics, phases, stats | CPU, NumPy | milliseconds |
| Frame and video annotation | CPU, OpenCV | no model involved |
| Qwen narrative | GPU via Ollama | one call per report section |

---

## Guardrails you can rely on

1. The source video is opened `O_RDONLY`. Its directory is never on the writable
   whitelist, and S0 hard-fails if it overlaps one.
2. There is no delete primitive in the codebase. Nothing is ever removed.
3. All writes route through `io_guard`, which permits only `runs/`, `outputs/`,
   `sessions/` and `models/`.
4. Reports are versioned, never overwritten.
5. S9 blocks publication of any number that is not present in `05_metrics.json`.
6. Section 7 renders `Individualized assessment required` unless a cited entry
   exists in `config/benchmarks.json`.
7. Ollama is called on localhost only. No outbound network at run time.

---

## Troubleshooting

**`archery doctor` says the pose model is missing.** Rerun `bootstrap.ps1`, or
download it by hand to `models\pose_landmarker_full.task` from
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`.

**ffprobe not found.** Not fatal. Metadata falls back to OpenCV, which gives fps,
resolution and frame count but not codec or rotation. Install ffmpeg properly
with `winget install Gyan.FFmpeg` and open a new PowerShell window.

**Why S2 runs MediaPipe in a separate process.** On this Windows box,
MediaPipe's native bindings fail to initialise if pyarrow is already loaded in
the same process (both bundle protobuf and abseil). pandas 2.x loads pyarrow
eagerly, so S2 hands pose estimation to `archery.pose_worker`, a separate
interpreter that loads only numpy, cv2 and mediapipe. `archery doctor` reports
both the worker's health and whether the conflict exists on the machine. Check
the worker on its own with `.\.venv\Scripts\python.exe -m archery.pose_worker --selftest`.

**S2 fails with `DLL load failed while importing _framework_bindings`.** Two
OpenCV distributions are fighting over `site-packages\cv2`, or the Visual C++
runtime is missing. Run `.\scripts\repair_opencv.ps1`, which removes every
OpenCV package, deletes the leftover `cv2` directory, reinstalls only
`opencv-contrib-python` (the one MediaPipe is built against), reinstalls
MediaPipe and verifies the import. If it still fails,
`winget install Microsoft.VCRedist.2015+.x64` and open a new PowerShell window.

**A step fails its own checks.** Read `runs\<run_id>\checks_S<N>.json`. Every
check records what it looked at. Fix the cause, then rerun that step with
`--only S<N> --force`.

**Frame count hit the cap.** `frames.max_frames` in `config/pipeline.yaml`. The
default 20000 is a disk guard, not a quality limit.
