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
