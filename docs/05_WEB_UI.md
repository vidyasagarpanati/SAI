# 05 - Local web UI (`archery serve`)

Status: BUILT. All five phases ship. `archery serve`.

## What it is

A thin local front end over the CLI that already exists. It starts the same
code path, reads the same `state.json`, and serves the same artefacts. It adds
no analysis logic of its own, so a number that is wrong is wrong in the
pipeline and gets fixed there.

## Decisions

| # | Decision | Chosen |
|---|---|---|
| W1 | Demo pacing | A library of finished runs is the landing page, so a real report is one click away. Starting a live run also genuinely works. |
| W2 | Binding | `127.0.0.1` only, screen-shared. No authentication, because nothing is exposed. |
| W3 | Video input | Pick from a watched folder. The source file is never copied, moved or opened for writing. |
| W4 | Stack | FastAPI plus the Jinja templates already in the repo. No build step, no node_modules. |
| W5 | Batch | Several videos can be queued and run one after another, FIFO, one at a time. |
| W6 | Scope | Built as the product front end, with the demo as the first milestone. |

## Why `state.json` is the whole data layer

Every step already records its status, start, finish, duration, input hash,
output paths, stats, and every check with its name, severity and detail. The
library row, the live progress table, the links to the report and the
annotated video, and the "why did this fail" panel are all reads of a file the
pipeline writes anyway. No database, no second index, nothing to keep in sync.

## Architecture

```
archery serve --port 8000
   |
   +-- FastAPI app, 127.0.0.1 only
   |      pages rendered with Jinja, progress by polling every 2s
   |
   +-- JobQueue: one worker, FIFO
          each job = subprocess.Popen([sys.executable, "-m", "archery.cli",
                                       "run", "--video", ...])
```

A subprocess rather than an in-process call, for three reasons. MediaPipe has
already crashed this project's interpreter once on Windows and a crash must not
take the server with it. Cancel becomes a kill. And it keeps the CLI the single
entry point, so the web UI cannot drift away from what the command line does.

One job at a time: the pipeline is CPU-bound on a machine that is already
loaded, and two runs of the same video would write to the same run directory.

Progress comes from two places: `state.json` for structure, and the captured
stdout for the running commentary the CLI already prints.

## Pages

| Route | What it shows |
|---|---|
| `/` | Library. One row per run directory: athlete, video, date, status, elapsed or total, a dot per step, and links to the report, the annotated video and the evidence file. Checkboxes feed the compare form. |
| `/run/{run_id}` | The live step table, each step's checks with their detail, the captured log, and a cancel button while it runs. |
| `/new` | The session form and the video picker. |
| `/compare` | The runs ticked in the library, an SD threshold, and a button that produces the comparison report. |
| `/doctor` | The environment check. A failed demo gets a one-click diagnosis rather than a terminal. |

## The new-run form

Every field comes from `schemas/session.schema.json`, so the form and the
schema cannot disagree: athlete name, age, gender, height, weight, bow type,
discipline, draw hand, camera view, camera distance, session date, declared
frame rate, expected shots, notes. Plus the free-text profile block, the
force-plate export, and any plate images.

Before anything starts, the form echoes back what it understood and names every
NOT PROVIDED field, exactly as the CLI does. The same parser does the work, so
there is one definition of what "Height: 170 cms" means.

`session_date` is worth filling in here even though it is optional: `compare`
uses it to warn when the order of runs disagrees with the actual chronology.

## Serving files safely

The browser never sends a path. It sends a run id; the server reads that run's
`state.json`, takes the output path the pipeline recorded, and refuses to serve
it unless it resolves inside `outputs/` or `runs/`. The annotated video is
served with range requests so it can be scrubbed.

The web layer opens nothing for writing. Every write still goes through the
pipeline and its io_guard whitelist, and a test asserts the web package
contains no write path of its own.

## Phases

1. **Library, run detail, artefact serving, doctor.** Demo-capable against runs
   made from the command line.
2. **New run:** one video, the full session form, the physio inputs, live
   progress, cancel.
3. **Queue:** several videos, FIFO.
4. **Compare from the UI.**
5. **Polish:** filter by athlete, group by session date, a run's key frames
   inline.

## Verification

1. The artefact routes refuse a run id that escapes the runs directory, and
   refuse a recorded path that resolves outside `outputs/` or `runs/`.
2. The web package imports nothing that writes, and contains no `open(...,"w")`,
   mirroring the pose-worker isolation test.
3. The library parses a real `state.json` from a completed run, including one
   that FAILED mid-way and one that produced a PARTIAL report.
4. Starting a job twice for the same video queues the second rather than
   running both.
5. Cancel kills the subprocess and the run is left resumable, not corrupt.
6. The form rejects the same out-of-range values the CLI rejects, with the same
   message, because it calls the same parser.
7. A Playwright smoke run: the library loads, a report opens, the video element
   gets a 206 response, and compare produces a file.
8. `archery serve` binds 127.0.0.1 and a request to the machine's LAN address
   is refused.

## Demo risks worth naming

- **Ollama must be up.** S8 is 15 of the 22 minutes and it fails without it.
  The doctor page shows its status and the model name before you start.
- **A step failing in front of the room.** This is survivable and arguably a
  good story: the run detail page names the failing check, and the pipeline
  still publishes a PARTIAL report with the affected sections marked NOT
  AVAILABLE.
- **The library must not be empty.** Keep at least three finished runs and one
  comparison on the machine, made the day before.

## New dependencies

`fastapi` and `uvicorn`, pinned. Nothing else: the templates, the charts and
the reports are all already in the repo.


## Part 2 - What shipped

| File | What it does |
|---|---|
| `src/archery/web/library.py` | Reads `runs/*/state.json` into rows, and resolves a run id to the files it produced. All path safety lives here, in one place. |
| `src/archery/web/jobs.py` | Single-worker FIFO queue. Each job is a subprocess of the CLI. |
| `src/archery/web/app.py` | The routes. Serves and starts; computes nothing. |
| `src/archery/web/stages.py` | Step ids to labels a person in the room can read, and the parser that spots `== S2 pose: started` in the output. |
| `templates/web/*.html.j2` | Eight pages in the same visual language as the reports. |
| `src/archery/cli.py` | `archery serve`, and `doctor_rows` split out of `cmd_doctor` so the CLI and the page cannot disagree about what is wrong. |
| `tests/test_web.py` | 27 tests against real run directories. |

```
archery serve --videos "D:\Archery-InputData" --open
```

### Things worth knowing

- **A run is DONE when S10 is DONE**, not when every step ran. S7 is skippable
  and a skipped annotated video does not make a finished run unfinished.
- **The recorded path is not a permission.** Even a path the pipeline itself
  wrote into `state.json` is refused unless it resolves inside `runs/` or
  `outputs/`. A test moves a recorded report outside those roots and asserts
  the server stops serving it.
- **The same video cannot be queued twice**, because both runs would write into
  one run directory and fight over `state.json`.
- **Cancel leaves the run resumable.** Every finished step keeps its cached
  output, so starting again picks up where it stopped.
- **The form and the CLI share a parser.** `Height: 170 cms` means the same
  thing in both, and an out-of-range value is refused with the line quoted.
- Session files land in `sessions/<video>.json`, so any run started from the
  browser can be repeated from the command line.
- **The queue shows the stage, not the last log line.** `Stage 10 of 12 -
  Writing the report`, with a progress rule under it that turns green when the
  run finishes and red when it fails. The stage number comes from the step's
  position in `STEP_ORDER`, not from counting output lines, so a step served
  from cache still advances the count honestly. The last line the pipeline
  printed is kept as the cell's tooltip, so the operator detail is not lost.

### Verification, all passing (154 tests, 27 new)

Run ids that are not directory names are refused, including `..`, separators,
percent-encoded traversal and the empty string. Artefact and frame routes
refuse traversal. A recorded path outside the served roots is not served. The
library parses a real `state.json`, including a run that failed at S3, and the
failing check reaches the page with its detail. The web package contains no
write path of its own except the one session write, which goes through
`io_guard`. Queueing the same video twice is refused, several videos run in
order, cancelling a queued job removes it. The form rejects what the CLI
rejects. Compare is reachable from the library and its refusals are shown
rather than swallowed. `cmd_serve` binds 127.0.0.1 and contains no `0.0.0.0`.

### Not built, deliberately

No authentication, because nothing is exposed. No upload: the folder picker
means a 500MB file is never copied. No re-rendering of reports: the pages
serve the files the pipeline already made.
