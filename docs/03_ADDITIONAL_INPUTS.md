# 03 - Angle Accuracy and Additional Inputs

Scope: the two enhancements requested after tag `v0.1.0-pipeline-complete`.
Prompt changes are DONE (Part 0). Code changes are NOT started and wait on the
open decisions in Part 4.

---

## Part 0 - Prompt changes already made

`config/prompts/system.md`, four new hard rules, additive only:

| Rule | Purpose |
|---|---|
| 9  | Key prefix names the source (`shot*`/`all*` video, `posture.*` plate, `hr.*` heart rate, `athlete.*` operator). Sources may not be merged. Missing non-video source prints `NOT PROVIDED - CANNOT BE CONFIRMED`, distinct from the video wording in rule 4. |
| 10 | `_2d` keys are image-plane, plain keys are 3D. Cite one, never average. `FORESHORTENED` means the two disagree: direction only, never magnitude. |
| 11 | Heart-rate zones are estimates from a named formula. Guidance limited to load, pacing, breathing, timing, recovery. No medication, no condition names, no symptom reading. Out-of-range value prints `REFER TO A QUALIFIED PRACTITIONER`. |
| 12 | Anthropometrics are operator-entered facts. Usable only for training-load appropriateness, drill scaling and the age-based heart-rate estimate. No health or body-composition judgement. |

`config/prompts/sections.yaml`:

- `p1` (PHYSIOLOGICAL INTERVENTION & HEART RATE ANALYSIS) written, replaces the
  current placeholder string in S10.
- `p2` (INTEGRATED INTERVENTION) written, cross-checks plate against video and
  says explicitly when the two agree or conflict.
- `s02_quality` now also prints the inventory of supplied vs NOT PROVIDED inputs.
- `s05_biomech` now prefers `posture.*` over video estimation for centre of
  gravity, sway and symmetry, and must name the force plate as the source.

Both physio sections already exist in `report_spec.MASTER`/`PHYSIO`, so the
fixed 20-section order does not change and nothing is renumbered.

---

## Part 1 - Feature 1, elbow and shoulder angle accuracy

### 1.1 Diagnosis

Three separate causes, all confirmed in the code.

1. **Wrong space.** `s03_kinematics.add_angle` defaults to `space="world"`, so
   every joint angle (`elbow_*`, `shoulder_*`, `wrist_*`, `hip_*`, `knee_*`,
   `ankle_*`) is computed from MediaPipe world landmarks. Their `wz` is inferred
   from a single view; at full draw the upper arm points partly at the camera,
   which is the worst case for inferred depth. The number printed next to the
   elbow therefore disagrees with what the eye sees in that same frame. The
   vertex/ray order is correct (`vertex=elbow, rays=shoulder,wrist`) so the
   definition is not the fault.
2. **No aspect correction.** `img` is normalised `x,y` in `[0,1]`. On 16:9 the
   x axis is stretched 1.78x relative to y, so any angle taken in that space is
   distorted. This already affects the shipped `shoulder_tilt_deg`,
   `pelvic_tilt_deg`, `trunk_*` and head-tilt measures.
3. **Model capacity.** `pose_landmarker_full.task` is the least accurate of the
   three for limb endpoints under occlusion (bow hand, string hand).

### 1.2 Changes

- `geometry`: image-plane angles take pixel coordinates (`x*W`, `y*H`).
  `framedata` already carries `width`/`height`.
- `s03_kinematics`: every joint angle emitted twice,
  `<name>_deg` (3D, unchanged key and meaning) and `<name>_deg_2d`
  (image plane, aspect-corrected), plus `<name>_deg_2d3d_diff` = absolute
  difference.
- New second shoulder definition, both spaces:
  `shoulder_<side>_girdle_deg` = vertex shoulder, rays elbow and **opposite
  shoulder**. This is the arm-to-shoulder-line angle a coach reads at full
  draw; the existing `shoulder_<side>_deg` (arm to trunk) is kept unchanged.
  The report prints both, labelled, so the two are never confused.
- Reliability flag: new gate `quality_gates.angle_2d3d_disagree_deg`
  (default 15). If the median `..._2d3d_diff` inside a phase exceeds it, that
  measure's confidence in `05_metrics.json` drops to LOW with reason
  `FORESHORTENED`. Rule 10 then forces the model to describe direction only.
- `pipeline.yaml`: `paths.pose_model_file: pose_landmarker_heavy.task`.
  `doctor` must verify the file is present before S2 runs.
- Overlay (`overlay.py`, used by S6 and S7): the number drawn next to a joint is
  the `_2d` value, suffixed `2D`, because it is the only one that can agree with
  the pixels the viewer is looking at. The side panel adds the 3D value on a
  second line when the pair disagrees by more than the gate, marked
  `FORESHORTENED`.
- Report: both values and the difference, with one sentence of fixed rendered
  text explaining the two spaces. No model prose needed for that explanation.

### 1.3 Verification steps for this feature

Added to `tests/test_s03_kinematics.py` unless noted.

1. Planar synthetic archer (all landmarks in the camera plane):
   `|_deg - _deg_2d| < 0.5` for every joint.
2. Same archer rotated 40 deg about the vertical axis: `_2d3d_diff` exceeds the
   gate and the confidence in `05_metrics.json` is LOW with reason
   `FORESHORTENED` (`tests/test_s04_s05.py`).
3. Aspect invariance: the same pose rendered at 1920x1080 and 1080x1080 yields
   the same `_deg_2d` within 0.1 deg. This test fails on today's code.
4. Girdle angle: synthetic pose with the bow arm exactly in line with the
   shoulder line gives `shoulder_bow_girdle_deg` = 180 +/- 0.5.
5. Overlay: rendered key frame contains the `2D` suffix and, for a foreshortened
   joint, the `FORESHORTENED` marker (`tests/test_s06_s07.py`).
6. Grounding: `_2d` and plain keys are distinct evidence keys, and a section
   citing one is not auto-linked to the other
   (`tests/test_grounding_linking.py`).
7. Runbook: re-measure S2 wall time with the heavy model and record it in
   `docs/02_RUNBOOK.md` next to the full-model time.

### 1.4 Cost

Heavy is roughly 2 to 3x full on CPU. S2 measured 2.5 min on the Kalpana clip,
so expect 5 to 8 min. Nothing else in the pipeline gets slower. Evidence file
grows by roughly 3 extra keys per joint per phase; `llm.max_evidence_lines`
selection already caps what any one call sees, so prompt size is unchanged.

---

## Part 2 - Feature 2, additional inputs

### 2.1 Free-text block

Entered once at the start of the run, echoed back before analysis:

```
Height: 160 cms
Weight: 63.9 kgs
Age: 20 years
Resting HR: 62 bpm
Average HR: 118 bpm
```

Rules: case-insensitive keys, unit token optional, one field per line, unknown
lines kept verbatim under `notes` and never parsed for numbers. The CLI prints
what it understood and marks every unrecognised field `NOT PROVIDED`; the same
inventory reaches Section 2 via the INPUTS block.

### 2.2 Force plate - what the supplied sample actually contains

Sample read: `Body Sway.csv` (20 trials) and `Body Sway 1.csv` (header + 1 row).

- One row per trial, 146 columns, single header row, units carried in the
  header text in square brackets.
- Columns 1-19 are metadata: `Project, Visit, Visit notes, Date,
  Upper Extremities, Sensory Manipulation, Stance Position, Duration [s],
  Dual Task, Footwear, Repetition, Measurement notes, Subject - ID,
  Subject - Name, Subject - Date of birth, Subject - Sex, Subject - Height [m],
  Subject - Weight [kg], Subject - Notes`.
- Columns 20-146 are measures in five families: global general (sway path,
  sway velocity, amplitudes, areas, 95% prediction ellipse, frequency bands),
  interval-specific (1st/2nd/3rd interval, endurance and fatigue indices),
  structural (diffusion slopes, density plots, recurrence quantification,
  sample entropy), and contralateral (left leg, right leg, left/right ratio).
- **The file holds four athletes** (Ansh Tanwar, Kalpana Ragar, Samarth Kumar,
  Tamanna Verma), 4 to 6 trials each, all on one date. The parser must select
  by athlete, not assume one subject per file.
- Trial conditions present: `Upper Extremities` in {`Free` 50 s,
  `On Hips` 10 s, `DRAW TO HOLD R` 10 s}, all `Open Eyes - No Manipulation`,
  `Parallel Stance`, `No Task`, `Competition Shoes`, repetitions 1-2.
- **Columns 115-116 (sample entropy) and 117-146 (all contralateral) are
  exactly 0.000 in all 20 rows.** That is a single-plate capture, not a real
  zero. Hard rule: a measure that is exactly zero across every trial of a file
  is emitted as unavailable, never as a value. Without this rule the report
  would state perfect left/right symmetry, which is the worst possible failure
  mode for an evidence-based report.
- Parsing must use the `csv` module, not `split(",")`: the two free-text notes
  columns can contain commas. The sample happens not to, so a naive parser
  would pass today and break later.

Derived rules:

- Athlete match on normalised `Subject - Name` against `session.athlete_name`
  (case and whitespace insensitive). No match or more than one match is a hard
  error naming the candidates, never a silent pick.
- `Subject - Height [m]`, `Subject - Weight [kg]`, `Subject - Date of birth`
  and `Subject - Sex` are used to cross-check the free-text entries. A conflict
  is printed and both values are carried into the report as a flagged
  disagreement; the free-text value wins for the run because the operator typed
  it for this session.
- Compare trials only within the same `(Upper Extremities, Sensory
  Manipulation, Stance Position, Duration, Dual Task)` tuple. Kalpana's `Free`
  trial is 50 s and her `DRAW TO HOLD R` trials are 10 s, so their sway paths
  (2035 mm vs 133.6 mm) are not comparable and must never be put in one table.
- Evidence keys: `posture.<CONDITION>.<measure>.<stat>` where CONDITION is the
  slugged tuple, stat is `mean`/`sd`/`n` across repetitions, and `sd` follows
  the existing `stats.min_shots_for_sd` rule.
- Measure whitelist for the evidence file, to keep prompts small. Everything
  else stays in the parsed JSON and is rendered in a table but is not offered
  to the model: sway path total/AP/ML, sway velocity total/AP/ML, average and
  maximal amplitude AP/ML, sway area total, 95% prediction ellipse area, mean
  frequency AP/ML, sway velocity endurance and fatigue index total.
- `.xlsx` accepted through `openpyxl` with the identical column contract.

### 2.3 Force plate supplied as images

Images are attached as figures with a caption and are cited as OBSERVED. **No
number is read out of an image.** OCR of a plot or a screenshot is not evidence
and would break the "Python measures, the model writes" rule the whole pipeline
rests on. If only images are supplied, `posture.*` stays empty and p2 prints
`NOT PROVIDED - CANNOT BE CONFIRMED`.

### 2.4 Heart rate

`hr.*` keys from the free-text summary fields, plus an optional two-column
`time,bpm` CSV which, when present, also yields min/max/mean and time in zone.
Zone boundaries are computed from the age-predicted maximum using the formula
named in `config/benchmarks.json` so the citation travels with the number.
`benchmarks.json` gains its first cited entry for this; every printed boundary
carries the formula name and the word estimate.

### 2.5 Anthropometrics

Exactly three uses, per the agreed scope: age-appropriate training-load
guidance, BMI as a plain computed number, and the age-based maximum-heart-rate
estimate. BMI is rendered as a number with its formula and no category label.

### 2.6 Where this lives in the pipeline

New step **S0P, physio ingest**, run after S0 and before S5, writing
`00b_physio.json`. It is a separate step so that supplying a corrected CSV
re-runs the parse and the report only, and never re-runs pose estimation. Its
cache key covers the input file hashes, the free-text block and
`steps/s0p_physio.py`, matching the existing source-fingerprint scheme. S5
merges its keys into `evidence_index`; S10 renders p1 and p2 from the model
output instead of today's placeholder strings.

Guardrails unchanged: the CSV and any images are opened `O_RDONLY` through
`io_guard`, nothing is written outside the run directory, and the HTML stays
versioned `_vNN`.

### 2.7 CLI surface

Rule that decides the design: the resume cache keys every step on config, session
fields, upstream hashes and source code. Anything typed on the command line must
therefore be written into the session file before a step runs, or the cache would
not see it and a changed input would silently reuse a stale report. So the flags
below are only a way to fill `session.json`; the file stays the single source of
truth, and `run` snapshots the resolved version into the run directory as
`00_session_resolved.json`, which is what gets hashed and what the report's
provenance block prints.

Three ways in, same destination.

**1. Interactive, matches the "type it at the start" model**

```
archery init-session --video "C:\Users\w10\Documents\Testing Ai\Kalapna.mov" --ask
```

Prompts field by field, echoes what it understood, writes the session file and
exits without running anything.

**2. Flags**

```
archery init-session ^
  --video "C:\Users\w10\Documents\Testing Ai\Kalapna.mov" ^
  --athlete "Kalpana Ragar" ^
  --profile "Height: 160 cms; Weight: 63.9 kgs; Age: 20 years; Resting HR: 62 bpm; Average HR: 118 bpm" ^
  --force-plate "C:\Users\w10\Documents\Testing Ai\Body Sway.csv" ^
  --force-plate-image "C:\...\plate_report_p1.png"
```

`--profile` takes the free-text block with `;` standing in for a line break,
because multi-line quoting differs between cmd and PowerShell. `--profile-file`
takes the same text from a file, one field per line, and is the better route for
anything longer than a line. `--force-plate-image` repeats. `--hr-file` takes the
optional `time,bpm` CSV.

**3. Same flags on `run`**

```
archery run --video "...\Kalapna.mov" --profile-file profile.txt --force-plate "...\Body Sway.csv"
```

Updates the session file first, then runs. Without flags, `run` uses whatever the
session file already holds, so the normal second run is just `archery run --video ...`.

**Echo-back, printed before any step starts**

```
INPUTS UNDERSTOOD
  height_cm       160.0    free text
  weight_kg        63.9    free text
  age_y              20    free text
  hr_rest_bpm        62    free text
  hr_mean_bpm       118    free text
  force_plate     Body Sway.csv -> "Kalpana Ragar", 4 trials, 3 conditions
  hr_file         NOT PROVIDED
  plate_images    NOT PROVIDED
CROSS-CHECK AGAINST FORCE PLATE
  height_cm  typed 160.0  file 160.0  agree
  weight_kg  typed  63.9  file  63.9  agree
UNPARSED, kept as notes and never used as evidence
  "morning session, slight headwind"
Proceed? [y/N]
```

`--yes` skips the confirmation for unattended runs. A value outside its
plausible range (height 100-250 cm, weight 20-200 kg, age 5-100 y, heart rate
25-230 bpm) is rejected with the offending line quoted, never clamped and never
silently dropped.

**Schema additions** (`schemas/session.schema.json`): top-level `height_cm` and
`weight_kg` alongside the existing `age`; `inputs_raw` holding the free-text block
verbatim for provenance; `physio` expanded to
`heart_rate {rest_bpm, mean_bpm, max_bpm, file}` and
`force_plate {file, athlete_match, images[]}`.

**Cost of changing an input.** Editing the profile or swapping the CSV changes
only the S0P hash, so S0P, S5, S8, S9 and S10 re-run and S0 to S4, S6 and S7 are
served from cache. Pose estimation and the annotated video are not repeated,
which is the whole reason physio ingest is its own step.

### 2.8 Verification steps for this feature

New file `tests/test_s0p_physio.py`, plus additions where noted.

1. Header contract: the 146 expected column names parse; a missing or renamed
   column fails loudly with the column name, and does not silently shift.
2. Multi-athlete select: the 20-row sample yields 4 trials for Kalpana and none
   from the other three athletes.
3. Ambiguity: two athletes whose names both match raises, naming both.
4. All-zero block: contralateral and sample-entropy keys are absent from
   `evidence_index` and render as unavailable, with a test asserting no
   `posture.*` key holds a value of exactly 0 that came from an all-zero column.
5. Comma in a notes field is parsed correctly (constructed row, quoted field).
6. Condition isolation: a 10 s and a 50 s trial of the same athlete never share
   a mean, and their keys differ.
7. Unit fidelity: `Subject - Height [m]` 1.6 becomes 160 cm exactly once, and
   BMI computed from it matches an independently computed value.
8. Free text: every supported field parses with and without its unit token;
   an unparsed line appears in the inventory as NOT PROVIDED and never as 0.
9. Conflict: free-text weight differing from the CSV weight produces a flagged
   disagreement in the report, not a silent overwrite.
10. Grounding: p1 and p2 outputs pass `check_text` with the new key namespaces,
    and a typed heart-rate number is rejected (`tests/test_grounding_linking.py`).
11. Rule 11 enforcement: a generated p1 containing a medication or condition
    word is rejected by `banned_phrases` (new entries).
12. Absent inputs: a run with no physio inputs at all produces the same report
    as today plus the NOT PROVIDED inventory line, and all 71 existing tests
    still pass.
13. Prompt budget: p1 and p2 prompts stay under `llm.warn_prompt_tokens`
    (`tests/test_prompt_budget.py`).

---

## Part 3 - Open decisions

Code work starts once these are settled.

| # | Decision | Recommendation |
|---|---|---|
| D1 | Heart-rate input format. No sample supplied yet. | Free-text summary fields now, optional `time,bpm` CSV later. Confirm, or send a sample export. |
| D2 | Age-predicted maximum heart rate formula. | Tanaka `208 - 0.7 x age`, printed with the formula name and the word estimate. The alternative is Fox `220 - age`, which is more familiar but less accurate for young athletes. |
| D3 | BMI presentation. | Plain computed number plus formula, no category label and no health wording, per rule 12. |
| D4 | Which force-plate condition is primary. | Report every condition present, and treat `DRAW TO HOLD R` as the archery-specific one that Section 5 and p2 lead with. |
| D5 | Force-plate images. | Figures only, no numbers extracted. See 2.3. |
| D6 | The sample file's athletes are minors in at least two cases, and the CSV carries names and dates of birth. | Store only the matched athlete's row set in the run directory, and print the athlete name but not the date of birth in the HTML. Confirm this is what you want before the parser is written. |

---

## Part 5 - What shipped

Feature 2 is implemented. Feature 1, the angle accuracy work in Part 1, has not
started.

Decisions settled before the build: Tanaka for the age-predicted maximum, free
text only for heart rate, athlete name printed but never a date of birth, and
training-load guidance age-gated below eighteen. Taken as defaults and stated:
BMI as a plain number with its formula, every plate condition reported with
DRAW TO HOLD leading, plate images as figures with no numbers read from them.

New and changed files:

| File | What it does |
|---|---|
| `src/archery/profile_text.py` | Parses the free-text block. Rejects out-of-range values with the line quoted, keeps unmatched lines as notes. |
| `src/archery/forceplate.py` | Reads the .csv or .xlsx export. Column contract, athlete selection, all-zero rule, condition grouping. |
| `src/archery/steps/sp_physio.py` | Step SP. BMI, Tanaka zones, per-condition statistics, evidence keys, the input inventory. |
| `src/archery/cli.py` | `--profile`, `--profile-file`, `--force-plate`, `--force-plate-image`, `--ask`, `--yes`, and the echo-back. |
| `config/benchmarks.json` | First cited entries: Tanaka, the percentage-of-maximum zones, BMI, the youth threshold. |
| `config/banned_phrases.txt` | Clinical and body-composition language, enforced on p1 as it already was on Section 10. |
| `templates/report.html.j2` | p1 and p2 rendered from evidence instead of the placeholder strings. |
| `tests/test_sp_physio.py` | 22 tests against the real export, including five end to end. |

Verification, all passing (92 tests): header contract named by column, four
athletes reduced to one, ambiguous and unknown names refused, channels that are
zero everywhere never rendered as a value, a comma inside a notes field, .csv
and .xlsx reading identically, conditions of different duration never sharing a
key, SD withheld below the trial floor, BMI and Tanaka matching independently
computed values, the estimate emitted at MEDIUM confidence, youth flagged,
typed-versus-export conflicts reported rather than resolved, a run with no
inputs producing no keys and no zeros, images alone producing no numbers, a
broken export failing the step without writing evidence, editing the export
re-running SP, physio edits not invalidating the video steps, the report never
naming the other athletes or printing a date of birth, and no
body-composition language anywhere in the HTML.

Not built, by decision: the `time,bpm` heart-rate CSV reader (D1, free text
first) and any reading of numbers out of a plate image (D5).

## Part 6 - Feature 1, what shipped

Triggered by a real frame: draw elbow reported 109.9 deg on a folded draw arm.

Changed from the spec in Part 1 in one respect. The plan was to add `_2d` keys
beside the existing world-space keys. Instead the plain key IS the image-plane
angle and the world value moved to `<name>_3d`. The reason is that the plain
keys are what the overlay, the evidence file, the section prompts and the
report tables all reference; adding a parallel set would have left the wrong
number as the default everywhere it already appears. The gap key kept its
spec name, `<name>_2d3d_diff`.

| Change | Why |
|---|---|
| Joint angles measured in the image plane, in pixels | The only value that can be checked against the frame. World depth is inferred from one camera and fails worst at full draw. |
| `<name>_3d` and `<name>_2d3d_diff` added | The world value stays available as a cross-check rather than disappearing. |
| `quality_gates.angle_2d3d_disagree_deg: 15` | Above the gap the joint is not square to the camera: S5 marks it FORESHORTENED at LOW, the overlay appends `!`. |
| Image-plane work moved from normalised coordinates to pixels | Second bug, found while fixing the first. Normalised axes scale independently, so on 16:9 x was stretched 1.78x and the shipped trunk, pelvic, shoulder and head tilts were all wrong. |
| S1 records `frame_width` / `frame_height`; S3 refuses to run without them | A defaulted aspect ratio is how a wrong number reaches a report silently. |
| `shoulder_<side>_girdle_deg` added | Arm against the shoulder-girdle line, the angle a coach reads at full draw, alongside the existing arm-against-trunk definition. |
| Pose model switched to `pose_landmarker_heavy.task` | Its weakest endpoints under occlusion are the bow hand and string hand, which is where the wrist and elbow angles are read. |
| Report gains an angle-space table | A foreshortened joint is visible, not merely downgraded. |

Deliberately not changed: distances and speeds stay in normalised units,
because the phase thresholds in `phase_rules.yaml` are tuned against them and
re-tuning is a separate, separately-testable job. One consequence to watch:
`draw_elbow_rate_deg_s` now derives from the image-plane angle, so
`min_elbow_rate_deg_per_s` operates on a different, better-behaved series.

## Change log

- 2026-09-27 Feature 1 implemented after a real frame reproduced the defect.
- 2026-09-27 Feature 2 implemented end to end. Step SP added between S0 and S1.
  Feature 1 not started.
- 2026-09-26 Prompt enhanced (Part 0). Force-plate sample analysed. Spec written.
  No code changed.
