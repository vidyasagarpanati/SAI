# 04 - Cross-video comparison (`archery compare`)

Status: SPEC. No code written. Blocked on the phase list from the ChatGPT
thread, which may change the taxonomy everything below is built on.

## What it is for

An athlete films on different days. Each video holds a different number of
shots. The coach wants to see, across all of it, which phase in which shot
took how long, and which shots sit outside that athlete's normal.

## Two things that constrain the honest version

**Shot 2 of one video is not shot 2 of another.** There is no shared shot
axis between a two-shot and a three-shot video. Comparison is per-phase across
all shots; shot index is a label, never a join key.

**Timings travel across videos, positions mostly do not.** Phase durations are
in seconds and S0 measures the true frame rate, so they compare cleanly. Joint
angles are image-plane projections and are only comparable when the camera view
matches. Anchor distance is normalised by shoulder width and does travel.

## Decisions

| # | Decision | Chosen |
|---|---|---|
| C1 | Shape | A separate `archery compare` command, read-only over finished runs. Nothing in the per-video pipeline changes. |
| C2 | Which runs | Exactly the run ids passed on the command line. No athlete resolution, no scanning. |
| C3 | Baseline | Leave-one-out: each shot is judged against the mean and SD of all the OTHER shots in the set, so a slow shot cannot inflate the baseline that judges it. |
| C4 | Comparability | Durations plotted across everything. Position charts only within a camera-view group, with the excluded sessions named. |
| C5 | Mixed athletes | Allowed, warned, and every series labelled by athlete. The report states that the baseline is then not one athlete's own history. |
| C6 | Time axis | The order the runs are passed. See the mitigation below. |
| C7 | Narrative | None. Charts and tables only, no model calls, deterministic and seconds to run. |

### C6, and the risk the user accepted

Command-line order was chosen over `session_date`. Nothing then stops a
mis-ordered command producing a backwards trend, so `compare` prints the order
back with each run's `session_date` beside it when the field is set, and raises
a loud warning when the dates disagree with the order given. It reorders
nothing: the order asked for is the order drawn.

## Inputs and outputs

```
archery compare --runs Kalapna__a8964b39 Kalapna__3f21c0de --out kalpana_6wk
archery compare --runs <id> <id> <id> --sd 2.0
```

Reads `runs/<id>/05_metrics.json` and nothing else. Never opens a video, never
writes inside a run directory. 05_metrics.json is already the single frozen
evidence file, so the comparison inherits the rule that a number not in it
cannot appear in a report.

Writes:
- `outputs/Compare_<label>_vNN.html`, versioned like every other report,
  self-contained, charts as inline SVG with no scripts.
- `runs/_compare/<label>/compare_evidence.json`, the numbers behind every mark
  on every chart, so nothing rendered is unsourced.

## What it draws

1. **Shot timeline.** One horizontal bar per shot, segmented by phase, grouped
   by session in the order given. x is seconds from the start of that shot.
   This is the chart that answers the original question directly.
2. **Per-phase duration, session by session.** One small multiple per detected
   phase: a dot per shot, a line through the session means, and a shaded band
   at the leave-one-out mean plus or minus `--sd` (default 2).
3. **Deviation table.** Every shot-and-phase beyond the band, ranked by
   absolute z, with the shot, session, phase, duration, baseline and z.
4. **Position trends.** Only for camera-view groups holding two or more
   sessions, same small-multiple form, over the core aim measures. Groups with
   one session are listed as not comparable rather than drawn.

## Rules that keep it honest

- A phase not detected in a shot is a gap in the chart, never a zero.
- SD is withheld below `stats.min_shots_for_sd`, the existing three-shot rule,
  and the band is then not drawn.
- A phase whose `boundary_confidence` is LOW is drawn hollow, so a deviation
  resting on a soft boundary is visible as such.
- Frame-rate differences between sessions are printed but do not block anything:
  durations are in seconds.
- Runs whose `05_metrics.json` is missing, or which produced a PARTIAL report,
  are named and excluded rather than silently skipped.

## Verification steps

1. Two runs of the same video produce identical durations and an empty
   deviation table.
2. A synthetic set where one shot's AIM phase is stretched by 50% flags exactly
   that shot, and no other.
3. Leave-one-out really is leave-one-out: the flagged shot's own duration is
   absent from the baseline it is compared against.
4. A two-shot set draws no SD band and says why.
5. Mixed camera views: durations chart includes every session, position charts
   exclude the odd one out and name it.
6. Mixed athletes: the warning appears and every series carries the athlete.
7. Passed order disagreeing with `session_date` raises the warning and the
   chart order still matches the command.
8. A missing or PARTIAL run is named and excluded, and compare still produces a
   report from the rest.
9. Every number drawn on a chart is present in `compare_evidence.json`.
10. compare writes nothing inside any run directory, and no source video is
    opened (io_guard whitelist plus a test that asserts no video read).

## Open before implementation

- The phase list from the ChatGPT thread. The pipeline currently detects nine
  phases (stance, pre-draw, draw, anchor, aim, expansion, release,
  follow-through, recovery). If that thread defines a different set or
  different boundaries, it changes S4 and `phase_rules.yaml`, and every chart
  here is drawn on top of it.
