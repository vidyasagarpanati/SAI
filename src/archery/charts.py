"""Inline SVG charts for the comparison report.

No JavaScript and no plotting dependency: the report has to stay a single
self-contained file that opens anywhere, like every other output here. Hover
is a native SVG <title> on each mark, which browsers render as a tooltip
without a script.

Colour decisions, and why they are what they are.

The shot cycle has nine phases. Nine categorical hues cannot be made
colourblind-safe, and a nine-step single-hue ramp fails its adjacent-lightness
check, so colour does not carry phase identity at all. Each segment is
DIRECTLY LABELLED with the phase badge, a legend lists badge to name in cycle
order, and the two blue steps alternate purely so a boundary is visible. Both
steps were validated against the chart surface in light and dark.

Everything else is one series, so it needs no legend. Flags use the reserved
status colours and always ship with a marker shape and a printed z, never
colour alone.
"""
from __future__ import annotations

import html
import math

# Validated: 2-step ordinal, one hue, monotone lightness, adjacent dL >= 0.06,
# light end clears the surface. Light #86b6ef/#2a78d6, dark #cde2fb/#5598e7.
SEG_A, SEG_B = "seg-a", "seg-b"


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value, places=2) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _ticks(lo: float, hi: float, count: int = 4) -> tuple[list[float], int]:
    """Tick values and the number of decimals they need to stay distinct.
    Printing 0.475 and 0.45 both as "0.5" makes an axis that cannot be read."""
    if hi <= lo:
        return [lo], 2
    raw = (hi - lo) / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    decimals = max(0, -math.floor(math.log10(step)) + (1 if step * 10 % 10 else 0))
    return out, min(decimals, 4)


# --------------------------------------------------------------- shot timeline
def shot_timeline(payload: dict, row_h: int = 26, pad_left: int = 132,
                  width: int = 900) -> str:
    """One bar per shot, segmented by phase. x is seconds from the start of
    that shot, so the bars share an origin and their shapes are comparable even
    though the shots came from different videos at different moments."""
    shots = payload["shots"]
    names = payload["phase_names"]
    badges = payload["phase_badges"]
    order = payload["phase_order"]
    if not shots:
        return '<p class="na">No shots to draw.</p>'

    longest = max((s["total_s"] or 0) for s in shots) or 1.0
    plot_w = width - pad_left - 70
    top = 34
    rows, y = [], top
    last_session = None
    for s in shots:
        if s["session_label"] != last_session:
            rows.append(("header", s, y))
            y += 22
            last_session = s["session_label"]
        rows.append(("shot", s, y))
        y += row_h
    height = y + 16

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="Phase durations for every shot, seconds from the start of the shot">']

    # x axis at the top, so it is visible without scrolling to the bottom
    tick_vals, _dp = _ticks(0, longest)
    for t in tick_vals:
        x = pad_left + plot_w * t / longest
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{height - 14}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{top - 14}" text-anchor="middle">{t:g}s</text>')

    for kind, s, ry in rows:
        if kind == "header":
            date = next((x["session_date"] for x in payload["sessions"]
                         if x["label"] == s["session_label"]), None)
            who = f' &middot; {esc(s["athlete"])}' if len(payload["totals"]["athletes"]) > 1 else ""
            label = f'{esc(s["session_label"])}{" &middot; " + esc(date) if date else ""}{who}'
            out.append(f'<text class="group" x="4" y="{ry + 14}">{label}</text>')
            continue
        out.append(f'<text class="rowlab" x="{pad_left - 8}" y="{ry + 15}" '
                   f'text-anchor="end">{esc(s["label"])}</text>')
        cursor = 0.0
        for i, code in enumerate([c for c in order if c in s["phases"]]):
            p = s["phases"][code]
            dur = p["duration_s"]
            if dur is None:
                continue
            x = pad_left + plot_w * cursor / longest
            w = max(plot_w * dur / longest - 2, 1.0)      # 2px surface gap
            cls = SEG_A if i % 2 == 0 else SEG_B
            soft = " soft" if p.get("confidence") == "LOW" else ""
            out.append(
                f'<g><rect class="{cls}{soft}" x="{x:.1f}" y="{ry + 4}" width="{w:.1f}" '
                f'height="{row_h - 10}" rx="3"/>'
                f'<title>{esc(s["label"])} &#183; {esc(names[code])} &#183; '
                f'{dur:.2f}s &#183; boundary {esc(p.get("confidence") or "n/a")}</title></g>')
            if w > 26:
                out.append(f'<text class="badge {cls}-ink" x="{x + w / 2:.1f}" '
                           f'y="{ry + row_h / 2 + 3:.0f}" text-anchor="middle">'
                           f'{esc(badges[code])}</text>')
            cursor += dur
        if s["total_s"] is not None:
            out.append(f'<text class="total" x="{pad_left + plot_w * cursor / longest + 6:.1f}" '
                       f'y="{ry + 15}">{s["total_s"]:.2f}s</text>')
    out.append("</svg>")
    return "\n".join(out)


def timeline_legend(payload: dict) -> str:
    items = "".join(
        f'<span class="lg"><b>{esc(payload["phase_badges"][c])}</b> {esc(payload["phase_names"][c])}</span>'
        for c in payload["phase_order"])
    return (f'<div class="legend">{items}</div>'
            '<p class="cap">Every bar starts at zero: the x axis is seconds from the start '
            'of that shot, not clock time, so shots filmed weeks apart sit on one '
            'scale. Segments alternate shade only so the boundaries are visible; '
            'the badge on each segment names the phase. A hollow segment rests on a '
            'LOW-confidence boundary. A phase that was not detected is absent from the '
            'bar, never drawn as zero.</p>')


# ------------------------------------------------------- per-phase facet plots
def _facet(title: str, subtitle: str, points: list[dict], flagged: set,
           x_labels: list[str], pooled_mean: float | None, units: str,
           w: int = 290, h: int = 170) -> str:
    """One small multiple: a dot per shot, x by session, y by value."""
    pad_l, pad_b, pad_r = 48, 30, 10
    pad_t = 44 if flagged else 34          # headroom for the flag marker
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    vals = [p["value"] for p in points]
    lo, hi = min(vals), max(vals)

    if hi - lo < 1e-9:
        # Identical in every shot. Drawing three dots on an invented axis would
        # imply a variation that is not there.
        return (f'<svg class="facet flat" viewBox="0 0 {w} {h}" role="img" '
                f'aria-label="{esc(title)}, identical in every shot">'
                f'<text class="facet-title" x="0" y="13">{esc(title)}</text>'
                f'<text class="facet-sub" x="0" y="27">{esc(subtitle)}</text>'
                f'<text class="flatval" x="{w / 2:.0f}" y="{h / 2 + 4:.0f}" '
                f'text-anchor="middle">{_fmt(lo, 3)} {esc(units)}</text>'
                f'<text class="facet-sub" x="{w / 2:.0f}" y="{h / 2 + 24:.0f}" '
                f'text-anchor="middle">identical in all {len(points)} shots, '
                f'no variation to chart</text></svg>')

    if pooled_mean is not None:
        lo, hi = min(lo, pooled_mean), max(hi, pooled_mean)
    span = hi - lo
    lo, hi = lo - span * 0.18, hi + span * 0.18

    def X(i):
        n = max(len(x_labels) - 1, 1)
        return pad_l + (pw * i / n if len(x_labels) > 1 else pw / 2)

    def Y(v):
        return pad_t + ph - ph * (v - lo) / (hi - lo)

    out = [f'<svg class="facet" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">',
           f'<text class="facet-title" x="0" y="13">{esc(title)}</text>',
           f'<text class="facet-sub" x="0" y="27">{esc(subtitle)}</text>']
    tick_vals, dp = _ticks(lo, hi, 3)
    for t in tick_vals:
        y = Y(t)
        out.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{pad_l - 6}" y="{y + 3:.1f}" '
                   f'text-anchor="end">{t:.{dp}f}</text>')
    for i, lab in enumerate(x_labels):
        out.append(f'<text class="tick" x="{X(i):.1f}" y="{h - 10}" text-anchor="middle">{esc(lab)}</text>')
    if pooled_mean is not None:
        out.append(f'<line class="mean" x1="{pad_l}" y1="{Y(pooled_mean):.1f}" '
                   f'x2="{w - pad_r}" y2="{Y(pooled_mean):.1f}"/>')

    by_x: dict[int, list[float]] = {}
    for p in points:
        by_x.setdefault(p["xi"], []).append(p["value"])
    means = [(i, sum(v) / len(v)) for i, v in sorted(by_x.items())]
    if len(means) > 1:
        d = " ".join(f'{"M" if k == 0 else "L"}{X(i):.1f},{Y(m):.1f}'
                     for k, (i, m) in enumerate(means))
        out.append(f'<path class="trend" d="{d}"/>')

    for p in points:
        group = by_x[p["xi"]]
        jitter = 0.0
        if len(group) > 1:
            idx = group.index(p["value"]) if p["value"] in group else 0
            jitter = (idx - (len(group) - 1) / 2) * 7.0
        cx, cy = X(p["xi"]) + jitter, Y(p["value"])
        flag = p["shot"] in flagged
        cls = "dot flag" if flag else "dot"
        out.append(f'<g><circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="5"/>'
                   f'<title>{esc(p["shot"])} &#183; {_fmt(p["value"], 3)} {esc(units)}'
                   f'{" &#183; FLAGGED" if flag else ""}</title></g>')
        if flag:
            ty = max(cy - 13, pad_t - 10)      # never over the panel's own title
            out.append(f'<path class="flagmark" d="M{cx:.1f},{ty:.1f} l4.5,8 l-9,0 z"/>')
    out.append("</svg>")
    return "\n".join(out)


def phase_facets(payload: dict) -> str:
    """The x axis is whichever the comparison is actually about: shots when one
    session was given, sessions when several were."""
    order = payload["phase_order"]
    x_labels = payload["axis_labels"]
    flagged = {(d["shot"], d["phase"]) for d in payload["deviations"]}
    parts = []
    for code in order:
        points = [{"shot": s["label"], "xi": s["xi"],
                   "value": s["phases"][code]["duration_s"]}
                  for s in payload["shots"]
                  if code in s["phases"] and s["phases"][code]["duration_s"] is not None]
        if not points:
            continue
        b = payload["baselines"].get(code, {})
        sub = (f'n={b.get("n")} shots'
               + (", SD withheld" if b.get("sd_withheld") else
                  f', pooled SD {b["sd"]:.3f}s' if b.get("sd") is not None else ""))
        parts.append(_facet(payload["phase_names"][code], sub, points,
                            {p["shot"] for p in points if (p["shot"], code) in flagged},
                            x_labels, b.get("mean"), "s"))
    return f'<div class="facets">{"".join(parts)}</div>' if parts else \
        '<p class="na">No phase was detected in enough shots to chart.</p>'


def position_facets(payload: dict) -> str:
    all_labels = payload["axis_labels"]
    blocks = []
    for measure, spec in payload["positions"].items():
        for view, points in (spec.get("groups") or {}).items():
            # Keep only the columns this group actually occupies, so a group
            # confined to one session does not draw empty columns for the rest.
            cols = sorted({p["xi"] for p in points})
            remap = {c: i for i, c in enumerate(cols)}
            pts = [{"shot": p["shot"], "xi": remap[p["xi"]], "value": p["value"]}
                   for p in points]
            labels = [all_labels[c] for c in cols]
            vals = [p["value"] for p in pts]
            mean = sum(vals) / len(vals)
            units = "SW" if measure.endswith("_norm") else "deg"
            sub = ("scale-normalised, comparable across views"
                   if spec.get("view_free") else f"view: {view}")
            blocks.append(_facet(payload["measure_names"].get(measure, measure),
                                 f"{sub} &#183; n={len(pts)} shots", pts, set(),
                                 labels, mean, units))
    return f'<div class="facets">{"".join(blocks)}</div>' if blocks else ""
