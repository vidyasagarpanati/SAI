"""Shared annotation renderer for S6 (key frames) and S7 (every video frame).

One renderer, so the annotated images in the report and the annotated video can
never disagree. It implements the master prompt's ANNOTATED REFERENCE FRAME
REQUIREMENTS (A to F) and returns an account of what it drew, so S6 can verify
coverage mechanically instead of trusting that it looks right.

Layout: the video frame is kept intact on the left and a side panel is added on
the right for the measurement table, observation and coaching boxes. Text never
covers the athlete. Joint-angle labels sit on the frame with leader lines,
placed by a collision check that keeps them off every landmark.

Numbers drawn here are passed in by the caller. S6 passes the rounded values
from 05_metrics.json, so the image shows exactly what the evidence file holds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from archery import landmarks as L
from archery.phase_defs import BADGE, COLOR_BGR, DISPLAY

# BGR colours, named as the master prompt names them
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
GREEN = (60, 200, 60)
ORANGE = (0, 150, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 230, 255)
RED = (50, 50, 235)
BLUE = (235, 130, 40)
GRAY = (175, 175, 175)
BG = (25, 25, 25)
BOW_COL, DRAW_COL, LEG_COL = GREEN, ORANGE, BLUE

FONT = cv2.FONT_HERSHEY_SIMPLEX

REQUIRED_KEY_FRAME_ELEMENTS = [
    "landmarks", "landmark_ids", "topology",
    "origin_pelvis", "origin_sg", "origin_head",
    "C1_gravity", "C2_horizontal", "C3_pelvis_plane", "C4_shoulder_girdle",
    "C5_trunk_line", "C6_bow_arm", "C7_draw_arm", "C8_head_plane", "C9_hip_level",
    "joint_angle_labels", "phase_badge", "timestamp", "frame_number", "shot_number",
    "phase_name", "measurement_table", "observation_callout", "coaching_box",
]


@dataclass
class RenderReport:
    drawn: set[str] = field(default_factory=set)
    skipped: dict[str, str] = field(default_factory=dict)
    n_labels: int = 0
    label_joint_overlaps: int = 0

    def did(self, name: str) -> None:
        self.drawn.add(name)
        self.skipped.pop(name, None)

    def skip(self, name: str, why: str) -> None:
        if name not in self.drawn:
            self.skipped[name] = why

    def missing(self, required: list[str]) -> list[str]:
        return [r for r in required if r not in self.drawn and r not in self.skipped]


# ------------------------------------------------------------------ primitives
def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _ok(p) -> bool:
    return p is not None and np.all(np.isfinite(p))


def dashed(img, p1, p2, color, thick=2, dash=14, gap=9, dotted=False):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1:
        return
    d = (p2 - p1) / length
    step = dash + gap
    pos = 0.0
    while pos < length:
        a = p1 + d * pos
        if dotted:
            cv2.circle(img, _pt(a), max(1, thick), color, -1, cv2.LINE_AA)
        else:
            b = p1 + d * min(pos + dash, length)
            cv2.line(img, _pt(a), _pt(b), color, thick, cv2.LINE_AA)
        pos += step


def extended(p1, p2, w: int, h: int):
    """Endpoints of the infinite line through p1,p2, clipped to the image."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    d /= n
    far = (w + h) * 2
    a, b = p1 - d * far, p1 + d * far
    ok, q1, q2 = cv2.clipLine((0, 0, w, h), _pt(a), _pt(b))
    return (q1, q2) if ok else None


def star(img, c, r, color):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.45
        pts.append((c[0] + rad * math.cos(ang), c[1] + rad * math.sin(ang)))
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def diamond(img, c, r, color):
    pts = np.array([(c[0], c[1] - r), (c[0] + r, c[1]), (c[0], c[1] + r), (c[0] - r, c[1])], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def wrap(text: str, width_chars: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------ label layer
class Labels:
    """Collects text labels, places them without collisions, then draws them
    with one semi-transparent background pass."""

    def __init__(self, w: int, h: int, s: float, joints: list[tuple[int, int]]):
        self.w, self.h, self.s = w, h, s
        self.joints = joints
        self.rects: list[tuple[int, int, int, int]] = []
        self.ops: list[dict] = []
        self.overlaps = 0

    def _size(self, text, scale, thick):
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        return tw, th + base

    def _free(self, r, avoid_joints=True):
        x0, y0, x1, y1 = r
        if x0 < 0 or y0 < 0 or x1 > self.w or y1 > self.h:
            return False
        for a in self.rects:
            if not (x1 < a[0] or x0 > a[2] or y1 < a[1] or y0 > a[3]):
                return False
        if avoid_joints:
            m = int(6 * self.s)
            for jx, jy in self.joints:
                if x0 - m <= jx <= x1 + m and y0 - m <= jy <= y1 + m:
                    return False
        return True

    def add(self, anchor, text, color, scale=0.55, thick=1, leader=True, fixed=None):
        scale *= self.s
        thick = max(1, int(round(thick * self.s)))
        tw, th = self._size(text, scale, thick)
        pad = int(4 * self.s)
        bw, bh = tw + 2 * pad, th + 2 * pad
        if fixed is not None:
            rect = (fixed[0], fixed[1], fixed[0] + bw, fixed[1] + bh)
        else:
            ax, ay = _pt(anchor)
            rect = None
            for radius in (38, 64, 96, 132):
                r = radius * self.s
                for ang in (-30, 30, -150, 150, -90, 90, 0, 180):
                    cx = ax + r * math.cos(math.radians(ang))
                    cy = ay + r * math.sin(math.radians(ang))
                    x0 = int(cx - bw / 2 if abs(math.cos(math.radians(ang))) < 0.3
                             else (cx if math.cos(math.radians(ang)) > 0 else cx - bw))
                    y0 = int(cy - bh / 2)
                    cand = (x0, y0, x0 + bw, y0 + bh)
                    if self._free(cand):
                        rect = cand
                        break
                if rect:
                    break
            if rect is None:   # fall back: nearest in-bounds spot, counted as an overlap
                x0 = int(min(max(0, ax + 20 * self.s), self.w - bw))
                y0 = int(min(max(0, ay - bh - 10 * self.s), self.h - bh))
                rect = (x0, y0, x0 + bw, y0 + bh)
                if not self._free(rect):
                    self.overlaps += 1
        self.rects.append(rect)
        self.ops.append({"rect": rect, "text": text, "color": color, "scale": scale,
                         "thick": thick, "pad": pad, "th": th,
                         "anchor": _pt(anchor) if (leader and anchor is not None) else None})

    def draw(self, img, alpha=0.62):
        layer = img.copy()
        for op in self.ops:
            x0, y0, x1, y1 = op["rect"]
            cv2.rectangle(layer, (x0, y0), (x1, y1), BG, -1)
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)
        for op in self.ops:
            x0, y0, x1, y1 = op["rect"]
            if op["anchor"] is not None:
                ax, ay = op["anchor"]
                ex = min(max(ax, x0), x1)
                ey = min(max(ay, y0), y1)
                cv2.line(img, (ax, ay), (ex, ey), op["color"], max(1, op["thick"]), cv2.LINE_AA)
            cv2.putText(img, op["text"], (x0 + op["pad"], y1 - op["pad"] - int(0.25 * op["th"])),
                        FONT, op["scale"], op["color"], op["thick"], cv2.LINE_AA)


def _fmt(v, dec=1, unit=" deg"):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{dec}f}{unit}"


# ------------------------------------------------------------------ main entry
def render(frame: np.ndarray, pts: np.ndarray, vis: np.ndarray, values: dict, *,
           draw_hand: str, shot: int | None, phase: str | None, t_s: float, frame_idx: int,
           mode: str = "key", track_conf: float = 0.5,
           measurements: list[tuple[str, str]] | None = None,
           observation: str | None = None, coaching: str | None = None,
           extras: dict | None = None) -> tuple[np.ndarray, RenderReport]:
    """Annotate one frame.

    pts     (33, 2) pixel coordinates, NaN where undetected
    vis     (33,) landmark confidence
    values  measure name -> number (angles in degrees, distances in shoulder widths)
    mode    "key" draws everything including IDs and the full panel;
            "video" draws a lighter set tuned for motion
    """
    extras = extras or {}
    rep = RenderReport()
    h, w = frame.shape[:2]
    s = h / 1080.0
    img = frame.copy()
    good = np.isfinite(pts).all(axis=1) & (np.nan_to_num(vis, nan=0) >= track_conf)
    present = np.isfinite(pts).all(axis=1)

    def P(i):
        return pts[i] if present[i] else None

    def mid(i, j):
        return (pts[i] + pts[j]) / 2.0 if present[i] and present[j] else None

    ID = L.ID
    bow, drw = L.side(L.bow_side(draw_hand)), L.side(L.draw_side(draw_hand))
    pelvis = mid(ID["LEFT_HIP"], ID["RIGHT_HIP"])
    sg = mid(ID["LEFT_SHOULDER"], ID["RIGHT_SHOULDER"])
    nose = P(ID["NOSE"])
    lw = max(1, int(round(2 * s)))

    # ---- C: reference lines (drawn first, underneath the skeleton) ----------
    if _ok(pelvis):
        dashed(img, (pelvis[0], 0), (pelvis[0], h), MAGENTA, lw)
        rep.did("C1_gravity")
        dashed(img, (0, pelvis[1]), (w, pelvis[1]), CYAN, lw)
        rep.did("C2_horizontal")
        dashed(img, (0, pelvis[1] + 6 * s), (w, pelvis[1] + 6 * s), CYAN, max(1, lw - 1), dotted=True)
        rep.did("C9_hip_level")
    else:
        for c in ("C1_gravity", "C2_horizontal", "C9_hip_level"):
            rep.skip(c, "pelvis landmarks not detected")

    def ext_line(i, j, color, name, why):
        if present[i] and present[j]:
            seg = extended(pts[i], pts[j], w, h)
            if seg:
                cv2.line(img, seg[0], seg[1], color, lw, cv2.LINE_AA)
                rep.did(name)
                return True
        rep.skip(name, why)
        return False

    ext_line(ID["LEFT_HIP"], ID["RIGHT_HIP"], GREEN, "C3_pelvis_plane", "hip landmarks not detected")
    ext_line(ID["LEFT_SHOULDER"], ID["RIGHT_SHOULDER"], ORANGE, "C4_shoulder_girdle",
             "shoulder landmarks not detected")
    if _ok(nose) and _ok(sg) and _ok(pelvis):
        dashed(img, nose, sg, YELLOW, lw)
        dashed(img, sg, pelvis, YELLOW, lw)
        rep.did("C5_trunk_line")
    else:
        rep.skip("C5_trunk_line", "nose, shoulder or hip landmarks not detected")
    if present[ID["LEFT_EAR"]] and present[ID["RIGHT_EAR"]]:
        dashed(img, pts[ID["LEFT_EAR"]], pts[ID["RIGHT_EAR"]], GRAY, lw)
        rep.did("C8_head_plane")
    else:
        rep.skip("C8_head_plane", "ear landmarks not detected")

    # ---- A: skeleton, landmarks, IDs ----------------------------------------
    for a, b in L.CONNECTIONS:
        if present[a] and present[b]:
            col = L.REGION_COLOR_BGR[L.REGION_OF[a]]
            if not (good[a] and good[b]):
                col = tuple(int(c * 0.45) for c in col)
            cv2.line(img, _pt(pts[a]), _pt(pts[b]), col, lw, cv2.LINE_AA)
    rep.did("topology")

    # C6 bow arm, C7 draw arm, emphasised over the skeleton
    if present[bow["shoulder"]] and present[bow["wrist"]]:
        cv2.line(img, _pt(pts[bow["shoulder"]]), _pt(pts[bow["wrist"]]), RED, lw + 1, cv2.LINE_AA)
        rep.did("C6_bow_arm")
    else:
        rep.skip("C6_bow_arm", "bow shoulder or wrist not detected")
    if all(present[drw[k]] for k in ("shoulder", "elbow", "wrist")):
        cv2.polylines(img, [np.array([_pt(pts[drw[k]]) for k in ("shoulder", "elbow", "wrist")])],
                      False, BLUE, lw + 1, cv2.LINE_AA)
        rep.did("C7_draw_arm")
    else:
        rep.skip("C7_draw_arm", "draw shoulder, elbow or wrist not detected")

    r_pt = max(2, int(round(4 * s)))
    id_scale = 0.38 * s
    for i in range(33):
        if not present[i]:
            continue
        col = L.REGION_COLOR_BGR[L.REGION_OF[i]]
        if good[i]:
            cv2.circle(img, _pt(pts[i]), r_pt, col, -1, cv2.LINE_AA)
            cv2.circle(img, _pt(pts[i]), r_pt, WHITE, 1, cv2.LINE_AA)
        else:
            dim = tuple(int(c * 0.45) for c in col)
            cv2.circle(img, _pt(pts[i]), max(1, r_pt - 2), dim, -1, cv2.LINE_AA)
        if mode == "key":
            txt = str(i) if good[i] else f"{i} ({float(np.nan_to_num(vis[i])):.2f})"
            cv2.putText(img, txt, (_pt(pts[i])[0] + r_pt + 1, _pt(pts[i])[1] - r_pt),
                        FONT, id_scale, col if good[i] else GRAY, 1, cv2.LINE_AA)
    rep.did("landmarks")
    if mode == "key":
        rep.did("landmark_ids")

    # ---- B: reference-frame origins -----------------------------------------
    joints_px = [_pt(pts[i]) for i in range(33) if present[i]]
    labels = Labels(w, h, s, joints_px)
    if _ok(pelvis):
        cv2.circle(img, _pt(pelvis), int(9 * s), GREEN, max(2, lw), cv2.LINE_AA)
        labels.add(pelvis, "PELVIS ORIGIN", GREEN, 0.45)
        rep.did("origin_pelvis")
    else:
        rep.skip("origin_pelvis", "hip landmarks not detected")
    if _ok(sg):
        diamond(img, _pt(sg), int(9 * s), ORANGE)
        labels.add(sg, "SG ORIGIN", ORANGE, 0.45)
        rep.did("origin_sg")
    else:
        rep.skip("origin_sg", "shoulder landmarks not detected")
    if _ok(nose):
        star(img, _pt(nose), 11 * s, MAGENTA)
        labels.add(nose, "HEAD / NOSE", MAGENTA, 0.45)
        rep.did("origin_head")
    else:
        rep.skip("origin_head", "nose not detected")

    # reference-line labels with their measured angles
    if mode == "key":
        if _ok(pelvis):
            labels.add(None, "GRAVITY / Z-GLOBAL", MAGENTA, 0.42, leader=False,
                       fixed=(int(pelvis[0] + 6 * s), int(8 * s)))
            labels.add(None, "HORIZONTAL (PERP.)", CYAN, 0.42, leader=False,
                       fixed=(int(8 * s), int(pelvis[1] - 30 * s)))
        if "C3_pelvis_plane" in rep.drawn:
            labels.add(pts[ID["RIGHT_HIP"]], f"PELVIS PLANE {_fmt(values.get('pelvic_tilt_deg'))}", GREEN, 0.45)
        if "C4_shoulder_girdle" in rep.drawn:
            labels.add(pts[ID["RIGHT_SHOULDER"]],
                       f"SHOULDER GIRDLE {_fmt(values.get('shoulder_tilt_deg'))}", ORANGE, 0.45)
        if "C5_trunk_line" in rep.drawn:
            labels.add((sg + pelvis) / 2, f"TRUNK LINE {_fmt(values.get('trunk_inclination_deg'))}", YELLOW, 0.45)
        if "C6_bow_arm" in rep.drawn:
            labels.add((pts[bow["shoulder"]] + pts[bow["wrist"]]) / 2,
                       f"BOW-ARM {_fmt(values.get('elbow_bow_deg'))}", RED, 0.45)
        if "C7_draw_arm" in rep.drawn:
            labels.add(pts[drw["elbow"]], f"DRAW-ARM {_fmt(values.get('elbow_draw_deg'))}", BLUE, 0.45)
        if "C8_head_plane" in rep.drawn:
            labels.add(pts[ID["RIGHT_EAR"]], f"HEAD PLANE {_fmt(values.get('head_tilt_deg'))}", GRAY, 0.45)

    # ---- D: joint-angle labels ---------------------------------------------
    joint_specs = [
        ("SHOULDER", "shoulder", "bow", BOW_COL), ("ELBOW", "elbow", "bow", BOW_COL),
        ("WRIST", "wrist", "bow", BOW_COL),
        ("SHOULDER", "shoulder", "draw", DRAW_COL), ("ELBOW", "elbow", "draw", DRAW_COL),
        ("WRIST", "wrist", "draw", DRAW_COL),
    ]
    n_joint = 0
    for label, joint, role, col in joint_specs:
        lm = (bow if role == "bow" else drw)[joint]
        if present[lm]:
            labels.add(pts[lm], f"{label}: {_fmt(values.get(f'{joint}_{role}_deg'))}", col)
            n_joint += 1
    for label, joint in (("HIP", "hip"), ("KNEE", "knee"), ("ANKLE", "ankle")):
        for lr in ("left", "right"):
            lm = L.side(lr.upper())[joint]
            if present[lm]:
                labels.add(pts[lm], f"{label}: {_fmt(values.get(f'{joint}_{lr}_deg'))}", LEG_COL)
                n_joint += 1
    if n_joint:
        rep.did("joint_angle_labels")
    else:
        rep.skip("joint_angle_labels", "no joint landmarks detected")

    # ---- E: phase-specific annotations -------------------------------------
    if mode == "key" and phase:
        _phase_specific(img, labels, pts, present, values, phase, draw_hand, extras, rep, s, w, h)

    labels.draw(img)
    rep.n_labels = len(labels.ops)
    rep.label_joint_overlaps = labels.overlaps

    # ---- F: phase badge + time/frame/shot ----------------------------------
    _badge(img, phase, shot, t_s, frame_idx, s, rep)

    # ---- side panel -------------------------------------------------------------
    canvas = _panel(img, mode, phase, shot, t_s, frame_idx, measurements or [], observation,
                    coaching, rep, s)
    return canvas, rep


def _phase_specific(img, labels, pts, present, values, phase, draw_hand, extras, rep, s, w, h):
    ID = L.ID
    drw = L.side(L.draw_side(draw_hand))
    bow = L.side(L.bow_side(draw_hand))
    lw = max(1, int(round(2 * s)))
    if phase == "STANCE":
        for i in (ID["LEFT_HEEL"], ID["RIGHT_HEEL"], ID["LEFT_FOOT_INDEX"], ID["RIGHT_FOOT_INDEX"]):
            if present[i]:
                x, y = _pt(pts[i])
                cv2.rectangle(img, (x - int(6 * s), y - int(6 * s)), (x + int(6 * s), y + int(6 * s)), BLUE, lw)
        lh, rh = ID["LEFT_HEEL"], ID["RIGHT_HEEL"]
        if present[lh] and present[rh]:
            cv2.line(img, _pt(pts[lh]), _pt(pts[rh]), WHITE, lw, cv2.LINE_AA)
            labels.add((pts[lh] + pts[rh]) / 2,
                       f"STANCE WIDTH: {_fmt(values.get('stance_width_norm'), 2, ' SW')}", WHITE, 0.45)
        for side, heel, toe in (("L", "LEFT_HEEL", "LEFT_FOOT_INDEX"), ("R", "RIGHT_HEEL", "RIGHT_FOOT_INDEX")):
            if present[ID[heel]] and present[ID[toe]]:
                d = pts[ID[toe]] - pts[ID[heel]]
                ang = math.degrees(math.atan2(-d[1], abs(d[0]) + 1e-9))
                labels.add(pts[ID[toe]], f"FOOT {side} PROGRESSION (image): {ang:.1f} deg", BLUE, 0.42)
        labels.add(None, "WEIGHT DISTRIBUTION: NOT ASSESSABLE FROM VIDEO (needs force data)",
                   GRAY, 0.42, leader=False, fixed=(int(8 * s), int(h - 34 * s)))
        rep.did("E_stance")
    elif phase == "ANCHOR":
        mouth = (pts[ID["MOUTH_LEFT"]] + pts[ID["MOUTH_RIGHT"]]) / 2
        if present[drw["wrist"]] and np.isfinite(mouth).all():
            cv2.line(img, _pt(pts[drw["wrist"]]), _pt(mouth), MAGENTA, lw, cv2.LINE_AA)
            labels.add(mouth, f"ANCHOR DIST: {_fmt(values.get('anchor_distance_norm'), 3, ' SW')}", MAGENTA, 0.45)
        for k, p in enumerate(extras.get("previous_anchor_points", [])):
            if _ok(p):
                x, y = _pt(p)
                d = int(6 * s)
                cv2.line(img, (x - d, y - d), (x + d, y + d), YELLOW, lw)
                cv2.line(img, (x - d, y + d), (x + d, y - d), YELLOW, lw)
                labels.add(p, f"PREV ANCHOR S{extras.get('previous_anchor_shots', [])[k]}", YELLOW, 0.4)
        rep.did("E_anchor")
    elif phase == "AIM":
        trail = np.asarray(extras.get("pelvis_trail", []), float)
        trail = trail[np.isfinite(trail).all(axis=1)] if trail.size else trail
        if len(trail) >= 5:
            mu = trail.mean(axis=0)
            cov = np.cov(trail.T) + np.eye(2) * 1e-6
            vals, vecs = np.linalg.eigh(cov)
            k = math.sqrt(5.991)       # 95% ellipse, chi-square 2 dof
            axes = (max(2, int(k * math.sqrt(vals[1]))), max(2, int(k * math.sqrt(vals[0]))))
            ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
            cv2.ellipse(img, _pt(mu), axes, ang, 0, 360, CYAN, lw, cv2.LINE_AA)
            labels.add(mu, "SWAY ELLIPSE 95% (pelvic origin, this phase)", CYAN, 0.42)
        heel_y = [pts[ID[x]][1] for x in ("LEFT_HEEL", "RIGHT_HEEL") if present[ID[x]]]
        pelvis = (pts[ID["LEFT_HIP"]] + pts[ID["RIGHT_HIP"]]) / 2
        if heel_y and np.isfinite(pelvis).all():
            gy = float(np.mean(heel_y))
            dashed(img, pelvis, (pelvis[0], gy), WHITE, lw)
            cv2.circle(img, _pt((pelvis[0], gy)), int(6 * s), WHITE, lw)
            labels.add((pelvis[0], gy), "COM PROJ (pelvic origin to ground) [INFERRED]", WHITE, 0.42)
        rep.did("E_aim")
    elif phase in ("RELEASE", "FOLLOW_THROUGH"):
        path = np.asarray(extras.get("draw_hand_path", []), float)
        if path.size:
            path = path[np.isfinite(path).all(axis=1)]
        if len(path) >= 2:
            cv2.polylines(img, [path.astype(np.int32)], False, YELLOW, lw, cv2.LINE_AA)
            cv2.arrowedLine(img, _pt(path[0]), _pt(path[-1]), YELLOW, lw, cv2.LINE_AA, tipLength=0.08)
            disp = extras.get("draw_hand_displacement_sw")
            labels.add(path[-1], f"DRAW HAND DISP: {_fmt(disp, 2, ' SW')}", YELLOW, 0.45)
        bpath = np.asarray(extras.get("bow_wrist_path", []), float)
        if bpath.size:
            bpath = bpath[np.isfinite(bpath).all(axis=1)]
        if len(bpath) >= 2:
            cv2.arrowedLine(img, _pt(bpath[0]), _pt(bpath[-1]), RED, lw, cv2.LINE_AA, tipLength=0.15)
            labels.add(bpath[-1], "BOW-ARM TRAJECTORY", RED, 0.42)
        shift = extras.get("sg_shift_sw")
        if shift is not None and present[bow["shoulder"]]:
            labels.add(pts[bow["shoulder"]], f"SG LATERAL SHIFT vs PRE-DRAW: {_fmt(shift, 3, ' SW')}", ORANGE, 0.42)
        rep.did("E_release")


def _badge(img, phase, shot, t_s, frame_idx, s, rep):
    if not phase:
        return
    col = COLOR_BGR.get(phase, (90, 90, 90))
    x0, y0 = int(14 * s), int(14 * s)
    bw, bh = int(250 * s), int(96 * s)
    layer = img.copy()
    cv2.rectangle(layer, (x0, y0), (x0 + bw, y0 + bh), col, -1)
    cv2.addWeighted(layer, 0.78, img, 0.22, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), WHITE, max(1, int(s)))
    cv2.putText(img, f"PHASE {BADGE[phase]}", (x0 + int(14 * s), y0 + int(42 * s)),
                FONT, 1.15 * s, WHITE, max(2, int(3 * s)), cv2.LINE_AA)
    cv2.putText(img, DISPLAY[phase].upper(), (x0 + int(14 * s), y0 + int(78 * s)),
                FONT, 0.62 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    rep.did("phase_badge")
    rep.did("phase_name")
    info = f"SHOT {shot if shot else '-'}   t = {t_s:.3f} s   FRAME {frame_idx}"
    cv2.putText(img, info, (x0, y0 + bh + int(28 * s)), FONT, 0.6 * s, WHITE,
                max(1, int(2 * s)), cv2.LINE_AA)
    rep.did("timestamp")
    rep.did("frame_number")
    rep.did("shot_number")


def _panel(img, mode, phase, shot, t_s, frame_idx, measurements, observation, coaching, rep, s):
    h, w = img.shape[:2]
    pw = max(int(0.46 * h), 380)
    panel = np.full((h, pw, 3), 22, np.uint8)
    x, y = int(18 * s), int(40 * s)
    lh = int(30 * s)
    sc = 0.6 * s
    th = max(1, int(round(1.5 * s)))
    chars = max(24, int(pw / (11.5 * s)))

    def line(text, color=WHITE, scale=sc, bold=False):
        nonlocal y
        cv2.putText(panel, text, (x, y), FONT, scale, color, th + (1 if bold else 0), cv2.LINE_AA)
        y += lh

    # Title on a bar in the phase colour, white text: dark phase colours (dark red,
    # dark green) are unreadable as text on the dark panel.
    title = f"SHOT {shot or '-'}  |  " + (f"PHASE {BADGE[phase]} {DISPLAY[phase].upper()}" if phase else "NO PHASE")
    bar_col = COLOR_BGR.get(phase, (70, 70, 70)) if phase else (70, 70, 70)
    cv2.rectangle(panel, (0, 0), (pw, int(56 * s)), bar_col, -1)
    y = int(38 * s)
    line(title, WHITE, 0.66 * s, True)
    y += int(10 * s)
    line(f"t = {t_s:.3f} s   frame {frame_idx}", GRAY)
    y += int(8 * s)

    line("MEASUREMENT SUMMARY", YELLOW, sc, True)
    for label, value in measurements:
        cv2.putText(panel, label, (x, y), FONT, 0.52 * s, GRAY, th, cv2.LINE_AA)
        (vw, _), _ = cv2.getTextSize(value, FONT, 0.52 * s, th)
        cv2.putText(panel, value, (pw - x - vw, y), FONT, 0.52 * s, WHITE, th, cv2.LINE_AA)
        y += int(26 * s)
    if measurements:
        rep.did("measurement_table")
    else:
        rep.skip("measurement_table", "no measurements supplied")

    if mode == "key":
        for title_, text, name, col in (("OBSERVATION", observation, "observation_callout", CYAN),
                                        ("COACHING IMPLICATION", coaching, "coaching_box", ORANGE)):
            y += int(12 * s)
            line(title_, col, sc, True)
            if text:
                for ln in wrap(text, chars)[:9]:
                    line(ln, WHITE, 0.5 * s)
                rep.did(name)
            else:
                line("(not supplied)", GRAY, 0.5 * s)
                rep.skip(name, "no text supplied")

    y = h - int(118 * s)
    line("LEGEND", GRAY, 0.5 * s, True)
    for txt, col in (("face / upper limb / lower limb", WHITE), ("gravity . horizontal . trunk", MAGENTA),
                     ("bow arm (red)   draw arm (blue)", RED)):
        line(txt, col, 0.45 * s)

    canvas = np.hstack([img, panel])
    ch, cw = canvas.shape[:2]
    if cw % 2 or ch % 2:   # x264 needs even dimensions
        canvas = cv2.copyMakeBorder(canvas, 0, ch % 2, 0, cw % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas
