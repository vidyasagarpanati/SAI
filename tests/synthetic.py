"""Synthetic archer generator.

Builds a physically plausible 33-landmark sequence for a right-draw archer seen
from the side, so S3, S4 and S5 can be exercised without a video. Used by the
tests and by ``archery selftest``.

The shot cycle it produces:
    0.0-1.0 s  stance, bow arm down
    1.0-1.6 s  setup, bow arm raises
    1.6-2.4 s  draw, draw wrist travels to the face
    2.4-3.2 s  anchor and aim, held steady
    3.2-3.5 s  expansion, draw elbow closes slightly
    3.5 s      release, draw wrist snaps back
    3.5-4.1 s  follow-through
    4.1-5.0 s  recovery, bow arm drops
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from archery.landmarks import ID

FPS = 60.0
DURATION_S = 5.0


def _ease(t: float) -> float:
    return float(np.clip(t, 0.0, 1.0) ** 2 * (3 - 2 * np.clip(t, 0.0, 1.0)))


def build(seed: int = 7, noise: float = 0.0015, hook: bool = False,
          letdown: bool = False, flicker_s: tuple[float, float] | None = None,
          dropout_s: tuple[float, float] | None = None, soft_release: bool = False
          ) -> tuple[pd.DataFrame, dict]:
    """``hook`` adds hooking and gripping: from 0.4 s to 1.0 s the draw hand rests
    on the string beside the bow hand, then rises with it into setup.

    ``letdown`` replaces the release with a let-down: from 3.5 s the draw hand
    eases back off the face over a second with no snap, and the bow comes down.

    ``flicker_s`` throws the draw wrist off the face for that window during the
    hold, the way a tracking glitch does, without the archer moving.

    ``dropout_s`` loses the draw hand for that window (visibility near zero),
    the way a hand blurred or hidden behind the head is lost on real footage.

    ``soft_release`` is the release as a smoothed low-frame-rate clip shows it:
    from a still hold the hand moves a short way back along the jaw and stays
    there, a peak far below the snapping release above."""
    rng = np.random.default_rng(seed)
    n = int(FPS * DURATION_S)
    t = np.arange(n) / FPS
    pts = np.zeros((n, 33, 2))

    # Static skeleton in normalised image space. Archer faces left, camera side-on.
    base = {
        "NOSE": (0.50, 0.170), "LEFT_EYE_INNER": (0.487, 0.163), "LEFT_EYE": (0.482, 0.162),
        "LEFT_EYE_OUTER": (0.477, 0.162), "RIGHT_EYE_INNER": (0.513, 0.163),
        "RIGHT_EYE": (0.518, 0.162), "RIGHT_EYE_OUTER": (0.523, 0.162),
        "LEFT_EAR": (0.470, 0.175), "RIGHT_EAR": (0.530, 0.175),
        "MOUTH_LEFT": (0.492, 0.196), "MOUTH_RIGHT": (0.508, 0.196),
        "LEFT_SHOULDER": (0.455, 0.275), "RIGHT_SHOULDER": (0.545, 0.275),
        "LEFT_HIP": (0.470, 0.520), "RIGHT_HIP": (0.530, 0.520),
        "LEFT_KNEE": (0.468, 0.710), "RIGHT_KNEE": (0.532, 0.710),
        "LEFT_ANKLE": (0.466, 0.900), "RIGHT_ANKLE": (0.534, 0.900),
        "LEFT_HEEL": (0.462, 0.918), "RIGHT_HEEL": (0.538, 0.918),
        "LEFT_FOOT_INDEX": (0.492, 0.930), "RIGHT_FOOT_INDEX": (0.568, 0.930),
    }
    for name, (x, y) in base.items():
        pts[:, ID[name]] = (x, y)

    # Bow arm is the LEFT arm for a right-draw archer. It rises during setup,
    # holds extended, and drops during recovery.
    elev = np.zeros(n)
    for i, ti in enumerate(t):
        if ti < 1.0:
            elev[i] = 0.0
        elif ti < 1.6:
            elev[i] = _ease((ti - 1.0) / 0.6)
        elif ti < 4.1:
            elev[i] = 1.0
        else:
            elev[i] = 1.0 - _ease((ti - 4.1) / 0.9)

    if letdown:
        # Bow comes down with the let-down instead of holding a follow-through.
        for i, ti in enumerate(t):
            if ti >= 3.5:
                elev[i] = 1.0 - _ease((ti - 3.5) / 1.0)

    sh_l = np.array(base["LEFT_SHOULDER"])
    down_wrist = sh_l + np.array([-0.02, 0.30])
    up_wrist = sh_l + np.array([-0.24, -0.01])
    pts[:, ID["LEFT_WRIST"]] = down_wrist + (up_wrist - down_wrist) * elev[:, None]
    pts[:, ID["LEFT_ELBOW"]] = (sh_l + pts[:, ID["LEFT_WRIST"]]) / 2.0 + np.array([0.005, 0.012])

    # Draw arm is the RIGHT arm. anchor_progress goes 0 (bow-side) to 1 (at face).
    anchor_p = np.zeros(n)
    for i, ti in enumerate(t):
        if ti < 1.6:
            anchor_p[i] = 0.0
        elif ti < 2.4:
            anchor_p[i] = _ease((ti - 1.6) / 0.8)
        elif ti < 3.5:
            anchor_p[i] = 1.0
        elif letdown:
            anchor_p[i] = 1.0 - _ease((ti - 3.5) / 1.0)
        elif soft_release:
            anchor_p[i] = 1.0 - 0.45 * _ease((ti - 3.5) / 0.25)
        elif ti < 3.6:
            anchor_p[i] = 1.0 - _ease((ti - 3.5) / 0.1) * 1.6
        else:
            anchor_p[i] = -0.6 + 0.6 * _ease((ti - 3.6) / 1.4)

    mouth = (np.array(base["MOUTH_LEFT"]) + np.array(base["MOUTH_RIGHT"])) / 2.0
    start_wrist = np.array(base["RIGHT_SHOULDER"]) + np.array([-0.20, -0.005])
    anchor_wrist = mouth + np.array([0.030, 0.006])
    pts[:, ID["RIGHT_WRIST"]] = (
        start_wrist + (anchor_wrist - start_wrist) * anchor_p[:, None])
    if hook:
        at_string = down_wrist + np.array([0.03, -0.02])
        w = np.array([_ease((ti - 0.2) / 0.2) if ti < 1.0 else 1.0 - _ease((ti - 1.0) / 0.6)
                      for ti in t])
        w[t >= 1.6] = 0.0
        pts[:, ID["RIGHT_WRIST"]] = (pts[:, ID["RIGHT_WRIST"]] * (1 - w[:, None])
                                     + at_string * w[:, None])

    if flicker_s:
        glitch = (t >= flicker_s[0]) & (t < flicker_s[1])
        pts[glitch, ID["RIGHT_WRIST"]] = start_wrist

    sh_r = np.array(base["RIGHT_SHOULDER"])
    elbow_lift = 0.06 * anchor_p
    pts[:, ID["RIGHT_ELBOW"]] = (
        np.array([sh_r[0] + 0.085, sh_r[1] - 0.010])[None, :]
        * np.ones((n, 1)) - np.array([[0.0, 1.0]]) * elbow_lift[:, None])

    # Expansion: the draw elbow closes a little between 3.2 s and release.
    exp_mask = (t >= 3.2) & (t < 3.5)
    pts[exp_mask, ID["RIGHT_ELBOW"], 0] += 0.012 * _vec_ease((t[exp_mask] - 3.2) / 0.3)

    for prefix in ("LEFT", "RIGHT"):
        w = pts[:, ID[f"{prefix}_WRIST"]]
        e = pts[:, ID[f"{prefix}_ELBOW"]]
        direction = w - e
        norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
        unit = direction / norm
        pts[:, ID[f"{prefix}_INDEX"]] = w + unit * 0.030
        pts[:, ID[f"{prefix}_PINKY"]] = w + unit * 0.026 + np.array([0.0, 0.008])
        pts[:, ID[f"{prefix}_THUMB"]] = w + unit * 0.022 - np.array([0.0, 0.008])

    pts += rng.normal(0.0, noise, pts.shape)

    # World landmarks: hip-centred metric space, roughly 1.7 m tall.
    hip_mid = (pts[:, ID["LEFT_HIP"]] + pts[:, ID["RIGHT_HIP"]]) / 2.0
    world = np.zeros((n, 33, 3))
    world[:, :, 0] = (pts[:, :, 0] - hip_mid[:, None, 0]) * 1.9
    world[:, :, 1] = (pts[:, :, 1] - hip_mid[:, None, 1]) * 1.9
    world[:, :, 2] = rng.normal(0.0, 0.004, (n, 33))

    visibility = np.full((n, 33), 0.93)
    visibility[:, [ID["LEFT_EAR"]]] = 0.55        # far-side ear, partially occluded
    visibility[:, [ID["LEFT_FOOT_INDEX"]]] = 0.62
    if dropout_s:
        lost = (t >= dropout_s[0]) & (t < dropout_s[1])
        for name in ("RIGHT_WRIST", "RIGHT_INDEX", "RIGHT_PINKY", "RIGHT_THUMB"):
            visibility[lost, ID[name]] = 0.05

    frame_idx = np.repeat(np.arange(n, dtype=np.int32), 33)
    lm_idx = np.tile(np.arange(33, dtype=np.int16), n)
    df = pd.DataFrame({
        "frame": frame_idx,
        "t_ms": (frame_idx * (1000.0 / FPS)).astype(np.float32),
        "lm": lm_idx,
        "x": pts[:, :, 0].reshape(-1), "y": pts[:, :, 1].reshape(-1),
        "z": np.zeros(n * 33),
        "visibility": visibility.reshape(-1),
        "presence": visibility.reshape(-1),
        "wx": world[:, :, 0].reshape(-1), "wy": world[:, :, 1].reshape(-1),
        "wz": world[:, :, 2].reshape(-1),
    })

    truth = {
        "fps": FPS, "n_frames": n,
        "phases_s": {
            "stance": [0.0, 1.0], "setup": [1.0, 1.6], "draw": [1.6, 2.4],
            "anchor_aim": [2.4, 3.2], "expansion": [3.2, 3.5],
            "release_s": 3.5, "follow_through": [3.5, 4.1], "recovery": [4.1, 5.0],
        },
        "draw_hand": "right",
    }
    return df, truth


def _vec_ease(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x ** 2 * (3 - 2 * x)


def build_multi(n_shots: int = 3, seed: int = 11, letdowns: tuple[int, ...] = ()
                ) -> tuple[pd.DataFrame, dict]:
    """``n_shots`` consecutive shot cycles with slightly different noise per shot,
    so cross-shot statistics have real (small) variation to measure. Cycles whose
    0-based index is in ``letdowns`` end in a let-down instead of a release."""
    parts, per = [], None
    for k in range(n_shots):
        df, truth = build(seed=seed + k, noise=0.0015 + 0.0004 * k, letdown=k in letdowns)
        per = truth["n_frames"]
        df = df.copy()
        df["frame"] = df["frame"] + k * per
        df["t_ms"] = (df["frame"] * (1000.0 / FPS)).astype(np.float32)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    truth = {"fps": FPS, "n_frames": per * n_shots, "n_shots": n_shots,
             "release_s": [3.5 + k * DURATION_S for k in range(n_shots)],
             "draw_hand": "right"}
    return out, truth
