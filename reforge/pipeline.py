"""The preset pipeline: frame -> declutter -> restyle -> sky -> optics -> grade."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import optics, sky
from .declutter import clutter_mask, paint_out, smooth_light
from .optics import LUMA
from .palettes import gradient_map
from .presets import Preset

StageHook = Callable[[str, np.ndarray], None]


def _frame(preset: Preset, src: np.ndarray, w: int, h: int):
    """Scale/rotate the source so preset.centre lands mid-frame.

    Returns the placed image, an alpha of where real pixels landed (feathered, so padded
    "width" fits fade into generated sky) and the affine transform for later lookups.
    """
    sh, sw = src.shape[:2]
    sc = (max(w / sw, h / sh) if preset.fit == "cover" else w / sw) * preset.zoom
    if sc < 0.6:  # pre-shrink big sources with INTER_AREA so the warp doesn't alias
        pre = sc / 0.8
        src = cv2.resize(src, None, fx=pre, fy=pre, interpolation=cv2.INTER_AREA)
        sh, sw = src.shape[:2]
        sc /= pre
    cxs, cys = preset.centre[0] * sw, preset.centre[1] * sh
    M = cv2.getRotationMatrix2D((cxs, cys), preset.angle, sc)
    M[0, 2] += w / 2 - cxs
    M[1, 2] += h / 2 - cys
    img = cv2.warpAffine(src.astype(np.float32) / 255.0, M, (w, h), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REFLECT)
    ey, ex = np.mgrid[0:sh, 0:sw].astype(np.float32)
    e = np.clip(np.minimum(np.minimum(ex, sw - ex), np.minimum(ey, sh - ey)) / (0.04 * max(sw, sh)), 0, 1)
    alpha = cv2.warpAffine(e * e * (3 - 2 * e), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return np.clip(img, 0, 1), alpha, M, (sw, sh, sc)


def restyle(img: np.ndarray, alpha: np.ndarray, smooth_l: np.ndarray, preset: Preset, k: float) -> np.ndarray:
    """Dual gradient map. Luminance picks the position along a palette; the pixel's
    warmth ((R-B)/sum) blends between the cool and the warm palette. Structure and the
    image's own warm/cool separation survive, but every colour now comes from two curated
    ramps instead of the raw filter assignment."""
    lum = img @ LUMA
    sky_level = np.percentile(smooth_l, 8)
    white = np.percentile(lum[alpha > 0.5], 99.95)
    t = np.clip((lum - sky_level * 1.15) / (white - sky_level * 1.15), 0, 1) ** preset.gamma
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    warmth = cv2.GaussianBlur((r - b) / (r + g + b + 0.06), (0, 0), 1.5 * k + 0.3)
    pivot, span = preset.warm
    wv = np.clip((warmth - pivot) / span + 0.5, 0, 1)
    wv = (wv * wv * (3 - 2 * wv))[..., None]
    col = gradient_map(t, preset.cool_pal) * (1 - wv) + gradient_map(t, preset.warm_pal) * wv
    col *= np.clip(t * 4.0, 0, 1)[..., None]     # let the faintest light fall to true black
    return col * alpha[..., None]


def render(preset: Preset, originals: Path, width: int = 5120, on_stage: StageHook | None = None) -> np.ndarray:
    w, h = width, width * 9 // 16
    k = w / 5120.0
    rng = np.random.default_rng(preset.seed)
    stage = on_stage or (lambda name, im: None)

    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}[preset.reduce]
    src = cv2.imread(str(Path(originals) / preset.filename), flag)
    if src is None:
        raise FileNotFoundError(f"{preset.filename} not found in {originals} — run `reforge fetch` first")

    # 1. frame
    img, alpha, M, (sw, sh, sc) = _frame(preset, src, w, h)
    del src
    stage("1-framed", img)

    # 2. declutter
    smooth = smooth_light(img)
    smooth_l = smooth @ LUMA
    protect = []
    for cx, cy, rx, ry in preset.protect:
        c = M @ np.array([cx * sw, cy * sh, 1.0])
        protect.append((c[0], c[1], rx * sw * sc, ry * sh * sc, -preset.angle))
    mask = clutter_mask(img, smooth_l, preset.sky_t, protect)
    if on_stage:
        overlay = img.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.6 + np.array([0.05, 0.05, 0.55], np.float32) * 0.4
        stage("2a-clutter-mask", overlay)
    img = paint_out(img, mask, smooth, k)
    del mask
    stage("2b-decluttered", img)

    # 3. restyle
    col = restyle(img, alpha, smooth_l, preset, k)
    subject = np.clip(col @ LUMA, 0, 1)
    del img
    stage("3-restyled", col)

    # 4. generated sky
    canvas = np.zeros((h, w, 3), np.float32)
    if preset.nebula > 0:
        neb, _ = sky.nebula(h, w, k, preset.neb_cols, seed=preset.seed)
        canvas += neb * preset.nebula
        del neb
    stars = sky.starfield(rng, h, w, k)
    # Hero diffraction stars, only where the subject is dark
    dark = cv2.dilate(cv2.resize(subject, (w // 8, h // 8), interpolation=cv2.INTER_AREA),
                      np.ones((9, 9), np.uint8))
    placed = tries = 0
    while placed < 5 and tries < 400:
        tries += 1
        fx, fy = 0.05 + 0.9 * rng.random(), 0.06 + 0.88 * rng.random()
        if dark[int(fy * (h // 8 - 1)), int(fx * (w // 8 - 1))] > 0.05:
            continue
        b = 0.35 + 0.65 * rng.random() ** 2
        colour = [(1.0, 0.92, 0.80), (0.80, 0.90, 1.0), (1.0, 0.97, 0.92)][placed % 3]
        sky.spike_star(stars, fx * w, fy * h, b, 120 + 160 * b, colour, k)
        placed += 1
    stars *= (1 - np.clip(subject * 2.0, 0, 1))[..., None]   # stars sit *behind* the galaxy
    canvas += stars
    del stars
    stage("4-sky", canvas)

    # 5. composite + optics
    canvas = optics.screen(canvas, col)
    del col
    stage("5-composite", canvas)
    canvas = optics.bloom(canvas, k, threshold=0.8)
    if preset.flare is not None:
        fx, fy = M @ np.array([preset.flare[0] * sw, preset.flare[1] * sh, 1.0])
        tint = gradient_map(np.array([0.5], np.float32), preset.cool_pal)[0]
        tint = tint / max(1e-3, tint.max())
        canvas += optics.anamorphic_streak(h, w, k, float(fx), float(fy))[..., None] * tint * 0.75

    # 6. grade
    canvas = optics.grade(canvas, gamma=1.04, saturation=1.1)
    canvas = optics.chromatic_aberration(canvas, 0.001)
    canvas = optics.vignette(canvas, 0.55)
    canvas = optics.grain(canvas, rng, 0.006)
    out = optics.to_uint8(canvas)
    stage("6-final", out)
    return out
