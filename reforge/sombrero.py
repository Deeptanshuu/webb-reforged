"""Hero pipeline for an edge-on disk galaxy (tuned on a 6000x3737 Sombrero frame).

Unlike the preset pipeline, this one *rebuilds* the scene: the galaxy is lifted out of
its frame as a clean, isolated layer, re-coloured, tilted and re-placed on a fully
generated sky — optionally hanging above a planet's horizon.

    clean()   : strip the coloured dots from the disk (texture cloning) and swap the
                surroundings for a smooth halo
    compose() : gradient-map, place, add sky / stars / planet / optics / grade
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from . import optics, sky
from .optics import LUMA
from .palettes import gradient_map

StageHook = Callable[[str, np.ndarray], None]


@dataclass(frozen=True)
class DiskGeometry:
    """Where the galaxy sits, in pixels of a reference frame ``ref_w`` wide."""
    ref_w: float = 6000.0
    core: tuple[float, float] = (3000.0, 1885.0)
    disk_axes: tuple[float, float] = (2330.0, 300.0)    # ellipse hugging the dust ring
    bulge_axes: tuple[float, float] = (560.0, 330.0)
    bulge_offset: float = -40.0                         # bulge centre sits slightly above the disk


@dataclass(frozen=True)
class Variant:
    palette: str
    angle: float
    scale: float
    core_at: tuple[float, float]          # where the core lands, as a fraction of the frame
    neb_cols: tuple
    neb_strength: float
    streak_tint: tuple[float, float, float]
    streak_strength: float
    planet: bool = False


VARIANTS = {
    "inferno": Variant("ember", -9.0, 0.86, (0.5, 0.47), ((0.55, 0.42, 0.05), (0.42, 0.06, 0.30)), 0.5,
                       (1.0, 0.70, 0.30), 0.9),
    "horizon": Variant("glacier", -4.0, 0.80, (0.5, 0.36), ((0.60, 0.35, 0.06), (0.30, 0.10, 0.45)), 0.4,
                       (1.0, 0.80, 0.45), 0.7, planet=True),
}

# Fixed Webb-style hero stars: (x, y, brightness, spike length, RGB)
HERO_STARS = [(0.86, 0.80, 1.0, 260, (1.0, 0.92, 0.80)), (0.13, 0.20, 0.7, 200, (0.80, 0.90, 1.0)),
              (0.74, 0.13, 0.45, 150, (0.85, 0.92, 1.0)), (0.22, 0.83, 0.5, 170, (1.0, 0.95, 0.88)),
              (0.95, 0.38, 0.35, 120, (0.80, 0.88, 1.0))]


def clean(img: np.ndarray, geo: DiskGeometry = DiskGeometry()) -> np.ndarray:
    """Return a float32 BGR galaxy layer with the off-palette dots removed.

    Inside the disk, dots are found by *hue*, not brightness: the disk is purple (green is
    the weakest channel) while the stray cyan/green/yellow sources all have green well
    above min(red, blue). Each dot is repaired by cloning real disk texture from a
    sideways offset (the disk runs horizontally, so the offset lands on matching dust
    lanes), with its low frequencies shifted to agree with a Telea inpaint.
    Outside the disk, everything is replaced by a heavily median-filtered halo.
    """
    h, w = img.shape[:2]
    s = w / geo.ref_w
    f = img.astype(np.float32) / 255.0

    cx, cy = geo.core[0] * s, geo.core[1] * s
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disk = (((xx - cx) / (geo.disk_axes[0] * s)) ** 2 + ((yy - cy) / (geo.disk_axes[1] * s)) ** 2) <= 1
    bulge = (((xx - cx) / (geo.bulge_axes[0] * s)) ** 2
             + ((yy - (cy + geo.bulge_offset * s)) / (geo.bulge_axes[1] * s)) ** 2) <= 1
    region = cv2.GaussianBlur((disk | bulge).astype(np.float32), (0, 0), 45 * s)

    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    offhue = np.clip(g - np.minimum(r, b), 0, 1)
    kk = int(121 * s) | 1
    top = cv2.morphologyEx(offhue, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((top > 0.06).astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        _, _, bw, bh, area = stats[i]
        keep[i] = area < (140 * s) ** 2 and max(bw, bh) / max(1, min(bw, bh)) < 3
    mask = keep[lab].astype(np.uint8) * 255
    d = int(45 * s) | 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))

    small = 0.5 if w > 3000 else 1.0
    img_s = cv2.resize(img, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask, (img_s.shape[1], img_s.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(img_s, mask_s, max(3, int(12 * s * small)), cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0

    low_fill = cv2.GaussianBlur(filled, (0, 0), 18 * s)
    clone = filled.copy()
    nl, lab2, st2, _ = cv2.connectedComponentsWithStats(mask, 8)
    pad = int(30 * s)
    for i in range(1, nl):
        x, y, bw, bh, _ = st2[i]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        for mult in (1.4, -1.4, 2.4, -2.4, 3.4, -3.4):
            D = int(mult * (bw + 2 * pad))
            if 0 <= x0 + D and x1 + D <= w and not mask[y0:y1, x0 + D:x1 + D].any():
                break
        else:
            continue
        patch = f[y0:y1, x0 + D:x1 + D] + (low_fill[y0:y1, x0:x1] - low_fill[y0:y1, x0 + D:x1 + D])
        sel = (lab2[y0:y1, x0:x1] == i)[..., None]
        clone[y0:y1, x0:x1] = np.where(sel, patch, clone[y0:y1, x0:x1])
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 6 * s)[..., None]
    inner = f * (1 - soft) + np.clip(clone, 0, 1) * soft

    lo = cv2.resize(img, (w // 16, h // 16), interpolation=cv2.INTER_AREA)
    for _ in range(3):
        lo = cv2.medianBlur(lo, 9)
    halo = cv2.resize(cv2.GaussianBlur(lo.astype(np.float32) / 255.0, (0, 0), 3), (w, h),
                      interpolation=cv2.INTER_CUBIC)
    rg = region[..., None]
    return np.clip(inner * rg + halo * (1 - rg), 0, 1)


def _planet(canvas, galaxy, xx, yy, w, h, k, core_x):
    """A dark world whose horizon crests 80% down the frame, lit only at the limb."""
    R = 9000 * k
    pcx, pcy = w * 0.42, h * 0.80 + R
    d = np.sqrt((xx - pcx) ** 2 + (yy - pcy) ** 2) - R          # < 0 inside the planet
    body = np.clip(0.5 - d / (1.5 * k), 0, 1)
    canvas *= (1 - body)[..., None]
    galaxy *= (1 - body)[..., None]
    clouds = sky.fbm(h, w, seed=11, base=6, octaves=7)
    lit = np.exp(-np.clip(-d / (650 * k), 0, 1) * 3.0)
    bands = np.clip((clouds - 0.35) / 0.4, 0, 1)
    canvas += ((0.012 + 0.42 * bands[..., None] * lit[..., None])
               * np.array([0.95, 0.72, 0.50], np.float32)) * body[..., None]
    # Atmosphere: a hair-thin bright limb, wide blue scattering above, faint glow below
    limb = np.exp(-(d / (6 * k)) ** 2)
    haze = np.exp(-np.clip(d, 0, None) / (120 * k)) * (d > -40 * k)
    inner = np.exp(np.clip(d, None, 0) / (60 * k)) * (d <= 0)
    face = np.exp(-((xx - core_x) / (w * 0.55)) ** 2)   # brightest beneath the galaxy
    canvas += (limb * 1.4 * face)[..., None] * np.array([0.85, 0.93, 1.0], np.float32)
    canvas += (haze * 0.40 * (0.3 + 0.7 * face))[..., None] * np.array([1.0, 0.55, 0.18], np.float32)
    canvas += (inner * 0.25 * face)[..., None] * np.array([0.35, 0.55, 1.0], np.float32)


def compose(layer: np.ndarray, variant: str = "inferno", width: int = 5120,
            geo: DiskGeometry = DiskGeometry(), on_stage: StageHook | None = None) -> np.ndarray:
    v = VARIANTS[variant]
    w, h = width, width * 9 // 16
    k = w / 5120.0
    rng = np.random.default_rng(42)
    stage = on_stage or (lambda name, im: None)
    s = layer.shape[1] / geo.ref_w

    # Galaxy: stretch luminance off the sky floor, gradient-map, feather the frame edges
    gh, gw = layer.shape[:2]
    gl = np.clip((layer @ LUMA - 0.07) / (0.98 - 0.07), 0, 1)
    ey, ex = np.mgrid[0:gh, 0:gw].astype(np.float32)
    e = np.clip(np.minimum(np.minimum(ex, gw - ex), np.minimum(ey, gh - ey)) / (900 * s), 0, 1)
    gl = (gl * (e * e * (3 - 2 * e))) ** 1.15
    col = gradient_map(gl, v.palette) * np.clip(gl * 4.0, 0, 1)[..., None]
    core = (geo.core[0] * s, geo.core[1] * s)
    M = cv2.getRotationMatrix2D(core, v.angle, v.scale * k / s)
    cx, cy = w * v.core_at[0], h * v.core_at[1]
    M[0, 2] += cx - core[0]
    M[1, 2] += cy - core[1]
    galaxy = np.clip(cv2.warpAffine(col, M, (w, h), flags=cv2.INTER_LANCZOS4, borderValue=0), 0, None)
    gal_lum = np.clip(galaxy @ LUMA, 0, 1)
    stage("3-restyled", galaxy)

    # Sky: faint warm glow around the core, nebula, stars (denser where the gas is)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / w) ** 2)
    canvas = np.array([0.022, 0.010, 0.006], np.float32) * np.exp(-rad * 2.5)[..., None]
    neb, wisps = sky.nebula(h, w, k, v.neb_cols, seed=0)
    canvas = canvas + neb * v.neb_strength
    stars = sky.scatter_stars(rng, h, w, k, int(9000 * k * k) + 1500, 0.7, 0.05, 0.9, 3.0)
    stars += sky.scatter_stars(rng, h, w, k, int(1400 * k * k) + 200, 1.4, 0.4, 3.0, 4.0)
    stars += sky.scatter_stars(rng, h, w, k, int(160 * k * k) + 30, 2.6, 1.0, 6.0, 4.0)
    dense = sky.scatter_stars(rng, h, w, k, int(14000 * k * k) + 2000, 0.6, 0.02, 0.35, 2.0)
    stars += dense * sky.blur(wisps, 30, k)[..., None] * 2.0
    for fx, fy, b, ln, c in HERO_STARS:
        if v.planet and fy > 0.7:
            fy -= 0.45      # keep them above the horizon
        sky.spike_star(stars, fx * w, fy * h, b, ln, c, k)
    stars *= (1 - np.clip(gal_lum * 1.5, 0, 1))[..., None]
    canvas += stars
    if v.planet:
        _planet(canvas, galaxy, xx, yy, w, h, k, cx)
    stage("4-sky", canvas)

    # Composite, optics, grade
    canvas = optics.screen(canvas, galaxy)
    stage("5-composite", canvas)
    canvas = optics.bloom(canvas, k, threshold=0.85, weights=(0.25, 0.18, 0.12))
    streak = optics.anamorphic_streak(h, w, k, cx, cy, halo=0.06, falloff=0.22)
    canvas += streak[..., None] * np.array(v.streak_tint, np.float32) * v.streak_strength
    canvas = optics.grade(canvas, gamma=1.05, saturation=1.12)
    canvas = optics.chromatic_aberration(canvas, 0.0012)
    canvas = optics.vignette(canvas, 0.6)
    canvas = optics.grain(canvas, rng, 0.007)
    out = optics.to_uint8(canvas)
    stage("6-final", out)
    return out
