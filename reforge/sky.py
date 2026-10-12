"""Procedural sky: fractal nebulae, a clean starfield and Webb-style diffraction stars.

Sizes are authored for a 5120px-wide frame and scaled by ``k = W / 5120``.
"""
from __future__ import annotations

import cv2
import numpy as np

# Star colours from warm K-type gold to hot blue-white (RGB -> BGR below).
STAR_COLOURS = np.array([[255, 214, 176], [255, 236, 214], [255, 250, 245], [222, 236, 255],
                         [176, 205, 255]], np.float32)[:, ::-1] / 255.0


def blur(a: np.ndarray, sig: float, k: float) -> np.ndarray:
    return cv2.GaussianBlur(a, (0, 0), max(0.3, sig * k))


def fbm(h: int, w: int, octaves: int = 6, base: int = 4, persistence: float = 0.55,
        seed: int = 0) -> np.ndarray:
    """Fractal value noise in 0..1: bicubic-upsampled random grids summed over octaves."""
    r = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw = base * 2 ** o
        gh = max(2, int(gw * 9 / 16))
        acc += amp * cv2.resize(r.random((gh, gw)).astype(np.float32), (w, h),
                                interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
    acc /= tot
    return (acc - acc.min()) / (acc.max() - acc.min())


def nebula(h: int, w: int, k: float, colours, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two domain-warped wisp fields, tinted, with darker dust filaments cut through them.

    Returns (BGR layer, wisp density) so callers can thicken stars where the gas is.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n1 = fbm(h, w, seed=seed + 1)
    n2 = fbm(h, w, seed=seed + 2, base=3)
    wx = (fbm(h, w, seed=seed + 3, base=2, octaves=3) - 0.5) * 260 * k
    wy = (fbm(h, w, seed=seed + 4, base=2, octaves=3) - 0.5) * 260 * k
    mx, my = (xx + wx).astype(np.float32), (yy + wy).astype(np.float32)
    n1 = cv2.remap(n1, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    n2 = cv2.remap(n2, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    w1 = np.clip((n1 - 0.52) / 0.3, 0, 1) ** 2.2
    w2 = np.clip((n2 - 0.55) / 0.3, 0, 1) ** 2.4
    c1, c2 = colours
    layer = w1[..., None] * np.array(c1, np.float32) + w2[..., None] * np.array(c2, np.float32)
    dust = np.clip((fbm(h, w, seed=seed + 7, base=5) - 0.45) / 0.25, 0, 1)
    layer *= (0.35 + 0.65 * dust)[..., None]
    return layer, w1 + w2


def scatter_stars(rng: np.random.Generator, h: int, w: int, k: float, n: int, sig: float,
                  bmin: float, bmax: float, alpha: float) -> np.ndarray:
    """Point stars with a power-law brightness spread, rendered as small Gaussians."""
    layer = np.zeros((h, w, 3), np.float32)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    b = bmin + (bmax - bmin) * rng.random(n) ** alpha
    np.add.at(layer, (ys, xs), STAR_COLOURS[rng.integers(0, len(STAR_COLOURS), n)] * b[:, None])
    return blur(layer, sig, k) * (sig * k) ** 2 * 2 * np.pi


def starfield(rng: np.random.Generator, h: int, w: int, k: float) -> np.ndarray:
    """Three tiers: dust-fine background, mid stars and a sprinkle of bright ones."""
    stars = scatter_stars(rng, h, w, k, int(8000 * k * k) + 1500, 0.7, 0.05, 0.9, 3.0)
    stars += scatter_stars(rng, h, w, k, int(1200 * k * k) + 200, 1.4, 0.4, 3.0, 4.0)
    stars += scatter_stars(rng, h, w, k, int(120 * k * k) + 25, 2.6, 1.0, 5.0, 4.0)
    return stars


def spike_star(stars: np.ndarray, sx: float, sy: float, bright: float, length: float,
               colour, k: float) -> None:
    """Add a Webb diffraction star in place: six spikes from the hexagonal mirror segments
    (at 90/30/150 degrees) plus the faint horizontal pair from the secondary-mirror strut."""
    h, w = stars.shape[:2]
    L = int(length * k * 3)
    x0, x1, y0, y1 = int(max(0, sx - L)), int(min(w, sx + L)), int(max(0, sy - L)), int(min(h, sy + L))
    if x1 <= x0 or y1 <= y0:
        return
    py, px = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = px - sx, py - sy
    acc = np.zeros_like(dx)
    for ang, ln, wt in ((90, 1.0, 1.0), (30, 1.0, 1.0), (150, 1.0, 1.0), (0, 0.45, 0.45)):
        a = np.deg2rad(ang)
        par = dx * np.cos(a) + dy * np.sin(a)
        perp = -dx * np.sin(a) + dy * np.cos(a)
        acc += wt * np.exp(-perp ** 2 / (2 * (1.1 * k) ** 2)) * np.exp(-np.abs(par) / (length * k * ln * 0.35))
    r2 = dx ** 2 + dy ** 2
    acc += 3.0 * np.exp(-r2 / (2 * (3.0 * k) ** 2)) + 0.5 * np.exp(-r2 / (2 * (16 * k) ** 2))
    stars[y0:y1, x0:x1] += acc[..., None] * np.array(colour, np.float32)[::-1] * bright
