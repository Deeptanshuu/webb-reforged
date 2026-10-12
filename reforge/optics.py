"""Lens and film simulation: bloom, anamorphic streak, chromatic aberration, grade, grain."""
from __future__ import annotations

import cv2
import numpy as np

from .sky import blur

LUMA = np.array([0.0722, 0.7152, 0.2126], np.float32)  # Rec.709 weights in BGR order


def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Screen blend: light adds like projected light, never clips harshly."""
    return 1 - (1 - np.clip(a, 0, 1)) * (1 - np.clip(b, 0, 1))


def bloom(canvas: np.ndarray, k: float, threshold: float, weights=(0.28, 0.2, 0.14),
          radii=(12, 50, 180)) -> np.ndarray:
    """Glow from the brightest parts at three radii (tight halo, soft glow, wide haze)."""
    lum = canvas @ LUMA
    hi = canvas * np.clip((lum - threshold) / (1 - threshold), 0, 1)[..., None]
    glow = np.zeros_like(canvas)
    for wt, r in zip(weights, radii):
        glow += wt * blur(hi, r, k)
    return screen(canvas, glow)


def anamorphic_streak(h: int, w: int, k: float, cx: float, cy: float, halo: float = 0.05,
                      falloff: float = 0.2) -> np.ndarray:
    """Thin horizontal flare line through (cx, cy), like a 2x anamorphic lens:
    a hair-thin core, a soft sheath and a faint wide haze, fading toward the frame edges."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dxs = np.abs(xx - cx) / w
    dys = yy - cy
    return (np.exp(-dys ** 2 / (2 * (2.2 * k) ** 2)) * 0.9
            + np.exp(-dys ** 2 / (2 * (14 * k) ** 2)) * 0.25
            + np.exp(-dys ** 2 / (2 * (60 * k) ** 2)) * halo) * np.exp(-dxs / falloff)


def grade(canvas: np.ndarray, gamma: float = 1.04, saturation: float = 1.1) -> np.ndarray:
    """Filmic S-curve (smoothstep blended 40/60 with linear) plus a saturation lift."""
    c = np.clip(canvas, 0, 1) ** gamma
    c = c * c * (3 - 2 * c) * 0.4 + c * 0.6
    lum = (c @ LUMA)[..., None]
    return lum + (c - lum) * saturation


def chromatic_aberration(canvas: np.ndarray, amount: float = 0.001) -> np.ndarray:
    """Scale red out and blue in about the centre: colour fringing grows toward the edges."""
    h, w = canvas.shape[:2]

    def radial_scale(ch, s):
        m = np.array([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]], np.float32)
        return cv2.warpAffine(ch, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    canvas[..., 2] = radial_scale(canvas[..., 2].copy(), 1 + amount)
    canvas[..., 0] = radial_scale(canvas[..., 0].copy(), 1 - amount)
    return canvas


def vignette(canvas: np.ndarray, strength: float = 0.55) -> np.ndarray:
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return canvas * (1 - strength * np.clip((d - 0.55) / 0.9, 0, 1) ** 1.7)[..., None]


def grain(canvas: np.ndarray, rng: np.random.Generator, sigma: float = 0.006) -> np.ndarray:
    h, w = canvas.shape[:2]
    return canvas + rng.normal(0, sigma, (h, w, 1)).astype(np.float32)


def to_uint8(canvas: np.ndarray) -> np.ndarray:
    return (np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8)
