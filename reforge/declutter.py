"""Remove background galaxies and stray point sources — the "random coloured dots".

Webb deep fields are full of tiny distant galaxies. They are real, but in a wallpaper
they read as noise. We find compact bright blobs with a morphological top-hat (on
luminance *and* on chroma, so saturated dots are caught even when dim) and paint them
out with a smooth, dot-free model of the local light.
"""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .optics import LUMA

DET_W, DET_H = 1920, 1080   # fixed detection scale (see below)


def smooth_light(img: np.ndarray) -> np.ndarray:
    """Dot-free estimate of the large-scale light: median-filter a 1/24 thumbnail and blow it up."""
    h, w = img.shape[:2]
    lo = cv2.resize(img, (max(16, w // 24), max(9, h // 24)), interpolation=cv2.INTER_AREA)
    lo8 = (np.clip(lo, 0, 1) * 255).astype(np.uint8)
    for _ in range(3):
        lo8 = cv2.medianBlur(lo8, 5)
    return cv2.resize(cv2.GaussianBlur(lo8.astype(np.float32) / 255.0, (0, 0), 1.5), (w, h),
                      interpolation=cv2.INTER_CUBIC)


def clutter_mask(img: np.ndarray, smooth_l: np.ndarray, sky_t: float,
                 protect: Sequence[tuple[float, float, float, float, float]] = ()) -> np.ndarray:
    """Return a uint8 mask (full size) of blobs to paint out.

    Detection runs at a fixed 1920x1080 regardless of output size: at 5K, sky noise
    bridges neighbouring blobs into giant connected components that fail the
    "small and compact" test, so nothing gets removed. At 1080p the noise averages out.

    Two rules:
      * anywhere the large-scale light is dim (< sky_t): remove every compact blob;
      * outside ``protect`` ellipses: also remove *strong* compact blobs sitting on bright
        halos. Protect ellipses (cx, cy, rx, ry, angle in canvas pixels) fence off real
        star-forming structure, whose knots look exactly like background galaxies.
    """
    h, w = img.shape[:2]
    kd = DET_W / 5120.0
    det = cv2.resize(img, (DET_W, DET_H), interpolation=cv2.INTER_AREA)
    kk = int(61 * kd) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    top = cv2.morphologyEx(det @ LUMA, cv2.MORPH_TOPHAT, kern)
    ctop = cv2.morphologyEx(det.max(axis=2) - det.min(axis=2), cv2.MORPH_TOPHAT, kern)
    score = np.maximum(top, 1.2 * ctop)
    sl_d = cv2.resize(smooth_l, (DET_W, DET_H), interpolation=cv2.INTER_AREA)
    lim = (110 * kd) ** 2
    mask = np.zeros((DET_H, DET_W), np.uint8)

    def stamp(binary: np.ndarray, max_aspect: float) -> None:
        # Each kept blob gets an ellipse ~1.6x its bounding box, which swallows its halo
        # (a plain dilation leaves ghost rings around bigger galaxies).
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < lim and max(bw, bh) / max(1, min(bw, bh)) < max_aspect:
                cv2.ellipse(mask, (int(x + bw / 2), int(y + bh / 2)),
                            (int(bw * 0.8 + 4), int(bh * 0.8 + 4)), 0, 0, 360, 255, -1)

    stamp((score > 0.022) & (sl_d < sky_t), max_aspect=6)
    if protect:
        prot = np.zeros((DET_H, DET_W), np.uint8)
        for cx, cy, rx, ry, ang in protect:
            cv2.ellipse(prot, (int(cx * DET_W / w), int(cy * DET_H / h)),
                        (int(rx * DET_W / w), int(ry * DET_H / h)), ang, 0, 360, 255, -1)
        stamp((score > 0.1) & (prot == 0), max_aspect=8)
    return (cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) > 40).astype(np.uint8) * 255


def paint_out(img: np.ndarray, mask: np.ndarray, smooth: np.ndarray, k: float) -> np.ndarray:
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 3 * k + 0.5)[..., None]
    return img * (1 - soft) + smooth * soft
