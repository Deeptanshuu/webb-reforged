"""Gradient-map palettes.

Every palette runs from black, through its colour, to white. Mapping luminance through
it repaints an image in one coherent hue family while keeping all of its tonal detail,
which is what kills the "random confetti" look of raw Webb composites.
"""
from __future__ import annotations

import numpy as np

Stops = list[tuple[float, tuple[int, int, int]]]  # (position 0..1, RGB 0..255)

PALETTES: dict[str, Stops] = {
    "fire": [(0, (0, 0, 0)), (0.05, (20, 4, 6)), (0.14, (110, 14, 20)), (0.26, (210, 50, 25)),
             (0.42, (255, 120, 40)), (0.60, (255, 190, 90)), (0.80, (255, 236, 190)), (1, (255, 255, 255))],
    "electric": [(0, (0, 0, 0)), (0.05, (3, 8, 28)), (0.14, (10, 40, 110)), (0.28, (30, 110, 210)),
                 (0.45, (90, 200, 255)), (0.65, (190, 240, 255)), (1, (255, 255, 255))],
    "magenta": [(0, (0, 0, 0)), (0.05, (22, 3, 26)), (0.14, (100, 10, 90)), (0.26, (200, 25, 140)),
                (0.42, (255, 70, 170)), (0.60, (255, 150, 200)), (0.80, (255, 220, 235)), (1, (255, 255, 255))],
    "cyan": [(0, (0, 0, 0)), (0.05, (0, 14, 22)), (0.14, (0, 60, 80)), (0.28, (0, 150, 170)),
             (0.45, (60, 230, 230)), (0.65, (190, 255, 250)), (1, (255, 255, 255))],
    "gold": [(0, (0, 0, 0)), (0.05, (18, 10, 2)), (0.14, (80, 45, 8)), (0.26, (170, 100, 20)),
             (0.42, (240, 170, 50)), (0.60, (255, 215, 120)), (0.80, (255, 242, 205)), (1, (255, 255, 255))],
    "sapphire": [(0, (0, 0, 0)), (0.05, (6, 4, 26)), (0.14, (30, 20, 100)), (0.28, (70, 60, 190)),
                 (0.45, (130, 140, 245)), (0.65, (210, 215, 255)), (1, (255, 255, 255))],
    "lava": [(0, (0, 0, 0)), (0.05, (24, 2, 2)), (0.14, (120, 10, 8)), (0.26, (220, 50, 10)),
             (0.42, (255, 110, 20)), (0.60, (255, 175, 70)), (0.80, (255, 230, 170)), (1, (255, 255, 255))],
    "violet": [(0, (0, 0, 0)), (0.05, (12, 2, 26)), (0.14, (55, 12, 110)), (0.28, (120, 50, 200)),
               (0.45, (180, 120, 255)), (0.65, (225, 200, 255)), (1, (255, 255, 255))],
    "aurora": [(0, (0, 0, 0)), (0.05, (0, 18, 10)), (0.14, (0, 80, 50)), (0.26, (10, 170, 100)),
               (0.42, (60, 240, 160)), (0.60, (160, 255, 210)), (0.80, (225, 255, 240)), (1, (255, 255, 255))],
    "indigo": [(0, (0, 0, 0)), (0.05, (8, 3, 24)), (0.14, (35, 15, 95)), (0.28, (75, 40, 170)),
               (0.45, (120, 90, 230)), (0.65, (200, 185, 255)), (1, (255, 255, 255))],
    "crimson": [(0, (0, 0, 0)), (0.05, (22, 0, 4)), (0.14, (100, 0, 18)), (0.26, (190, 10, 35)),
                (0.42, (245, 50, 60)), (0.60, (255, 130, 120)), (0.80, (255, 215, 205)), (1, (255, 255, 255))],
    "frost": [(0, (0, 0, 0)), (0.05, (2, 10, 18)), (0.14, (10, 45, 75)), (0.28, (35, 105, 150)),
              (0.45, (100, 175, 215)), (0.65, (190, 230, 245)), (1, (255, 255, 255))],
    "silver": [(0, (0, 0, 0)), (0.05, (6, 8, 12)), (0.14, (35, 45, 60)), (0.28, (90, 110, 135)),
               (0.45, (160, 185, 210)), (0.65, (220, 235, 245)), (1, (255, 255, 255))],
    # Single-palette maps used by the Sombrero hero pipeline
    "ember": [(0.00, (0, 0, 0)), (0.04, (14, 8, 40)), (0.10, (70, 20, 110)), (0.18, (190, 40, 100)),
              (0.28, (245, 90, 45)), (0.42, (255, 150, 60)), (0.60, (255, 205, 120)),
              (0.80, (255, 240, 210)), (1.0, (255, 255, 255))],
    "glacier": [(0.00, (0, 0, 0)), (0.04, (4, 12, 32)), (0.10, (12, 40, 90)), (0.18, (30, 100, 170)),
                (0.28, (90, 180, 230)), (0.42, (190, 230, 245)), (0.60, (255, 225, 180)),
                (0.80, (255, 240, 215)), (1.0, (255, 255, 255))],
}


def gradient_map(t: np.ndarray, palette: str | Stops) -> np.ndarray:
    """Map a 0..1 array through a palette. Returns float32 BGR in 0..1 (OpenCV order)."""
    stops = PALETTES[palette] if isinstance(palette, str) else palette
    xs = np.array([p for p, _ in stops], np.float32)
    cols = np.array([c for _, c in stops], np.float32)[:, ::-1] / 255.0
    out = np.empty(t.shape + (3,), np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(t, xs, cols[:, ch])
    return out


def swatch_sheet(names: list[str] | None = None, width: int = 1600, row_h: int = 44) -> np.ndarray:
    """Render the palettes as labelled gradient bars (BGR uint8) for the README."""
    import cv2

    names = names or list(PALETTES)
    label_w = 190
    sheet = np.full((row_h * len(names) + 16, width, 3), 13, np.uint8)
    ramp = np.linspace(0, 1, width - label_w - 24, dtype=np.float32)
    for i, name in enumerate(names):
        y = 8 + i * row_h
        bar = (gradient_map(ramp, name) * 255).astype(np.uint8)
        sheet[y + 6:y + row_h - 6, label_w:label_w + bar.shape[0]] = bar[None, :, :]
        cv2.putText(sheet, name, (18, y + row_h // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (225, 225, 225), 1, cv2.LINE_AA)
    return sheet
