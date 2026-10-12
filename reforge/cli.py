"""reforge — command-line entry point."""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .presets import PRESETS


def _stage_writer(folder: Path, preview_w: int = 1600):
    folder.mkdir(parents=True, exist_ok=True)

    def hook(name: str, im: np.ndarray) -> None:
        if im.dtype != np.uint8:
            im = (np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)
        h, w = im.shape[:2]
        im = cv2.resize(im, (preview_w, int(h * preview_w / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return hook


def _names(arg: list[str]) -> list[str]:
    if not arg or arg == ["all"]:
        return list(PRESETS)
    unknown = [n for n in arg if n not in PRESETS]
    if unknown:
        sys.exit(f"unknown preset(s): {', '.join(unknown)} — see `reforge list`")
    return arg


def cmd_list(_):
    for p in PRESETS.values():
        print(f"{p.name:28s} {p.title}\n{'':28s} source: {p.url}\n")


def cmd_fetch(a):
    out = Path(a.originals)
    out.mkdir(parents=True, exist_ok=True)
    for name in _names(a.presets):
        p = PRESETS[name]
        dest = out / p.filename
        if dest.exists():
            print(f"✓ {p.filename} already present")
            continue
        print(f"↓ {p.url}")
        req = urllib.request.Request(p.url, headers={"User-Agent": f"webb-reforged/{__version__}"})
        with urllib.request.urlopen(req) as r, open(dest.with_suffix(".part"), "wb") as fh:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while chunk := r.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        dest.with_suffix(".part").rename(dest)
        print()


def cmd_render(a):
    from .pipeline import render

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_5k" if a.width == 5120 else f"_{a.width}"
    for name in _names(a.presets):
        t0 = time.time()
        hook = _stage_writer(Path(a.stages) / name) if a.stages else None
        img = render(PRESETS[name], Path(a.originals), a.width, hook)
        dest = out / f"{name}{suffix}.jpg"
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✓ {dest}  ({time.time() - t0:.1f}s)")


def cmd_sombrero(a):
    from .sombrero import clean, compose

    src = cv2.imread(a.input, cv2.IMREAD_COLOR)
    if src is None:
        sys.exit(f"could not read {a.input}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    layer = clean(src)
    print(f"✓ cleaned galaxy layer ({time.time() - t0:.1f}s)")
    suffix = "_5k" if a.width == 5120 else f"_{a.width}"
    for variant in (["inferno", "horizon"] if a.variant == "both" else [a.variant]):
        hook = _stage_writer(Path(a.stages) / f"sombrero_{variant}") if a.stages else None
        if hook:
            hook("1-framed", src.astype(np.float32) / 255.0)
            hook("2b-decluttered", layer)
        img = compose(layer, variant, a.width, on_stage=hook)
        dest = out / f"sombrero_{variant}{suffix}.jpg"
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✓ {dest}")


def cmd_palettes(a):
    from .palettes import swatch_sheet

    cv2.imwrite(a.out, swatch_sheet())
    print(f"✓ {a.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="reforge", description="Cinematic wallpapers from James Webb images.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(required=True)

    s = sub.add_parser("list", help="show the presets")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("fetch", help="download source images from esawebb.org")
    s.add_argument("presets", nargs="*", help="preset names (default: all)")
    s.add_argument("--originals", default="originals")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("render", help="render presets to wallpapers")
    s.add_argument("presets", nargs="*", help="preset names (default: all)")
    s.add_argument("--width", type=int, default=5120, help="output width; height is 16:9 (default 5120)")
    s.add_argument("--out", default="out")
    s.add_argument("--originals", default="originals")
    s.add_argument("--stages", help="also dump each pipeline stage into this folder")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("sombrero", help="hero pipeline for an edge-on disk galaxy image")
    s.add_argument("input")
    s.add_argument("--variant", choices=["inferno", "horizon", "both"], default="both")
    s.add_argument("--width", type=int, default=5120)
    s.add_argument("--out", default="out")
    s.add_argument("--stages")
    s.set_defaults(fn=cmd_sombrero)

    s = sub.add_parser("palettes", help="render the palette swatch sheet")
    s.add_argument("--out", default="palettes.png")
    s.set_defaults(fn=cmd_palettes)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
