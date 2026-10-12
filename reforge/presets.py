"""One preset per wallpaper: where to frame the source, how to recolour it, what to add."""
from __future__ import annotations

from dataclasses import dataclass

ESA_WEBB = "https://esawebb.org/media/archives/images/large/{id}.jpg"


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    source_id: str                      # esawebb.org image id; file is originals/<id>.jpg
    credit: str
    # --- framing -------------------------------------------------------------------
    fit: str = "cover"                  # "cover" fills the frame, "width" pads with generated sky
    zoom: float = 1.0
    centre: tuple[float, float] = (0.5, 0.5)   # source point that lands mid-frame (0..1)
    angle: float = 0.0                  # degrees, counter-clockwise
    reduce: int = 1                     # JPEG decode reduction (1, 2, 4) for huge sources
    # --- declutter -----------------------------------------------------------------
    sky_t: float = 0.1                  # compact blobs on light below this are background clutter
    protect: tuple[tuple[float, float, float, float], ...] = ()  # (cx, cy, rx, ry) keep-out ellipses
    # --- restyle -------------------------------------------------------------------
    cool_pal: str = "electric"
    warm_pal: str = "fire"
    warm: tuple[float, float] = (0.1, 0.35)    # (pivot, span) of the warm/cool hue split
    gamma: float = 1.1
    # --- sky + optics --------------------------------------------------------------
    nebula: float = 0.3
    neb_cols: tuple = ((0.5, 0.25, 0.05), (0.08, 0.1, 0.5))   # BGR
    flare: tuple[float, float] | None = None   # source point the anamorphic streak crosses
    seed: int = 0

    @property
    def url(self) -> str:
        return ESA_WEBB.format(id=self.source_id)

    @property
    def filename(self) -> str:
        return f"{self.source_id}.jpg"


PRESETS: dict[str, Preset] = {p.name: p for p in [
    Preset(
        name="m82_solar_flare", title="Solar Flare — Cigar Galaxy (M82)", source_id="weic2612a",
        credit="NASA, ESA, CSA, A. Smercina (STScI), T. Williams (University of Manchester), A. Pagan (STScI)",
        reduce=2, centre=(0.525, 0.498), angle=-7, sky_t=0.10,
        cool_pal="electric", warm_pal="fire", warm=(0.12, 0.35), gamma=1.2,
        nebula=0.25, neb_cols=((0.5, 0.25, 0.05), (0.08, 0.1, 0.5)), flare=(0.525, 0.498), seed=3),
    Preset(
        name="phantom_neon_vortex", title="Neon Vortex — Phantom Galaxy (M74)", source_id="potm2410a",
        credit="ESA/Webb, NASA & CSA, A. Adamo (Stockholm University) and the FEAST JWST team",
        centre=(0.495, 0.49), sky_t=0.06,
        cool_pal="cyan", warm_pal="magenta", warm=(0.15, 0.4), gamma=1.15,
        nebula=0.0, flare=(0.495, 0.49), seed=4),
    Preset(
        name="arp107_gold_leaf", title="Gold Leaf — Arp 107", source_id="weic2423a",
        credit="NASA, ESA, CSA, STScI",
        zoom=0.9, centre=(0.57, 0.46), angle=4, sky_t=0.14,
        protect=((0.60, 0.36, 0.135, 0.38), (0.386, 0.525, 0.05, 0.11)),
        cool_pal="sapphire", warm_pal="gold", warm=(0.1, 0.35), gamma=1.2,
        nebula=0.35, neb_cols=((0.45, 0.15, 0.1), (0.05, 0.2, 0.35)), flare=(0.386, 0.525), seed=5),
    Preset(
        name="twin_spirals_molten", title="Molten — IC 2163 & NGC 2207", source_id="weic2426a",
        credit="NASA, ESA, CSA, STScI",
        fit="width", sky_t=0.08,
        cool_pal="violet", warm_pal="lava", warm=(0.12, 0.35), gamma=1.05,
        nebula=0.3, neb_cols=((0.35, 0.08, 0.3), (0.05, 0.12, 0.4)), flare=(0.655, 0.505), seed=6),
    Preset(
        name="penguin_aurora", title="Aurora — the Penguin & the Egg (Arp 142)", source_id="weic2420a",
        credit="NASA, ESA, CSA, STScI",
        centre=(0.5, 0.545), sky_t=0.24,
        protect=((0.435, 0.54, 0.25, 0.235), (0.21, 0.615, 0.07, 0.06)),
        cool_pal="indigo", warm_pal="aurora", warm=(0.08, 0.35), gamma=1.3,
        nebula=0.3, neb_cols=((0.3, 0.1, 0.4), (0.25, 0.35, 0.02)), flare=(0.21, 0.615), seed=7),
    Preset(
        name="centaurus_crimson_eclipse", title="Crimson Eclipse — Centaurus A", source_id="weic2615a",
        credit="NASA, ESA, CSA, STScI, A. Pagan (STScI), J. DePasquale (STScI), M. García Marín (ESA/STScI)",
        fit="width", sky_t=0.12,
        cool_pal="frost", warm_pal="crimson", warm=(0.05, 0.3), gamma=1.7,
        nebula=0.45, neb_cols=((0.3, 0.12, 0.08), (0.05, 0.02, 0.35)), flare=(0.504, 0.397), seed=8),
]}
