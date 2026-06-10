"""Model input features from the DELVE-MC catalog columns.

Every feature derives only from the brick object columns and the brick's own
exposure metadata, so inference runs from the DELVE-MC catalog alone (truth
catalogs, including DELVE DR3, are never inputs). Missing values stay NaN —
XGBoost handles them natively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import bricks
from .config import CORE_BANDS

# A_band / E(B-V) for DECam griz: DES DR1 (Abbott et al. 2018) Fitzpatrick-99
# coefficients, the same set the DELVE data releases use with the SFD EBV
# column already present per source
EXTINCTION_COEFF = {"G": 3.186, "R": 2.140, "I": 1.569, "Z": 1.196}

# adjacent colors plus one wide baseline, on extinction-corrected mags
COLORS = (("G", "R"), ("R", "I"), ("I", "Z"), ("G", "I"))

# anchor stars for the per-brick MAG_AUTO zero point (see _concentration):
# bright but unsaturated, pipeline-confident point sources
ANCHOR_PROB_MIN = 0.8
ANCHOR_SHARP_MAX = 0.3
ANCHOR_MAG_RANGE = (16.0, 20.5)
MIN_ANCHOR_STARS = 20

FEATURE_COLUMNS = tuple(
    [f"{b}{suffix}" for b in CORE_BANDS for suffix in ("MAG0", "ERR", "SCATTER")]
    + [f"NDET{b}" for b in CORE_BANDS]
    + [f"{b1}_{b2}" for b1, b2 in COLORS]
    + ["CHI", "SHARP", "PROB", "ELLIPTICITY", "ASEMI", "BSEMI",
       "SEEING", "FWHM_RATIO", "CONC"]
)


def brick_seeing(brickname: str) -> float:
    """Single per-brick seeing in arcsec: median over the core-band medians.

    The coadd morphology mixes exposures of all bands, so one band-agnostic
    number is the right normalizer for the coadd FWHM. NaN if the brick has
    no usable exposure metadata.
    """
    try:
        per_band = bricks.brick_seeing(brickname)
    except FileNotFoundError:
        per_band = {}
    vals = [per_band[b] for b in CORE_BANDS if b in per_band]
    return float(np.median(vals)) if vals else float("nan")


def _concentration(objects: pd.DataFrame) -> pd.Series:
    """MAG_AUTO-vs-PSF concentration, anchored per brick.

    MAG_AUTO is on the coadd's instrumental zero point — offsets of several
    magnitudes that vary brick to brick (verified on y4t2: -4.4 vs -5.9 mag
    on two test bricks) — so the raw MAG_AUTO - PSF difference is not
    comparable across bricks. Per band, the median difference of bright
    point-like anchor stars is subtracted (removes the zero point and the
    mean stellar color per band); the per-source median over bands is then
    re-centered on the anchor stars once more, so the combined statistic is
    ~0 for point sources in every brick regardless of which bands contribute
    per source. Positive values mean flux beyond the PSF fit (extended).
    """
    point_like = (objects["PROB"] >= ANCHOR_PROB_MIN) & (
        objects["SHARP"].abs() <= ANCHOR_SHARP_MAX
    )
    parts = []
    in_range = pd.Series(False, index=objects.index)
    for band in CORE_BANDS:
        diff = objects[f"{band}MAG"] - objects["MAG_AUTO"]
        bright = objects[f"{band}MAG"].between(*ANCHOR_MAG_RANGE)
        anchors = diff[point_like & bright].dropna()
        if len(anchors) < MIN_ANCHOR_STARS:
            continue
        parts.append(diff - anchors.median())
        in_range |= bright
    if not parts:
        return pd.Series(np.nan, index=objects.index)
    conc = pd.concat(parts, axis=1).median(axis=1)
    return conc - conc[point_like & in_range].median()


def brick_features(
    brickname: str, objects: pd.DataFrame | None = None
) -> pd.DataFrame:
    """FEATURE_COLUMNS frame aligned row-by-row with `objects`."""
    if objects is None:
        objects = bricks.read_objects(brickname)
    seeing = brick_seeing(brickname)

    out: dict[str, object] = {}
    for band in CORE_BANDS:
        out[f"{band}MAG0"] = (
            objects[f"{band}MAG"] - EXTINCTION_COEFF[band] * objects["EBV"]
        )
        out[f"{band}ERR"] = objects[f"{band}ERR"]
        out[f"{band}SCATTER"] = objects[f"{band}SCATTER"]
        out[f"NDET{band}"] = objects[f"NDET{band}"]
    for b1, b2 in COLORS:
        out[f"{b1}_{b2}"] = out[f"{b1}MAG0"] - out[f"{b2}MAG0"]
    for col in ("CHI", "SHARP", "PROB", "ELLIPTICITY", "ASEMI", "BSEMI"):
        out[col] = objects[col]
    out["SEEING"] = np.float32(seeing)
    out["FWHM_RATIO"] = objects["FWHM"] / seeing
    out["CONC"] = _concentration(objects)

    return pd.DataFrame(out, index=objects.index)[list(FEATURE_COLUMNS)]
