"""DELVE DR3 (DESDM) teacher labels from spread_model.

These distill another morphology classifier, not ground truth: only tight
two-sided high-S/N cuts are used, the ambiguous band stays unlabelled, and
the labels carry their own provenance flag for ablations. The classic
community star cut is |spread_model| < 0.003 + spreaderr_model; our cuts are
deliberately stricter on both sides. DR3 columns are never model inputs.

DELVE DR3 excludes the inner MC region entirely (verified 2026-06-09: zero
sources at the LMC/SMC centers and at the LMC outskirt test point) — see
data/labels/coverage/ for the per-brick map.

Schema verified live against Data Lab on 2026-06-09.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import LabelSource

BAND_PRIORITY = ("i", "r", "g", "z")

DELVEDR3 = LabelSource(
    name="delve_dr3",
    table="delve_dr3.coadd_objects",
    columns=(
        "coadd_object_id", "ra", "dec",
        "spread_model_g", "spreaderr_model_g",
        "spread_model_r", "spreaderr_model_r",
        "spread_model_i", "spreaderr_model_i",
        "spread_model_z", "spreaderr_model_z",
        "wavg_mag_psf_g", "wavg_mag_psf_r", "wavg_mag_psf_i", "wavg_mag_psf_z",
        "flags_gold",
    ),
    # teacher-label candidates only: require a clean source with at least one
    # well-measured spread_model (NaN comparisons are false in postgres)
    where=(
        "flags_gold = 0 AND (spreaderr_model_i < 0.003 OR "
        "spreaderr_model_r < 0.003 OR spreaderr_model_g < 0.003 OR "
        "spreaderr_model_z < 0.003)"
    ),
)

# two-sided teacher cuts (stricter than the |sm| < 0.003 + sme community cut)
STAR_MAX = 0.003     # |spread_model| + 3*spreaderr_model < STAR_MAX
GALAXY_MIN = 0.005   # spread_model - 3*spreaderr_model > GALAXY_MIN
MAX_SPREADERR = 0.003  # require a meaningful spread_model measurement


def _best_band(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """spread_model/spreaderr from the first well-measured band per source."""
    sm = pd.Series(np.nan, index=df.index)
    sme = pd.Series(np.nan, index=df.index)
    for band in BAND_PRIORITY:
        s, e = df[f"spread_model_{band}"], df[f"spreaderr_model_{band}"]
        ok = sm.isna() & s.notna() & e.notna() & (e > 0) & (e < MAX_SPREADERR)
        sm[ok], sme[ok] = s[ok], e[ok]
    return sm, sme


def star_mask(df: pd.DataFrame) -> pd.Series:
    sm, sme = _best_band(df)
    return ((sm.abs() + 3 * sme) < STAR_MAX) & (df["flags_gold"] == 0)


def galaxy_mask(df: pd.DataFrame) -> pd.Series:
    sm, sme = _best_band(df)
    return ((sm - 3 * sme) > GALAXY_MIN) & (df["flags_gold"] == 0)


def community_star_cut(df: pd.DataFrame) -> pd.Series:
    """The classic baseline cut, for evaluation comparisons."""
    sm, sme = _best_band(df)
    return sm.abs() < (0.003 + sme)
