"""Gaia DR3 star (point-source) labels.

The label is morphological — point source vs extended — so QSOs passing the
astrometric quality cuts are acceptable members of the point-source class.
Known bias: Gaia is complete only to G ~ 20.5-21 and favors blue sources.
"""

from __future__ import annotations

import pandas as pd

from .core import LabelSource

GAIA = LabelSource(
    name="gaia_dr3",
    table="gaia_dr3.gaia_source",
    columns=(
        "source_id", "ra", "dec",
        "parallax", "parallax_error", "pmra", "pmra_error",
        "pmdec", "pmdec_error",
        "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
        "ruwe", "astrometric_excess_noise", "astrometric_excess_noise_sig",
        "ipd_frac_multi_peak",
    ),
)


def point_source_mask(df: pd.DataFrame) -> pd.Series:
    """High-confidence point sources with clean, single-peaked astrometry."""
    return (
        (df["ruwe"] < 1.4)
        & (df["astrometric_excess_noise_sig"] < 2.0)
        & (df["ipd_frac_multi_peak"] <= 2)
        & df["phot_g_mean_mag"].notna()
    )
