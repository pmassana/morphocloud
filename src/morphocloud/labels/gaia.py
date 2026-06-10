"""Gaia DR3 star (point-source) labels.

The label is morphological — point source vs extended — so QSOs passing the
astrometric quality cuts are acceptable members of the point-source class.
Known bias: Gaia is complete only to G ~ 20.5-21 and favors blue sources.

Fetched from the ESA Gaia archive (switched from the Data Lab mirror
2026-06-10 after Data Lab result downloads degraded to ~20-50 KB/s; this
also spreads the load across archives). Same lowercase column names at both
archives; only the table name differs. ESA specifics (measured 2026-06-10):
async execution is reliable but linear in result size at ~350 rows/s, so

- the point_source_mask cuts run server-side too (`where=`, like LS/DR3):
  raw Gaia is hopeless (5.4M rows in the LMC-center tile alone, 504M south
  of dec -44) and assembly only ever consumes mask-passing rows. The cuts
  drop the LMC-center tile to 2.1M rows. The local mask is still applied
  after the fetch, so cached pre-cut tiles stay valid.
- max_rows keeps each COUNT-verified piece small enough to finish well
  inside tap.ASYNC_DEADLINE_S (300k rows ~ 15 min).
- sync queries are capped at 60 s execution and cannot fetch tiles.
"""

from __future__ import annotations

import pandas as pd

from .core import LabelSource

GAIA = LabelSource(
    name="gaia_dr3",
    table="gaiadr3.gaia_source",
    columns=(
        "source_id", "ra", "dec",
        "parallax", "parallax_error", "pmra", "pmra_error",
        "pmdec", "pmdec_error",
        "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
        "ruwe", "astrometric_excess_noise", "astrometric_excess_noise_sig",
        "ipd_frac_multi_peak",
    ),
    where=(
        "ruwe < 1.4 AND astrometric_excess_noise_sig < 2.0 "
        "AND ipd_frac_multi_peak <= 2 AND phot_g_mean_mag IS NOT NULL"
    ),
    service="esa_gaia",
    max_rows=300_000,
)


def point_source_mask(df: pd.DataFrame) -> pd.Series:
    """High-confidence point sources with clean, single-peaked astrometry."""
    return (
        (df["ruwe"] < 1.4)
        & (df["astrometric_excess_noise_sig"] < 2.0)
        & (df["ipd_frac_multi_peak"] <= 2)
        & df["phot_g_mean_mag"].notna()
    )
