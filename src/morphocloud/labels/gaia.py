"""Gaia DR3 star (point-source) labels.

The label is morphological — point source vs extended — so QSOs passing the
astrometric quality cuts are acceptable members of the point-source class.
Known bias: Gaia is complete only to G ~ 20.5-21 and favors blue sources.

Backend history:
- Data Lab mirror (until 2026-06-10): result downloads degraded to ~20-50 KB/s.
- ESA Gaia archive (2026-06-10): faster, but went into a maintenance/capacity
  outage (503s) the same day and is expected to stay unstable through the
  Gaia DR4 run-up. Kept as ``GAIA_ESA`` for fallback/cross-checks.
- CDS Vizier (primary, 2026-06-10): table ``I/355/gaiadr3`` — the same Gaia
  DR3 source catalogue (identical source_id and values), reachable and fast
  while ESA is down. Vizier renames the columns (Source, RA_ICRS, Plx, Gmag,
  RUWE, epsi, sepsi, IPDfmp…), so the source renames them back to the
  canonical lowercase names before caching and the cut runs server-side on the
  Vizier names. Coordinates are epoch 2016.0, matching ESA ``ra``/``dec``.

Both backends write the same ``gaia_dr3`` cache with identical schema, so the
tiles already fetched from ESA mix with Vizier tiles without conversion.

The point_source_mask cuts run server-side (`where=`): raw Gaia is hopeless
(5.4M rows in the LMC-center tile alone, 504M south of dec -44) and assembly
only consumes mask-passing rows. The cuts drop the LMC-center tile to ~2.1M
rows. The local mask is still applied after the fetch, so cached pre-cut tiles
stay valid. ``max_rows`` keeps each COUNT-verified async piece small; on Vizier
the COUNT is index-fast only via geometry (``geometry=True``), so the tile box
is sent as a CONTAINS/POLYGON predicate.
"""

from __future__ import annotations

import pandas as pd

from .core import LabelSource

#: Vizier I/355 column -> canonical Gaia (ESA/Data Lab) column.
_VIZIER_RENAME = {
    "Source": "source_id",
    "RA_ICRS": "ra",
    "DE_ICRS": "dec",
    "Plx": "parallax",
    "e_Plx": "parallax_error",
    "pmRA": "pmra",
    "e_pmRA": "pmra_error",
    "pmDE": "pmdec",
    "e_pmDE": "pmdec_error",
    "Gmag": "phot_g_mean_mag",
    "BPmag": "phot_bp_mean_mag",
    "RPmag": "phot_rp_mean_mag",
    "RUWE": "ruwe",
    "epsi": "astrometric_excess_noise",
    "sepsi": "astrometric_excess_noise_sig",
    "IPDfmp": "ipd_frac_multi_peak",
}

GAIA = LabelSource(
    name="gaia_dr3",
    table='"I/355/gaiadr3"',
    # fetched in canonical order so the post-rename schema matches the ESA tiles
    columns=tuple(_VIZIER_RENAME),
    ra_col="RA_ICRS",
    dec_col="DE_ICRS",
    where=(
        "RUWE < 1.4 AND sepsi < 2.0 "
        "AND IPDfmp <= 2 AND Gmag IS NOT NULL"
    ),
    service="vizier",
    geometry=True,
    rename=_VIZIER_RENAME,
    id_col="source_id",
    # ESA serves these as float32; match it so all gaia_dr3 tiles share a schema.
    cast={
        "pmdec_error": "float32",
        "phot_g_mean_mag": "float32",
        "phot_bp_mean_mag": "float32",
        "phot_rp_mean_mag": "float32",
        "astrometric_excess_noise": "float32",
        "astrometric_excess_noise_sig": "float32",
    },
    max_rows=500_000,
)

#: ESA Gaia archive backend — same labels, lowercase columns, no rename.
#: Fallback for when Vizier is unavailable or for cross-archive validation.
GAIA_ESA = LabelSource(
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
