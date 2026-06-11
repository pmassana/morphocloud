"""Gaia DR3 extragalactic candidates (Vizier I/356): galaxy labels, QSO flags.

The galaxy_candidates / qso_candidates tables ('"I/356/galcand"',
'"I/356/qsocand"' at CDS Vizier — same content as the ESA tables) are
completeness-driven and badly contaminated in crowded fields: south of dec
-44.875 the raw tables hold 924k "galaxies" and 3.2M "QSOs" (half the global
qso table in ~10% of the sky), of which 89k and 1.09M sit within 3 deg of the
LMC center — overwhelmingly misclassified Magellanic stars, since DSC did no
sky-position-dependent filtering. Only the release paper's "purer" sub-samples
are used (Gaia Collaboration, Bailer-Jones et al. 2023, A&A 674, A41, Sect. 8
and appendix ADQL); both definitions verified 2026-06-10 to reproduce the
published counts exactly on Vizier (2,891,132 galaxies; 1,942,825 quasars,
~95% purity each):

- galaxies: radius_sersic IS NOT NULL OR classlabel_dsc_joint='galaxy'
  OR vari_best_class_name='GALAXY'
- quasars:  gaia_crf_source=1 OR host_galaxy_flag<6 OR
  classlabel_dsc_joint='quasar' OR vari_best_class_name='AGN'

The paper computes those purities *excluding* "generous regions" around the
MCs — LMC: 9 deg around ICRS (81.3, -68.7); SMC: 6 deg around (16.0, -72.8)
(its appendix ADQL, verbatim) — so the ~95% is unvalidated inside, and the
purer samples are still contaminated there (e.g. RR-Lyrae-classed "quasars";
purer QSO density at the LMC is ~10x the typical sky value). Both masks
therefore exclude the cores; the same circles back the IN_MC_CORE flag in
assemble. Cost is small: galaxies keep 388k of 419k purer southern rows,
QSOs 229k of 271k. The circles cover 16.9% of the footprint bricks, where
labels from crowding-robust sources (Gaia astrometry, HST) still apply.

Roles (locked 2026-06-10, details in docs/plan.md):

- GALCAND → GALAXY labels (galaxy vote in assemble). The only wide-area
  galaxy source inside the MCs (LS DR10 / DELVE DR3 are blind there), though
  thin in the cores even before the circle cut (~700 purer labels in the LMC
  9-deg core) and biased to bright extended galaxies (Gaia G <~ 21).
- QSOCAND → provenance/evaluation flag ONLY, never a label vote. The Tier 1
  target is morphological (point vs extended) and QSOs are point sources:
  86% of purer southern QSOs pass the gaia_dr3 star cut, so they are correct
  members of the point-source class. The flag enables QSO-contamination
  measurements (12k purer QSOs behind the LMC alone) and downstream masking.
  Physical star-vs-extragalactic separation needs variability features and
  is deferred to a future variable-source classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import LabelSource

#: MC exclusion circles from the release paper's appendix ADQL (verbatim):
#: (name, ra, dec, radius), all deg ICRS.
MC_CORES = (
    ("LMC", 81.3, -68.7, 9.0),
    ("SMC", 16.0, -72.8, 6.0),
)

#: Vizier I/356 column -> canonical Gaia (ESA) column, galaxy_candidates.
_GALCAND_RENAME = {
    "Source": "source_id",
    "RA_ICRS": "ra",
    "DE_ICRS": "dec",
    "PGal": "classprob_dsc_combmod_galaxy",
    "PQSO": "classprob_dsc_combmod_quasar",
    "ClassDSCC": "classlabel_dsc",
    "ClassDCSSA": "classlabel_dsc_joint",  # (sic) Vizier's name
    "ClassOA": "classlabel_oa",
    "Class": "vari_best_class_name",
    "z": "redshift_ugc",
    "RadS": "radius_sersic",
    "nS": "n_sersic",
    "FlagSel": "source_selection_flags",
}

#: Vizier I/356 column -> canonical Gaia (ESA) column, qso_candidates.
_QSOCAND_RENAME = {
    "Source": "source_id",
    "RA_ICRS": "ra",
    "DE_ICRS": "dec",
    "PQSO": "classprob_dsc_combmod_quasar",
    "PGal": "classprob_dsc_combmod_galaxy",
    "ClassDSCC": "classlabel_dsc",
    "ClassDCSSA": "classlabel_dsc_joint",
    "ClassOA": "classlabel_oa",
    "Class": "vari_best_class_name",
    "z": "redshift_qsoc",
    "GCS": "gaia_crf_source",
    "ASF": "astrometric_selection_flag",
    "FlagHost": "host_galaxy_flag",
    "HostDet": "host_galaxy_detected",
    "FlagSel": "source_selection_flags",
}

GALCAND = LabelSource(
    name="gaia_galcand",
    table='"I/356/galcand"',
    columns=tuple(_GALCAND_RENAME),
    ra_col="RA_ICRS",
    dec_col="DE_ICRS",
    # the paper's purer-galaxy union; raw tiles are mostly MC contamination
    where="RadS IS NOT NULL OR ClassDCSSA='galaxy' OR Class='GALAXY'",
    service="vizier",
    geometry=True,
    rename=_GALCAND_RENAME,
    id_col="source_id",
    max_rows=500_000,
)

QSOCAND = LabelSource(
    name="gaia_qsocand",
    table='"I/356/qsocand"',
    columns=tuple(_QSOCAND_RENAME),
    ra_col="RA_ICRS",
    dec_col="DE_ICRS",
    # the paper's purer-quasar union
    where="GCS=1 OR ClassDCSSA='quasar' OR Class='AGN' OR FlagHost<6",
    service="vizier",
    geometry=True,
    rename=_QSOCAND_RENAME,
    id_col="source_id",
    max_rows=500_000,
)


def in_mc_core(ra, dec) -> np.ndarray:
    """True inside the release paper's LMC/SMC exclusion circles."""
    ra = np.radians(np.asarray(ra, dtype=float))
    dec = np.radians(np.asarray(dec, dtype=float))
    inside = np.zeros(ra.shape, dtype=bool)
    for _, ra0, dec0, radius in MC_CORES:
        ra0, dec0 = np.radians(ra0), np.radians(dec0)
        cos_sep = (np.sin(dec) * np.sin(dec0)
                   + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0))
        inside |= cos_sep > np.cos(np.radians(radius))
    return inside


def _stripped(col: pd.Series) -> pd.Series:
    """String compare helper; NaN never matches."""
    return col.astype(str).str.strip()


def galaxy_mask(df: pd.DataFrame) -> pd.Series:
    """Purer-galaxy union outside the MC cores (mirrors the server-side cut)."""
    purer = (
        df["radius_sersic"].notna()
        | (_stripped(df["classlabel_dsc_joint"]) == "galaxy")
        | (_stripped(df["vari_best_class_name"]) == "GALAXY")
    )
    return purer & ~in_mc_core(df["ra"], df["dec"])


def qso_mask(df: pd.DataFrame) -> pd.Series:
    """Purer-quasar union outside the MC cores (provenance flag, never a vote)."""
    purer = (
        (df["gaia_crf_source"] == 1)
        | (_stripped(df["classlabel_dsc_joint"]) == "quasar")
        | (_stripped(df["vari_best_class_name"]) == "AGN")
        | (df["host_galaxy_flag"] < 6)
    )
    return purer & ~in_mc_core(df["ra"], df["dec"])
