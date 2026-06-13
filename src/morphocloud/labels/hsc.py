"""Hubble Source Catalog v3 galaxy labels (MAST VO TAP).

The only galaxy-label source inside the MCs (LS DR10 and DELVE DR3 are blind
there) and the only one independent of DECam imaging. Labels come from the
concentration index `ci` (small-minus-large aperture magnitude difference,
averaged over detections); `numimages >= 2` (server-side) rejects
single-detection artifacts.

CI was validated against Gaia DR3 stars and LS DR10 galaxies on tiles
hp32-08298 (deep periphery field) and hp32-08329 (SMC interior); measured
distributions in docs/plan.md. What the validation showed:

- Unsaturated stars (G > 18.5) sit in a tight locus: ci p50 ~ 1.0, p95 ~ 1.2,
  so `ci > CI_GALAXY_MIN` leaks ~1% of faint stars or less.
- Saturated bright stars (G < ~18) inflate to ci ~ 1.3-2.5; they carry
  Gaia/DR3 star labels, so the conflicting claims are dropped by assemble.
- HSC *shreds* large galaxies (LS shape_r > 1.5") into point-like knots
  (ci ~ 1.1). Point-like therefore does NOT mean star at faint magnitudes,
  so HSC contributes **no star labels** in v1 and shredded galaxies are lost
  to the ambiguity band (purity over completeness).

HST resolves DELVE blends (0.05" vs ~1" PSF), so matching is many-to-one and
blend-aware — see `assemble._component_flags`.

Known biases: HST pointings oversample clusters/dense fields; shredding makes
the galaxy labels lean compact. Broad-band magnitudes (a_* = ACS,
w2_* = WFPC2, w3_* = WFC3) are carried for diagnostics, validity-range
documentation and the IR-only exclusion — never as model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .core import LabelSource

HSC = LabelSource(
    name="hsc_v3",
    table="dbo.summagaper2catview",
    columns=(
        "matchid", "matchra", "matchdec",
        "ci", "ci_sigma", "numimages", "numfilters",
        "a_f606w", "a_f814w", "w2_f606w", "w2_f814w",
        "w3_f606w", "w3_f814w", "w3_f110w", "w3_f160w",
    ),
    ra_col="matchra",
    dec_col="matchdec",
    where="numimages >= 2",
    service="mast_hsc",
    max_rows=100_000,
)

# Stellar-locus edge (faint, unsaturated stars: ci p95 ~ 1.2). Used only for
# blend bookkeeping in assemble — NOT for star labels, because point-like CI
# at faint magnitudes includes shredded-galaxy knots.
CI_STAR_MAX = 1.2
# Galaxy threshold. Between the two values is unlabelled.
CI_GALAXY_MIN = 1.6
# Crowding guard: stars leak past CI_GALAXY_MIN in dense interior fields
# (2.2% of faint SMC stars vs <1% in the periphery) because neighbor light
# inflates CI — but inconsistently across images (leaked stars: ci_sigma
# p50 = 0.66; real galaxies: 0.08). Requiring consistency keeps 100% of
# LS-confirmed galaxies and halves the worst-case star leak to ~0.9%.
CI_SIGMA_FRAC = 0.2

#: optical broad bands used to exclude IR-only sources: WFC3/IR's broader PSF
#: shifts the CI scale, and the thresholds above were validated on optical
#: detections
OPTICAL_MAGS = ("a_f606w", "a_f814w", "w2_f606w", "w2_f814w",
                "w3_f606w", "w3_f814w")


#: a point-like HSC source closer than this to another HSC detection would be
#: blended in DELVE's ~1" ground PSF, so it is not a clean DECam star label
ISOLATION_ARCSEC = 0.5


def point_source_mask(df: pd.DataFrame) -> pd.Series:
    """Point-like CI (blend bookkeeping only; NaN CI fails both masks)."""
    return (df["ci"] > 0) & (df["ci"] < CI_STAR_MAX)


def isolated_mask(df: pd.DataFrame,
                  radius_arcsec: float = ISOLATION_ARCSEC) -> pd.Series:
    """True for HSC sources with no *other* HSC detection within radius_arcsec.

    HST resolves blends DELVE sees as one source, so a point-like HSC source
    with a close neighbour would be a blended (bad) DECam star label. Computed
    over the whole loaded set with a local tangent-plane KDTree.
    """
    out = pd.Series(False, index=df.index)
    if len(df) < 2:
        return ~out if len(df) == 1 else out
    ra = df["matchra"].to_numpy(); dec = df["matchdec"].to_numpy()
    dec0 = np.radians(np.median(dec))
    tree = cKDTree(np.column_stack([ra * np.cos(dec0), dec]))
    nn, _ = tree.query(np.column_stack([ra * np.cos(dec0), dec]), k=2)
    return pd.Series(nn[:, 1] * 3600.0 >= radius_arcsec, index=df.index)


def star_candidate_mask(df: pd.DataFrame) -> pd.Series:
    """Isolated, point-like HSC sources with an optical detection — candidate
    DECam star labels. Carries ~27% irreducible compact-galaxy contamination
    (validated: not separable by CI or colour), so HSC stars get their own
    provenance flag (HST_STAR) and are ablation-tested / down-weightable.
    """
    optical = df[list(OPTICAL_MAGS)].notna().any(axis=1)
    return point_source_mask(df) & optical & isolated_mask(df)


def galaxy_mask(df: pd.DataFrame) -> pd.Series:
    """High-confidence extended sources with an optical broad-band detection.

    Extension must be consistent across images (the ci_sigma crowding guard);
    NaN ci_sigma fails the cut.
    """
    optical = df[list(OPTICAL_MAGS)].notna().any(axis=1)
    consistent = df["ci_sigma"] < CI_SIGMA_FRAC * df["ci"]
    return (df["ci"] > CI_GALAXY_MIN) & consistent & optical
