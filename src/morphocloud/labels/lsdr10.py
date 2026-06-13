"""Legacy Surveys DR10 (Tractor) galaxy labels.

Tractor model types: PSF (point source), REX/EXP/DEV/SER (extended). Galaxy
labels use the unambiguous extended types with clean masks. Tractor is
unreliable in the crowded LMC/SMC centers, so these labels concentrate in the
periphery — a documented selection bias.

Near the MCs, DR10 ran no independent detection: tractor rows there are pure
Gaia-forced PSF entries (ref_cat='GE'), verified 2026-06-09. Real detections
exist only in the periphery — see data/labels/coverage/ for the map.

Schema verified live against Data Lab on 2026-06-09.
"""

from __future__ import annotations

import pandas as pd

from .core import LabelSource

LSDR10 = LabelSource(
    name="ls_dr10",
    table="ls_dr10.tractor",
    columns=(
        "ls_id", "ra", "dec", "type", "ref_cat", "brick_primary",
        "maskbits", "fitbits",
        "nobs_g", "nobs_r", "nobs_i", "nobs_z",
        "dered_mag_g", "dered_mag_r", "dered_mag_i", "dered_mag_z",
        "shape_r", "shape_r_ivar",
    ),
    # galaxy label candidates only: full tiles are ~6x larger and 90% of the
    # rows never pass galaxy_mask anyway
    where="type IN ('DEV','EXP','SER') AND brick_primary = 1 AND maskbits = 0",
)

GALAXY_TYPES = ("DEV", "EXP", "SER")

# Star labels from LS DR10's own PSF detections. Gaia-forced rows (ref_cat='GE')
# are excluded server-side: they are Gaia positions re-listed (redundant with the
# Gaia star labels) and dominate the otherwise-blank MC interior. Full depth (no
# mag cut): LS PSF is the deepest star-label source, overtaking Gaia past r~21 and
# DR3 past r~21.5 (see notebooks/hsc_star_colourcolour.ipynb). LS shares DECam
# imaging with DELVE-MC, so these are a deeper *teacher*, not independent truth —
# carry their own provenance flag (LS_STAR) and ablation-test the weight.
LS_STAR = LabelSource(
    name="ls_dr10_star",
    table="ls_dr10.tractor",
    columns=(
        "ls_id", "ra", "dec", "type",
        "dered_mag_g", "dered_mag_r", "dered_mag_i", "dered_mag_z",
        "nobs_g", "nobs_r", "nobs_i", "nobs_z",
    ),
    where=("type='PSF' AND brick_primary=1 AND maskbits=0 AND nobs_r>=2 "
           "AND (ref_cat IS NULL OR ref_cat != 'GE')"),
)

# Artifact-region mask: LS rows whose MASKBITS flags a bright-star/defect
# region. The primary galaxy source above filters maskbits=0, so the masked
# rows it needs are not in its cache — this is a separate fetch of the
# complement. Note: near the MC cores DR10 ran no independent detection
# (Gaia-forced PSF rows only), so this mask is sparse exactly where DELVE-MC
# is densest — a documented coverage limit of the LS-maskbits approach.
#
# DR10 MASKBITS (verified vs legacysurvey.org/dr10/bitmasks, 2026-06-11):
#   0 NPRIMARY  1 BRIGHT  2-4 SATUR_{g,r,z}  5-7 ALLMASK_{g,r,z}
#   8/9 WISEM1/2  10 BAILOUT  11 MEDIUM  12 GALAXY  13 CLUSTER
#   14 SATUR_I  15 ALLMASK_I  16 SUB_BLOB
# Bits that are NOT instrumental artifacts, so must never exclude training
# data (an object can carry these alongside real artifact bits):
#   0  NPRIMARY - brick-edge bookkeeping; set even on brick_primary sources
#   11 MEDIUM   - medium-bright star halo (Gaia G<16); too much footprint for
#                 a mild halo (Pol's call 2026-06-11), not treated as artifact
#   12 GALAXY   - real SGA large galaxy
#   13 CLUSTER  - real globular cluster (e.g. 47 Tuc blankets the SMC tile)
#   16 SUB_BLOB - deblending bookkeeping, not a defect
# Everything else (BRIGHT, SATUR_*, ALLMASK_*, WISEM*, BAILOUT) flags an
# artifact-prone region.
LS_NONARTIFACT_BITS = (1 << 0) | (1 << 11) | (1 << 12) | (1 << 13) | (1 << 16)

LS_MASK = LabelSource(
    name="ls_dr10_mask",
    table="ls_dr10.tractor",
    columns=("ra", "dec", "maskbits"),
    where="maskbits != 0 AND brick_primary = 1",
)


def masked_mask(df: pd.DataFrame) -> pd.Series:
    """LS rows inside an artifact-prone region (any artifact maskbit set)."""
    mb = df["maskbits"].astype("int64")
    return (mb & ~LS_NONARTIFACT_BITS) != 0


def galaxy_mask(df: pd.DataFrame) -> pd.Series:
    """High-confidence extended sources (conservative: REX excluded)."""
    gtype = df["type"].astype(str).str.strip().str.upper()
    nobs = df[["nobs_g", "nobs_r", "nobs_z"]].max(axis=1)
    return (
        gtype.isin(GALAXY_TYPES)
        & (df["brick_primary"] == 1)
        & (df["maskbits"] == 0)
        & (nobs >= 2)
    )


def star_mask(df: pd.DataFrame) -> pd.Series:
    """LS DR10 PSF point sources. brick_primary / maskbits / GE exclusion are
    applied server-side in LS_STAR.where (not stored), so only the checkable
    cuts re-apply here."""
    gtype = df["type"].astype(str).str.strip().str.upper()
    nobs = df[["nobs_g", "nobs_r", "nobs_z"]].max(axis=1)
    return (gtype == "PSF") & (nobs >= 2)
