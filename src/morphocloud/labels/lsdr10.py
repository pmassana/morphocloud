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
