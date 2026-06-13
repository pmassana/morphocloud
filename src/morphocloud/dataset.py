"""Labelled training-set assembly: quality cuts, label merge, spatial splits.

A per-brick worker joins truth labels (labels.assemble) with model features
(features.brick_features), applies the quality and label-noise cuts, and tags
every row with a deterministic train/val/test split assigned to whole sky
regions (HEALPix superpixels) so the splits never share a neighborhood —
no spatial leakage.
"""

from __future__ import annotations

import functools
import hashlib

import healpy as hp
import numpy as np
import pandas as pd

from . import bricks, features
from .labels import assemble

# split unit: nside=16 nest superpixels (~13.4 deg^2, ~210 bricks each) —
# far larger than any plausible PSF/crowding correlation scale, small enough
# for ~165 units over the footprint so the fractions are well sampled
SPLIT_NSIDE = 16
SPLIT_FRACTIONS = (("train", 0.70), ("val", 0.15), ("test", 0.15))

# quality cuts: detected with a sane error in >= 2 of g/r/i
QUALITY_BANDS = ("G", "R", "I")
MAX_BAND_ERR = 0.5
MIN_GOOD_BANDS = 2

# HST label-noise guard: in bricks where more than this fraction of
# HST_GALAXY claims collide with star labels (crowded MC-bar pointings,
# ~50% purity below the Gaia limit in the worst fields), galaxy labels
# backed only by HST are dropped
MAX_HST_CONFLICT = 0.2

ID_COLUMNS = ("BRICKNAME", "OBJID", "RA", "DEC")


def quality_mask(objects: pd.DataFrame) -> pd.Series:
    """Sources detected with a sane error in >= MIN_GOOD_BANDS of g/r/i."""
    good = sum(
        ((objects[f"NDET{b}"] > 0) & (objects[f"{b}ERR"] < MAX_BAND_ERR)).astype(int)
        for b in QUALITY_BANDS
    )
    return good >= MIN_GOOD_BANDS


@functools.lru_cache(maxsize=None)
def _pixel_split(pix: int) -> str:
    """Deterministic split for one superpixel (stable across runs/machines)."""
    digest = hashlib.md5(f"morphocloud-split:{pix}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    acc = 0.0
    for name, frac in SPLIT_FRACTIONS:
        acc += frac
        if u < acc:
            return name
    return SPLIT_FRACTIONS[-1][0]


def brick_split(brickname: str) -> str:
    """Split of a brick: the superpixel of its center decides the whole brick."""
    row = bricks.load_brick_list().set_index("BRICKNAME").loc[brickname]
    pix = hp.ang2pix(SPLIT_NSIDE, row["RA"], row["DEC"], nest=True, lonlat=True)
    return _pixel_split(int(pix))


def brick_dataset(
    brickname: str, max_hst_conflict: float = MAX_HST_CONFLICT
) -> pd.DataFrame:
    """Labelled, quality-cut feature rows for one brick.

    Columns: ID_COLUMNS, SPLIT, LABEL (1=star, 0=galaxy), provenance flags,
    IN_MC_CORE (for split evaluation of the crowded MC cores),
    HST_CONFLICT_RATE (per-brick label-noise monitor), FEATURE_COLUMNS.
    Conflicting and unlabelled sources are excluded; needed label tiles are
    fetched on demand if not cached.
    """
    objects = bricks.read_objects(brickname)
    labels = assemble.brick_labels(brickname, objects)
    feats = features.brick_features(brickname, objects)

    claims = labels["HST_GALAXY"]
    star_vote = (labels["GAIA_STAR"] | labels["DR3_STAR"]
                 | labels["LS_STAR"] | labels["HST_STAR"])
    conflict_rate = (
        float((claims & star_vote).sum() / claims.sum()) if claims.any() else 0.0
    )

    keep = (
        quality_mask(objects)
        & labels["LABEL"].isin([assemble.STAR, assemble.GALAXY])
        # drop objects in LS-flagged artifact regions (bright-star ghosts,
        # saturation, bad pixels): not valid star/galaxy training examples
        & ~labels["IN_ARTIFACT_MASK"]
    )
    if conflict_rate > max_hst_conflict:
        hst_only = (
            labels["HST_GALAXY"] & ~labels["LS_GALAXY"] & ~labels["DR3_GALAXY"]
        )
        keep &= ~hst_only

    out = pd.concat(
        [
            objects.loc[keep, list(ID_COLUMNS)],
            labels.loc[keep, ["LABEL", *assemble.PROVENANCE_COLUMNS, "IN_MC_CORE"]],
            feats.loc[keep],
        ],
        axis=1,
    ).reset_index(drop=True)
    out["LABEL"] = out["LABEL"].astype(np.int8)
    out.insert(4, "SPLIT", brick_split(brickname))
    out.insert(5, "HST_CONFLICT_RATE", np.float32(conflict_rate))
    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split size, label balance, and mag/color/seeing distributions."""
    rows = []
    for split, g in df.groupby("SPLIT"):
        row = {
            "SPLIT": split,
            "N": len(g),
            "N_BRICKS": g["BRICKNAME"].nunique(),
            "F_STAR": float((g["LABEL"] == assemble.STAR).mean()),
        }
        for col in ("RMAG0", "G_R", "SEEING"):
            for q in (0.16, 0.50, 0.84):
                row[f"{col}_q{int(q * 100)}"] = float(g[col].quantile(q))
        rows.append(row)
    return pd.DataFrame(rows).set_index("SPLIT")
