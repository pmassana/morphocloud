"""Per-brick truth-label assembly.

Cross-matches a brick's objects against all label sources and combines the
votes: STAR if any star source claims it, GALAXY if any galaxy source claims
it, conflicting claims dropped (kept with label=CONFLICT for accounting).
Provenance flags record which surveys contributed each label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import bricks
from ..crossmatch import sky_match
from . import delvedr3, gaia, lsdr10, tiles

# verified against Gaia DR3: median offset ~0.01", 90th pct sep 0.15"
MATCH_RADIUS_ARCSEC = 0.5

STAR, GALAXY, CONFLICT = 1, 0, -1


def _match_flag(objects: pd.DataFrame, truth: pd.DataFrame,
                ra_col: str = "ra", dec_col: str = "dec") -> np.ndarray:
    """True for objects matched to a truth row within MATCH_RADIUS_ARCSEC.

    Each truth row may label only one object: in crowded fields several
    objects can claim the same truth source, and only the closest keeps it.
    """
    flag = np.zeros(len(objects), dtype=bool)
    if len(truth) == 0:
        return flag
    i_obj, i_truth, sep = sky_match(
        objects["RA"], objects["DEC"], truth[ra_col], truth[dec_col],
        MATCH_RADIUS_ARCSEC,
    )
    order = np.argsort(sep)
    seen: set[int] = set()
    for io, it in zip(i_obj[order], i_truth[order]):
        if it not in seen:
            seen.add(it)
            flag[io] = True
    return flag


def brick_labels(brickname: str, objects: pd.DataFrame | None = None) -> pd.DataFrame:
    """Labels and provenance for one brick's (BRICKUNIQ) objects.

    Returns a frame aligned row-by-row with `objects`, with columns LABEL
    (STAR/GALAXY/CONFLICT, NaN if unlabelled) and per-source provenance flags.
    """
    if objects is None:
        objects = bricks.read_objects(brickname)
    row = bricks.load_brick_list().set_index("BRICKNAME").loc[brickname]
    pixels = {int(p) for p in tiles.pixel_of(
        [row["RA"], row["RA1"], row["RA1"], row["RA2"], row["RA2"]],
        [row["DEC"], row["DEC1"], row["DEC2"], row["DEC1"], row["DEC2"]],
    )}

    g = gaia.GAIA.load(pixels)
    ls = lsdr10.LSDR10.load(pixels)
    d3 = delvedr3.DELVEDR3.load(pixels)

    prov = pd.DataFrame({
        "GAIA_STAR": _match_flag(objects, g[gaia.point_source_mask(g)]),
        "LS_GALAXY": _match_flag(objects, ls[lsdr10.galaxy_mask(ls)]),
        "DR3_STAR": _match_flag(objects, d3[delvedr3.star_mask(d3)]),
        "DR3_GALAXY": _match_flag(objects, d3[delvedr3.galaxy_mask(d3)]),
    })

    star_vote = prov["GAIA_STAR"] | prov["DR3_STAR"]
    galaxy_vote = prov["LS_GALAXY"] | prov["DR3_GALAXY"]
    label = pd.Series(np.nan, index=objects.index, name="LABEL")
    label[star_vote & ~galaxy_vote] = STAR
    label[galaxy_vote & ~star_vote] = GALAXY
    label[star_vote & galaxy_vote] = CONFLICT

    out = pd.concat([objects[["BRICKNAME", "OBJID"]], label, prov], axis=1)
    return out
