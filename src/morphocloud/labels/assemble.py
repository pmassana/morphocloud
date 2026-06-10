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
from . import delvedr3, gaia, hsc, lsdr10, tiles

# verified against Gaia DR3: median offset ~0.01", 90th pct sep 0.15"
MATCH_RADIUS_ARCSEC = 0.5

STAR, GALAXY, CONFLICT = 1, 0, -1

PROVENANCE_COLUMNS = (
    "GAIA_STAR", "LS_GALAXY", "DR3_STAR", "DR3_GALAXY",
    "HST_GALAXY", "HST_BLEND",
)


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


def _component_flags(objects: pd.DataFrame, truth: pd.DataFrame,
                     point: np.ndarray, extended: np.ndarray,
                     ra_col: str, dec_col: str):
    """Blend-aware many-to-one match for a high-resolution truth catalog.

    Each truth component is assigned to its nearest object within
    MATCH_RADIUS_ARCSEC. An object is a star only if every component in it is
    point-like, a galaxy only if every component is extended; mixed or
    ambiguous components leave it unlabelled, flagged as a blend instead.
    `point`/`extended` are boolean masks positionally aligned with `truth`.

    Returns (star, blend, galaxy) boolean arrays aligned with `objects`.
    """
    star = np.zeros(len(objects), dtype=bool)
    galaxy = np.zeros(len(objects), dtype=bool)
    blend = np.zeros(len(objects), dtype=bool)
    if len(truth) == 0:
        return star, galaxy, blend
    i_truth, i_obj, _ = sky_match(
        truth[ra_col], truth[dec_col], objects["RA"], objects["DEC"],
        MATCH_RADIUS_ARCSEC,
    )
    n_all = np.bincount(i_obj, minlength=len(objects))
    n_point = np.bincount(
        i_obj, weights=np.asarray(point)[i_truth], minlength=len(objects))
    n_ext = np.bincount(
        i_obj, weights=np.asarray(extended)[i_truth], minlength=len(objects))
    matched = n_all > 0
    star = matched & (n_point == n_all)
    galaxy = matched & (n_ext == n_all)
    blend = matched & ~star & ~galaxy
    return star, galaxy, blend


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
    hs = hsc.HSC.load(pixels)

    # the all-point-like flag is deliberately unused: point-like CI at faint
    # magnitudes includes shredded-galaxy knots, so HSC gives no star labels
    _, hst_galaxy, hst_blend = _component_flags(
        objects, hs, hsc.point_source_mask(hs), hsc.galaxy_mask(hs),
        hsc.HSC.ra_col, hsc.HSC.dec_col,
    )
    prov = pd.DataFrame({
        "GAIA_STAR": _match_flag(objects, g[gaia.point_source_mask(g)]),
        "LS_GALAXY": _match_flag(objects, ls[lsdr10.galaxy_mask(ls)]),
        "DR3_STAR": _match_flag(objects, d3[delvedr3.star_mask(d3)]),
        "DR3_GALAXY": _match_flag(objects, d3[delvedr3.galaxy_mask(d3)]),
        "HST_GALAXY": hst_galaxy,
        # matched to HST but mixed/ambiguous components: never labelled,
        # kept for accounting
        "HST_BLEND": hst_blend,
    })

    star_vote = prov["GAIA_STAR"] | prov["DR3_STAR"]
    galaxy_vote = prov["LS_GALAXY"] | prov["DR3_GALAXY"] | prov["HST_GALAXY"]
    label = pd.Series(np.nan, index=objects.index, name="LABEL")
    label[star_vote & ~galaxy_vote] = STAR
    label[galaxy_vote & ~star_vote] = GALAXY
    label[star_vote & galaxy_vote] = CONFLICT

    out = pd.concat([objects[["BRICKNAME", "OBJID"]], label, prov], axis=1)
    return out
