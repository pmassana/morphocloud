"""Per-brick truth-label assembly.

Cross-matches a brick's objects against all label sources and combines the
votes: STAR if any star source claims it, GALAXY if any galaxy source claims
it, conflicting claims dropped (kept with label=CONFLICT for accounting).
Provenance flags record which surveys contributed each label. Two flags are
accounting-only, never votes: HST_BLEND (mixed/ambiguous HST components) and
GAIA_QSO (purer Gaia QSO candidates — point sources, so their STAR labels
from the astrometric cuts are correct for the morphological target; the flag
exists for contamination measurements and downstream masking). IN_MC_CORE
marks objects inside the Gaia extragalactic paper's LMC/SMC exclusion
circles, for split evaluation of the crowded cores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import bricks
from ..crossmatch import sky_match
from . import delvedr3, gaia, gaia_extragal, hsc, lsdr10, tiles

# verified against Gaia DR3: median offset ~0.01", 90th pct sep 0.15"
MATCH_RADIUS_ARCSEC = 0.5

# Artifact-mask match radius: LS masked rows sample (not delineate) the masked
# region, so a DELVE-MC object is flagged if it sits within this radius of any
# masked LS detection. Looser than the label radius to bridge the sampling
# spacing inside a region without bleeding far past its edges.
MASK_MATCH_ARCSEC = 1.0

STAR, GALAXY, CONFLICT = 1, 0, -1

PROVENANCE_COLUMNS = (
    "GAIA_STAR", "GAIA_GALAXY", "GAIA_QSO", "LS_GALAXY", "LS_STAR",
    "DR3_STAR", "DR3_GALAXY", "HST_GALAXY", "HST_STAR", "HST_BLEND",
)


def _match_flag(objects: pd.DataFrame, truth: pd.DataFrame,
                ra_col: str = "ra", dec_col: str = "dec",
                radius: float = MATCH_RADIUS_ARCSEC) -> np.ndarray:
    """True for objects matched to a truth row within `radius` arcsec.

    Each truth row may label only one object: in crowded fields several
    objects can claim the same truth source, and only the closest keeps it.
    """
    flag = np.zeros(len(objects), dtype=bool)
    if len(truth) == 0:
        return flag
    i_obj, i_truth, sep = sky_match(
        objects["RA"], objects["DEC"], truth[ra_col], truth[dec_col],
        radius,
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
    (STAR/GALAXY/CONFLICT, NaN if unlabelled), per-source provenance flags,
    and the IN_MC_CORE region flag.
    """
    if objects is None:
        objects = bricks.read_objects(brickname)
    row = bricks.load_brick_list().set_index("BRICKNAME").loc[brickname]
    pixels = {int(p) for p in tiles.pixel_of(
        [row["RA"], row["RA1"], row["RA1"], row["RA2"], row["RA2"]],
        [row["DEC"], row["DEC1"], row["DEC2"], row["DEC1"], row["DEC2"]],
    )}

    g = gaia.GAIA.load(pixels)
    gx = gaia_extragal.GALCAND.load(pixels)
    qs = gaia_extragal.QSOCAND.load(pixels)
    ls = lsdr10.LSDR10.load(pixels)
    lst = lsdr10.LS_STAR.load(pixels)
    lm = lsdr10.LS_MASK.load(pixels)
    d3 = delvedr3.DELVEDR3.load(pixels)
    hs = hsc.HSC.load(pixels)

    # HSC stars: isolated, point-like, optically-detected components (blend-aware,
    # like the galaxy side). Carries ~27% compact-galaxy contamination that no CI
    # or colour cut removes, so it is a provenance-flagged, down-weightable vote.
    hst_star, hst_galaxy, hst_blend = _component_flags(
        objects, hs, hsc.star_candidate_mask(hs), hsc.galaxy_mask(hs),
        hsc.HSC.ra_col, hsc.HSC.dec_col,
    )
    prov = pd.DataFrame({
        "GAIA_STAR": _match_flag(objects, g[gaia.point_source_mask(g)]),
        "GAIA_GALAXY": _match_flag(objects, gx[gaia_extragal.galaxy_mask(gx)]),
        # purer Gaia QSOs: accounting/evaluation only, never a vote
        "GAIA_QSO": _match_flag(objects, qs[gaia_extragal.qso_mask(qs)]),
        "LS_GALAXY": _match_flag(objects, ls[lsdr10.galaxy_mask(ls)]),
        # LS PSF stars (full depth, Gaia-forced excluded): deepest star source
        "LS_STAR": _match_flag(objects, lst[lsdr10.star_mask(lst)]),
        "DR3_STAR": _match_flag(objects, d3[delvedr3.star_mask(d3)]),
        "DR3_GALAXY": _match_flag(objects, d3[delvedr3.galaxy_mask(d3)]),
        "HST_GALAXY": hst_galaxy,
        "HST_STAR": hst_star,
        # matched to HST but mixed/ambiguous components: never labelled,
        # kept for accounting
        "HST_BLEND": hst_blend,
    })

    star_vote = (prov["GAIA_STAR"] | prov["DR3_STAR"]
                 | prov["LS_STAR"] | prov["HST_STAR"])
    galaxy_vote = (prov["GAIA_GALAXY"] | prov["LS_GALAXY"]
                   | prov["DR3_GALAXY"] | prov["HST_GALAXY"])
    label = pd.Series(np.nan, index=objects.index, name="LABEL")
    label[star_vote & ~galaxy_vote] = STAR
    label[galaxy_vote & ~star_vote] = GALAXY
    label[star_vote & galaxy_vote] = CONFLICT

    out = pd.concat([objects[["BRICKNAME", "OBJID"]], label, prov], axis=1)
    out["IN_MC_CORE"] = gaia_extragal.in_mc_core(objects["RA"], objects["DEC"])
    out["IN_ARTIFACT_MASK"] = _in_artifact_mask(objects, lm)
    return out


def _in_artifact_mask(objects: pd.DataFrame, ls_mask: pd.DataFrame) -> np.ndarray:
    """True for objects within MASK_MATCH_ARCSEC of any masked LS row.

    A region flag (many-to-one): unlike the label matchers, every object near
    a masked detection is flagged, not just the nearest one.
    """
    in_mask = np.zeros(len(objects), dtype=bool)
    masked = ls_mask[lsdr10.masked_mask(ls_mask)] if len(ls_mask) else ls_mask
    if len(masked) == 0 or len(objects) == 0:
        return in_mask
    i_obj, _, _ = sky_match(
        objects["RA"], objects["DEC"],
        masked[lsdr10.LS_MASK.ra_col], masked[lsdr10.LS_MASK.dec_col],
        MASK_MATCH_ARCSEC,
    )
    in_mask[i_obj] = True
    return in_mask
