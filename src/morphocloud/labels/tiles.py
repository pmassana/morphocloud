"""HEALPix tiling of the footprint for chunked truth-catalog fetching.

Labels are fetched per HEALPix pixel (nside=32 nest, ~3.4 deg^2): one cone
query covering the pixel, then rows filtered to the pixel itself, so each
cached tile holds every label source inside that pixel exactly once.
"""

from __future__ import annotations

import healpy as hp
import numpy as np

NSIDE = 32
NEST = True

# padding around the pixel boundary box; rows are cut to exact pixel
# membership after the fetch, so this only needs to absorb edge effects
EPS_DEG = 0.01


def pixel_of(ra, dec) -> np.ndarray:
    """HEALPix pixel index for positions in degrees."""
    return hp.ang2pix(NSIDE, np.asarray(ra), np.asarray(dec), nest=NEST, lonlat=True)


def tile_id(pix: int) -> str:
    return f"hp{NSIDE}-{pix:05d}"


def tile_box(pix: int) -> tuple[float, float, float, float]:
    """(ra1, ra2, dec1, dec2) bounding the pixel; ra1 > ra2 marks RA wrap.

    ra1 == 0 and ra2 == 360 (full RA range) is returned for polar pixels.
    """
    vecs = hp.boundaries(NSIDE, pix, step=4, nest=NEST)
    ra, dec = hp.vec2ang(vecs.T, lonlat=True)
    dec1 = max(dec.min() - EPS_DEG, -90.0)
    dec2 = min(dec.max() + EPS_DEG, 90.0)
    if dec1 <= -89.5 or dec2 >= 89.5:
        return 0.0, 360.0, dec1, dec2
    # wrap-safe RA range: the box is the complement of the largest gap
    ra_sorted = np.sort(ra)
    gaps = np.diff(ra_sorted)
    wrap_gap = ra_sorted[0] + 360.0 - ra_sorted[-1]
    if wrap_gap >= gaps.max():
        ra1, ra2 = ra_sorted[0], ra_sorted[-1]  # no wrap
    else:
        i = int(np.argmax(gaps))
        ra1, ra2 = ra_sorted[i + 1], ra_sorted[i]  # box crosses RA=0
    return (ra1 - EPS_DEG) % 360.0, (ra2 + EPS_DEG) % 360.0, dec1, dec2


def pixels_for_brick(brick_row) -> set[int]:
    """All tiles a brick may touch (its center and four corners)."""
    ras = [brick_row["RA"], brick_row["RA1"], brick_row["RA1"],
           brick_row["RA2"], brick_row["RA2"]]
    decs = [brick_row["DEC"], brick_row["DEC1"], brick_row["DEC2"],
            brick_row["DEC1"], brick_row["DEC2"]]
    return set(pixel_of(ras, decs).tolist())


def footprint_pixels(brick_list) -> list[int]:
    """Sorted unique tiles touched by any brick in the footprint."""
    ras = np.concatenate([
        np.mod(np.asarray(brick_list[c]), 360.0)
        for c in ("RA", "RA1", "RA1", "RA2", "RA2")
    ])
    decs = np.concatenate([
        np.asarray(brick_list[c])
        for c in ("DEC", "DEC1", "DEC2", "DEC1", "DEC2")
    ])
    return sorted(np.unique(pixel_of(ras, decs)).tolist())
