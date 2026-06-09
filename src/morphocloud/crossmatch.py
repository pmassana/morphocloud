"""Sky cross-matching between DELVE-MC objects and truth catalogs."""

from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord


def sky_match(ra1, dec1, ra2, dec2, radius_arcsec: float):
    """Nearest-neighbor match of catalog 1 against catalog 2.

    Returns (idx1, idx2, sep_arcsec) for pairs closer than radius_arcsec;
    idx1/idx2 index into the input arrays.
    """
    c1 = SkyCoord(np.asarray(ra1), np.asarray(dec1), unit="deg")
    c2 = SkyCoord(np.asarray(ra2), np.asarray(dec2), unit="deg")
    idx, sep, _ = c1.match_to_catalog_sky(c2)
    keep = sep.arcsec <= radius_arcsec
    return np.flatnonzero(keep), idx[keep], sep.arcsec[keep]
