"""Sky cross-matching between DELVE-MC objects and truth catalogs."""

from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord


def sky_match(ra1, dec1, ra2, dec2, radius_arcsec: float):
    """Nearest-neighbor match of catalog 1 against catalog 2.

    Returns (idx1, idx2, sep_arcsec) for pairs closer than radius_arcsec;
    idx1/idx2 index into the input arrays. Either catalog may be empty
    (e.g. a brick with no unique objects), in which case there are no pairs.
    """
    ra1, dec1 = np.asarray(ra1), np.asarray(dec1)
    ra2, dec2 = np.asarray(ra2), np.asarray(dec2)
    if ra1.size == 0 or ra2.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy(), np.empty(0)
    c1 = SkyCoord(ra1, dec1, unit="deg")
    c2 = SkyCoord(ra2, dec2, unit="deg")
    idx, sep, _ = c1.match_to_catalog_sky(c2)
    keep = sep.arcsec <= radius_arcsec
    return np.flatnonzero(keep), idx[keep], sep.arcsec[keep]
