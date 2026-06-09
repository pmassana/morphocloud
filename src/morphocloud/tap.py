"""TAP access to external truth catalogs via the NOIRLab Astro Data Lab.

Gaia DR3, LS DR10 and DELVE DR3 are all mirrored at Data Lab, so one anonymous
TAP service covers every truth source. Results are small (label columns only)
and cached as parquet under data/labels/ by the labels modules.
"""

from __future__ import annotations

import pandas as pd
import pyvo

TAP_URL = "https://datalab.noirlab.edu/tap"

_service: pyvo.dal.TAPService | None = None


def service() -> pyvo.dal.TAPService:
    global _service
    if _service is None:
        _service = pyvo.dal.TAPService(TAP_URL)
    return _service


def query(adql: str, sync: bool = True, maxrec: int = 5_000_000) -> pd.DataFrame:
    """Run an ADQL query; use sync=False for region fetches that may be large."""
    svc = service()
    run = svc.run_sync if sync else svc.run_async
    return run(adql, maxrec=maxrec).to_table().to_pandas()


def circle_condition(ra: float, dec: float, radius_deg: float,
                     ra_col: str = "ra", dec_col: str = "dec") -> str:
    """Bounding-box condition covering a cone (RA-wrap and pole safe).

    Data Lab's TAP front end neither translates ADQL POINT/CIRCLE geometry nor
    parses bare q3c predicates, but plain range conditions are indexed and
    fast. The box over-covers the cone; callers post-filter exactly (the tile
    cache cuts to HEALPix pixel membership anyway).
    """
    import numpy as np

    dec1 = max(dec - radius_deg, -90.0)
    dec2 = min(dec + radius_deg, 90.0)
    max_abs_dec = max(abs(dec1), abs(dec2))
    if max_abs_dec >= 89.9:
        return f"{dec_col} BETWEEN {dec1:.6f} AND {dec2:.6f}"
    dra = radius_deg / np.cos(np.radians(max_abs_dec))
    ra1, ra2 = (ra - dra) % 360.0, (ra + dra) % 360.0
    if dra >= 180.0:
        return f"{dec_col} BETWEEN {dec1:.6f} AND {dec2:.6f}"
    return box_condition(ra1, ra2, dec1, dec2, ra_col, dec_col)


def box_condition(ra1: float, ra2: float, dec1: float, dec2: float,
                  ra_col: str = "ra", dec_col: str = "dec") -> str:
    """Plain ra/dec range condition; handles boxes crossing RA=0."""
    dec_part = f"{dec_col} BETWEEN {dec1:.6f} AND {dec2:.6f}"
    if ra1 <= ra2:
        ra_part = f"{ra_col} BETWEEN {ra1:.6f} AND {ra2:.6f}"
    else:
        ra_part = f"({ra_col} >= {ra1:.6f} OR {ra_col} <= {ra2:.6f})"
    return f"({ra_part} AND {dec_part})"
