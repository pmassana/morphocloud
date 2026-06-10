"""TAP access to the external truth catalogs.

Gaia DR3, LS DR10 and DELVE DR3 are mirrored at the NOIRLab Astro Data Lab;
HSC v3 (HST) is served by the MAST VO TAP. One anonymous TAP service per
archive, selected by name. Results are small (label columns only) and cached
as parquet under data/labels/ by the labels modules.

MAST quirks (verified 2026-06-09; Data Lab quirks are in docs/plan.md):
column names are lowercase, SUM(CASE ...) does not parse, results are hard-
capped at 100,000 rows (sync and async; maxrec above it is silently ignored)
and the overflow flag is unreliable (also set on complete results) — verify
completeness against COUNT(*) and split big queries (LabelSource.max_rows).
"""

from __future__ import annotations

import pandas as pd
import pyvo

SERVICES = {
    "datalab": "https://datalab.noirlab.edu/tap",
    "mast_hsc": "https://mast.stsci.edu/vo-tap/api/v0.1/hsc",
}

_services: dict[str, pyvo.dal.TAPService] = {}


def _service(name: str) -> pyvo.dal.TAPService:
    if name not in _services:
        _services[name] = pyvo.dal.TAPService(SERVICES[name])
    return _services[name]


def query(adql: str, sync: bool = True, maxrec: int = 5_000_000,
          service: str = "datalab") -> pd.DataFrame:
    """Run an ADQL query; use sync=False for region fetches that may be large."""
    svc = _service(service)
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
