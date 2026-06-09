"""Map where each truth catalog actually has sources over the footprint.

One ring256 (nside=256 RING, ~0.23 deg pixels) GROUP BY per catalog, then
per-brick counts. LS DR10 counts only real extended detections (DEV/EXP/SER):
near the MCs its tractor rows are pure Gaia-forced PSF entries, so total row
counts would fake coverage where no detection ever ran.

Outputs under data/labels/coverage/:
  ls_dr10_galaxies_ring256.parquet, delve_dr3_ring256.parquet,
  brick_coverage.parquet (BRICKNAME, n_lsgal, n_dr3)
"""

import healpy as hp
import numpy as np
import pandas as pd

from morphocloud import bricks
from morphocloud.config import DATA_DIR
from morphocloud.tap import query

OUT = DATA_DIR / "labels" / "coverage"
DEC_CUT = -44.8  # footprint reaches dec -44.875

QUERIES = {
    "ls_dr10_galaxies_ring256": (
        "SELECT ring256, COUNT(*) AS n FROM ls_dr10.tractor "
        f"WHERE dec < {DEC_CUT} AND type IN ('DEV','EXP','SER') GROUP BY ring256"
    ),
    "delve_dr3_ring256": (
        "SELECT ring256, COUNT(*) AS n FROM delve_dr3.coadd_objects "
        f"WHERE dec < {DEC_CUT} GROUP BY ring256"
    ),
}


def fetch(name: str, adql: str) -> pd.DataFrame:
    path = OUT / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = query(adql, sync=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def brick_pixel_matrix(b: pd.DataFrame) -> np.ndarray:
    """(nbricks, 5) ring256 pixels of each brick's center and corners."""
    ras = np.column_stack([b["RA"], b["RA1"], b["RA1"], b["RA2"], b["RA2"]])
    decs = np.column_stack([b["DEC"], b["DEC1"], b["DEC2"], b["DEC1"], b["DEC2"]])
    return hp.ang2pix(256, np.mod(ras, 360.0), decs, lonlat=True)


def per_brick_counts(pixmat: np.ndarray, counts: pd.DataFrame) -> np.ndarray:
    lookup = np.zeros(hp.nside2npix(256), dtype=np.int64)
    lookup[counts["ring256"].to_numpy()] = counts["n"].to_numpy()
    # sum over the brick's unique pixels (center may equal a corner pixel)
    total = np.zeros(len(pixmat), dtype=np.int64)
    for i, pix in enumerate(pixmat):
        total[i] = lookup[np.unique(pix)].sum()
    return total


def main():
    maps = {name: fetch(name, adql) for name, adql in QUERIES.items()}
    for name, df in maps.items():
        print(f"{name}: {len(df)} pixels, {df['n'].sum():,} sources")

    b = bricks.load_brick_list()
    pixmat = brick_pixel_matrix(b)
    cov = pd.DataFrame({
        "BRICKNAME": b["BRICKNAME"],
        "n_lsgal": per_brick_counts(pixmat, maps["ls_dr10_galaxies_ring256"]),
        "n_dr3": per_brick_counts(pixmat, maps["delve_dr3_ring256"]),
    })
    cov.to_parquet(OUT / "brick_coverage.parquet")

    nb = len(cov)
    print(f"\nbricks total: {nb}")
    print(f"DELVE DR3 coverage  (n_dr3 > 0):    {(cov['n_dr3'] > 0).mean():6.1%}")
    print(f"LS DR10 galaxy labels (n >= 10):    {(cov['n_lsgal'] >= 10).mean():6.1%}")
    print(f"LS DR10 galaxy labels (n >= 100):   {(cov['n_lsgal'] >= 100).mean():6.1%}")
    print(f"neither truth source:               "
          f"{((cov['n_dr3'] == 0) & (cov['n_lsgal'] < 10)).mean():6.1%}")


if __name__ == "__main__":
    main()
