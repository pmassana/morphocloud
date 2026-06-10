"""Map where the Hubble Source Catalog v3 has sources over the footprint.

HSC v3 is served by the MAST VO TAP (anonymous), not Astro Data Lab. Its
summary table has no healpix column, but GROUP BY on FLOOR expressions runs
server-side, so the whole census is three cheap 0.25-deg-grid queries.

MAST TAP quirks (verified 2026-06-09): column names are lowercase (matchra,
matchdec, ci, numimages); SUM(CASE ...) does not parse (strict ADQL 2.1
reserved words) but COUNT(*) + GROUP BY FLOOR(expr) works; counts over the
full southern cap return in seconds.

Per-cell counts:
  n_all  -- all matched sources
  n_img2 -- numimages >= 2 (the artifact cut from the plan)
  n_ext  -- numimages >= 2 AND ci > 1.3: PROVISIONAL extended cut for sizing
            the galaxy-label pool only; real per-instrument CI thresholds are
            a later, validated step.

Cells are assigned to bricks by cell center (cells straddle brick edges, so
brick-scale counts are approximate -- fine for a census).

Outputs under data/labels/coverage/:
  hsc_cells_0.25deg.parquet   cell-grid counts with tile/brick assignment
  hsc_tile_coverage.parquet   per nside32 tile (fetch planning)
  hsc_brick_coverage.parquet  per brick (BRICKNAME, n_hsc_img2, n_hsc_ext)
Prints summary stats, including HST coverage of the bricks blind to both
LS DR10 and DELVE DR3 (needs brick_coverage.parquet from build_coverage_maps).
"""

import numpy as np
import pandas as pd

from morphocloud import bricks
from morphocloud.config import DATA_DIR
from morphocloud.labels import tiles
from morphocloud.tap import query

OUT = DATA_DIR / "labels" / "coverage"
DEC_CUT = -44.8  # footprint reaches dec -44.875
HSC_TABLE = "dbo.summagaper2catview"
CELLS_PER_DEG = 4

GRIDS = {
    "n_all": f"matchdec < {DEC_CUT}",
    "n_img2": f"matchdec < {DEC_CUT} AND numimages >= 2",
    "n_ext": f"matchdec < {DEC_CUT} AND numimages >= 2 AND ci > 1.3",
}


def fetch_grid(where: str) -> pd.DataFrame:
    k = CELLS_PER_DEG
    adql = (
        f"SELECT FLOOR(matchra*{k}) AS cra, FLOOR(matchdec*{k}) AS cdec, "
        f"COUNT(*) AS n FROM {HSC_TABLE} WHERE {where} "
        f"GROUP BY FLOOR(matchra*{k}), FLOOR(matchdec*{k})"
    )
    df = query(adql, service="mast_hsc")
    df["cra"] = df["cra"].astype(np.int64)
    df["cdec"] = df["cdec"].astype(np.int64)
    # MAST flags these results as overflowed even when complete; verify by
    # cross-checking the grid total against a server-side count
    total = query(f"SELECT COUNT(*) AS n FROM {HSC_TABLE} WHERE {where}",
                  service="mast_hsc")["n"].iloc[0]
    assert df["n"].sum() == total, f"grid truncated: {df['n'].sum()} != {total}"
    return df.set_index(["cra", "cdec"])["n"]


def fetch_cells() -> pd.DataFrame:
    path = OUT / "hsc_cells_0.25deg.parquet"
    if path.exists():
        return pd.read_parquet(path)
    cells = pd.DataFrame({name: fetch_grid(where)
                          for name, where in GRIDS.items()})
    cells = cells.fillna(0).astype(np.int64).reset_index()
    cells["ra"] = (cells["cra"] + 0.5) / CELLS_PER_DEG
    cells["dec"] = (cells["cdec"] + 0.5) / CELLS_PER_DEG
    cells["pix"] = tiles.pixel_of(cells["ra"], cells["dec"])

    b = bricks.load_brick_list()
    cells["ibrick"] = containing_brick(b, cells["ra"].to_numpy(),
                                       cells["dec"].to_numpy())
    name = b["BRICKNAME"].to_numpy()
    cells["BRICKNAME"] = np.where(cells["ibrick"] >= 0,
                                  name[cells["ibrick"]], "")
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(path)
    return cells


def containing_brick(b: pd.DataFrame, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Index into b of the brick containing each point, -1 if outside."""
    ra1, ra2 = b["RA1"].to_numpy(), b["RA2"].to_numpy()
    dec1, dec2 = b["DEC1"].to_numpy(), b["DEC2"].to_numpy()
    wrap = ra1 > ra2
    idx = np.full(len(ra), -1, dtype=np.int64)
    for i, (r, d) in enumerate(zip(ra, dec)):
        in_ra = np.where(wrap, (r >= ra1) | (r < ra2), (ra1 <= r) & (r < ra2))
        hits = np.flatnonzero(in_ra & (dec1 <= d) & (d < dec2))
        if len(hits):
            idx[i] = hits[0]
    return idx


def main():
    cells = fetch_cells()
    south = cells[["n_all", "n_img2", "n_ext"]].sum()
    print(f"HSC v3 south of dec {DEC_CUT}: {south['n_all']:,} sources "
          f"({south['n_img2']:,} with numimages>=2, {south['n_ext']:,} ext)")

    fp = cells[cells["ibrick"] >= 0]
    infp = fp[["n_all", "n_img2", "n_ext"]].sum()
    print(f"in footprint bricks:          {infp['n_all']:,} sources "
          f"({infp['n_img2']:,} with numimages>=2, {infp['n_ext']:,} ext)")

    tile = fp.groupby("pix")[["n_all", "n_img2", "n_ext"]].sum().reset_index()
    tile["tile"] = [tiles.tile_id(p) for p in tile["pix"]]
    tile.to_parquet(OUT / "hsc_tile_coverage.parquet")
    npix = len(tiles.footprint_pixels(bricks.load_brick_list()))
    print(f"footprint tiles with HSC img2 sources: "
          f"{(tile['n_img2'] > 0).sum()} of {npix}")

    per_brick = fp.groupby("BRICKNAME")[["n_img2", "n_ext"]].sum()
    per_brick.columns = ["n_hsc_img2", "n_hsc_ext"]
    cov = pd.read_parquet(OUT / "brick_coverage.parquet")
    cov = cov.join(per_brick, on="BRICKNAME").fillna(0)
    cov[["n_hsc_img2", "n_hsc_ext"]] = (
        cov[["n_hsc_img2", "n_hsc_ext"]].astype(np.int64))
    cov.to_parquet(OUT / "hsc_brick_coverage.parquet")

    nb = len(cov)
    has_hst = cov["n_hsc_img2"] > 0
    print(f"\nbricks with HSC img2 sources: {has_hst.sum()} of {nb} "
          f"({has_hst.mean():.1%})")
    blind = (cov["n_dr3"] == 0) & (cov["n_lsgal"] < 10)
    print(f"blind bricks (no DR3, <10 LS galaxies): {blind.sum()}")
    print(f"  with HSC img2 sources:  {(blind & has_hst).sum()} "
          f"({(blind & has_hst).sum() / blind.sum():.1%})")
    print(f"  HSC img2 sources there: {cov.loc[blind, 'n_hsc_img2'].sum():,}")
    print(f"  HSC ext sources there:  {cov.loc[blind, 'n_hsc_ext'].sum():,} "
          f"(provisional ci > 1.3 cut)")


if __name__ == "__main__":
    main()
