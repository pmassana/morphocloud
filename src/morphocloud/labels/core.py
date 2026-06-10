"""Shared machinery for fetching and caching truth-label tiles."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import DATA_DIR
from ..tap import query
from . import tiles


@dataclass(frozen=True)
class LabelSource:
    """A remote truth catalog fetched per HEALPix tile and cached as parquet."""

    name: str
    table: str
    columns: tuple[str, ...]
    ra_col: str = "ra"
    dec_col: str = "dec"
    #: optional server-side row filter; keeps tile fetches to label candidates
    #: instead of full catalogs (an order of magnitude in transfer time)
    where: str | None = None
    #: named TAP endpoint in tap.SERVICES
    service: str = "datalab"
    #: server-side hard cap on returned rows (MAST: 100k; maxrec above it is
    #: silently ignored and the overflow flag is unreliable). When set, boxes
    #: are COUNT-checked, recursively split in dec to stay under the cap, and
    #: every piece is verified row-complete.
    max_rows: int | None = None

    def cache_path(self, pix: int):
        return DATA_DIR / "labels" / self.name / f"{tiles.tile_id(pix)}.parquet"

    def _box_cond(self, ra1: float, ra2: float, dec1: float, dec2: float,
                  dec_hi_inclusive: bool = True) -> str:
        hi = "<=" if dec_hi_inclusive else "<"
        cond = (f"{self.dec_col} >= {dec1:.6f} "
                f"AND {self.dec_col} {hi} {dec2:.6f}")
        if (ra1, ra2) != (0.0, 360.0):
            if ra1 <= ra2:
                ra_part = f"{self.ra_col} BETWEEN {ra1:.6f} AND {ra2:.6f}"
            else:
                ra_part = (f"({self.ra_col} >= {ra1:.6f} "
                           f"OR {self.ra_col} <= {ra2:.6f})")
            cond = f"{ra_part} AND {cond}"
        if self.where:
            cond = f"{cond} AND ({self.where})"
        return cond

    def tile_condition(self, pix: int) -> str:
        return self._box_cond(*tiles.tile_box(pix))

    def tile_adql(self, pix: int) -> str:
        return (f"SELECT {', '.join(self.columns)} FROM {self.table} "
                f"WHERE {self.tile_condition(pix)}")

    def _fetch_box(self, ra1: float, ra2: float, dec1: float, dec2: float,
                   dec_hi_inclusive: bool = True) -> list[pd.DataFrame]:
        cond = self._box_cond(ra1, ra2, dec1, dec2, dec_hi_inclusive)
        n = None
        if self.max_rows is not None:
            n = query(f"SELECT COUNT(*) AS n FROM {self.table} WHERE {cond}",
                      service=self.service)["n"].iloc[0]
            if n > self.max_rows:
                if dec2 - dec1 < 1e-4:
                    raise RuntimeError(
                        f"{self.name}: {n} rows in an unsplittable dec strip")
                mid = 0.5 * (dec1 + dec2)
                return (self._fetch_box(ra1, ra2, dec1, mid, False)
                        + self._fetch_box(ra1, ra2, mid, dec2,
                                          dec_hi_inclusive))
        adql = f"SELECT {', '.join(self.columns)} FROM {self.table} WHERE {cond}"
        df = query(adql, sync=False, service=self.service)
        if n is not None and len(df) != n:
            raise RuntimeError(
                f"{self.name}: fetched {len(df)} of {n} rows for {cond}")
        return [df]

    def fetch_tile(self, pix: int, overwrite: bool = False) -> pd.DataFrame:
        """Fetch one tile (box query, then cut to the pixel), with caching."""
        path = self.cache_path(pix)
        if path.exists() and not overwrite:
            return pd.read_parquet(path)
        df = pd.concat(self._fetch_box(*tiles.tile_box(pix)),
                       ignore_index=True)
        df = df[tiles.pixel_of(df[self.ra_col], df[self.dec_col]) == pix]
        df = df.reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.rename(path)
        return df

    def load(self, pixels) -> pd.DataFrame:
        """Concatenated labels for a set of tiles (fetching any not cached)."""
        parts = [self.fetch_tile(pix) for pix in sorted(set(pixels))]
        return pd.concat(parts, ignore_index=True)
