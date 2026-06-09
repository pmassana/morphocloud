"""Shared machinery for fetching and caching truth-label tiles."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import DATA_DIR
from ..tap import box_condition, query
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

    def cache_path(self, pix: int):
        return DATA_DIR / "labels" / self.name / f"{tiles.tile_id(pix)}.parquet"

    def tile_adql(self, pix: int) -> str:
        ra1, ra2, dec1, dec2 = tiles.tile_box(pix)
        if (ra1, ra2) == (0.0, 360.0):
            cond = f"{self.dec_col} BETWEEN {dec1:.6f} AND {dec2:.6f}"
        else:
            cond = box_condition(ra1, ra2, dec1, dec2, self.ra_col, self.dec_col)
        if self.where:
            cond = f"{cond} AND ({self.where})"
        return f"SELECT {', '.join(self.columns)} FROM {self.table} WHERE {cond}"

    def fetch_tile(self, pix: int, overwrite: bool = False) -> pd.DataFrame:
        """Fetch one tile (box query, then cut to the pixel), with caching."""
        path = self.cache_path(pix)
        if path.exists() and not overwrite:
            return pd.read_parquet(path)
        df = query(self.tile_adql(pix), sync=False)
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
