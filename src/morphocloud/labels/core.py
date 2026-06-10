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
    #: emit the tile box as an indexed ADQL geometry predicate
    #: (CONTAINS/POLYGON) instead of plain ra/dec ranges. Required on
    #: geometry-capable services where a bare-range COUNT is unindexed and
    #: ~25x slower (Vizier); Data Lab/MAST do not parse geometry, so leave off.
    geometry: bool = False
    #: rename fetched columns to canonical names before caching, so tiles from
    #: a renamed mirror (Vizier's Gaia: Source, RA_ICRS, Gmag, RUWE, sepsi…)
    #: are schema-identical on disk to the primary archive's lowercase names.
    rename: dict | None = None
    #: identity column (canonical, i.e. post-rename) used to drop the rare
    #: duplicate that a geometry split can produce on a shared dec edge.
    id_col: str | None = None
    #: optional post-rename dtype coercion, so tiles from a mirror that serves
    #: wider floats (Vizier: float64) match the primary archive's on-disk
    #: schema (ESA serves the photometry/noise columns as float32).
    cast: dict | None = None

    def cache_path(self, pix: int):
        return DATA_DIR / "labels" / self.name / f"{tiles.tile_id(pix)}.parquet"

    def _box_cond(self, ra1: float, ra2: float, dec1: float, dec2: float,
                  dec_hi_inclusive: bool = True) -> str:
        if self.geometry and (ra1, ra2) != (0.0, 360.0) and ra1 <= ra2:
            # Index-accelerated rectangle. CONTAINS is inclusive on both dec
            # edges, so split halves can share a boundary row; id_col dedup in
            # fetch_tile removes it (preferred over losing edge rows). RA-wrap
            # and polar tiles fall through to the range form below — absent
            # from the MC footprint, where only the dense tiles need the index.
            poly = (f"POLYGON('ICRS',"
                    f"{ra1:.6f},{dec1:.6f},{ra2:.6f},{dec1:.6f},"
                    f"{ra2:.6f},{dec2:.6f},{ra1:.6f},{dec2:.6f})")
            cond = f"1=CONTAINS(POINT('ICRS',{self.ra_col},{self.dec_col}),{poly})"
            if self.where:
                cond = f"{cond} AND ({self.where})"
            return cond
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
        if self.rename:
            df = df.rename(columns=self.rename)
        if self.id_col:
            df = df.drop_duplicates(subset=[self.id_col]).reset_index(drop=True)
        if self.cast:
            df = df.astype(self.cast)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.rename(path)
        return df

    def load(self, pixels) -> pd.DataFrame:
        """Concatenated labels for a set of tiles (fetching any not cached)."""
        parts = [self.fetch_tile(pix) for pix in sorted(set(pixels))]
        return pd.concat(parts, ignore_index=True)
