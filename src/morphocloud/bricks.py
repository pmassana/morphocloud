"""Readers for the DELVE-MC y4t2 brick catalogs.

Object files are DAOPHOT/ALLFRAME products whose band columns vary per brick
(g/r/i/z mostly present, u/Y patchy). `read_objects` normalizes every brick to
one schema: requested band columns always present, sentinels converted to NaN.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from astropy.table import Table

from .config import (
    BRICK_LIST,
    CORE_BANDS,
    DELVEMC_DATA,
    FWHM_SENTINEL,
    MAG_SENTINEL,
)

# ERR / SCATTER / MAGERR_AUTO columns use 9.99 as their missing sentinel
ERR_SENTINEL = 9.9

# band-independent float columns and their sentinels
MORPH_SENTINELS = {
    "CHI": MAG_SENTINEL,
    "SHARP": MAG_SENTINEL,
    "PROB": MAG_SENTINEL,
    "MAG_AUTO": MAG_SENTINEL,
    "MAGERR_AUTO": ERR_SENTINEL,
    "ASEMI": MAG_SENTINEL,
    "BSEMI": MAG_SENTINEL,
    "THETA": MAG_SENTINEL,
    "ELLIPTICITY": MAG_SENTINEL,
    "FWHM": FWHM_SENTINEL,
    "EBV": MAG_SENTINEL,
}


def object_path(brickname: str):
    return DELVEMC_DATA / f"{brickname}_object.fits.gz"


def meta_path(brickname: str):
    return DELVEMC_DATA / f"{brickname}_meta.fits"


@functools.lru_cache(maxsize=1)
def load_brick_list() -> pd.DataFrame:
    """Brick definitions for the footprint (one row per 0.25 deg brick)."""
    df = Table.read(BRICK_LIST).to_pandas()
    if df["BRICKNAME"].dtype == object:
        df["BRICKNAME"] = df["BRICKNAME"].str.decode("utf-8")
    df["BRICKNAME"] = df["BRICKNAME"].str.strip()
    return df


def available_bricks() -> list[str]:
    """Bricknames with an object catalog on disk."""
    return sorted(
        p.name.removesuffix("_object.fits.gz")
        for p in DELVEMC_DATA.glob("*_object.fits.gz")
        if not p.name.endswith("_joint_object.fits.gz")
    )


def _clean(values, sentinel: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).copy()
    v[v >= sentinel] = np.nan
    return v


def read_objects(
    brickname: str,
    bands: tuple[str, ...] = CORE_BANDS,
    unique_only: bool = True,
) -> pd.DataFrame:
    """Read one brick's object catalog with a normalized schema.

    Bands absent from the file come back as NaN columns with NDET=0, so every
    brick yields identical columns. `unique_only` keeps BRICKUNIQ==1 sources
    (deduplicated across brick overlaps); pass False for inference on full
    bricks.
    """
    t = Table.read(object_path(brickname))
    n = len(t)
    out: dict[str, np.ndarray] = {
        "OBJID": np.asarray(t["OBJID"]).astype(str),
        "RA": np.asarray(t["RA"], dtype=np.float64),
        "DEC": np.asarray(t["DEC"], dtype=np.float64),
    }
    for band in bands:
        if f"{band}MAG" in t.colnames:
            out[f"{band}MAG"] = _clean(t[f"{band}MAG"], MAG_SENTINEL)
            out[f"{band}ERR"] = _clean(t[f"{band}ERR"], ERR_SENTINEL)
            out[f"{band}SCATTER"] = _clean(t[f"{band}SCATTER"], MAG_SENTINEL)
            out[f"NDET{band}"] = np.asarray(t[f"NDET{band}"], dtype=np.int32)
        else:
            nan = np.full(n, np.nan, dtype=np.float32)
            out[f"{band}MAG"] = nan
            out[f"{band}ERR"] = nan.copy()
            out[f"{band}SCATTER"] = nan.copy()
            out[f"NDET{band}"] = np.zeros(n, dtype=np.int32)
    for col, sentinel in MORPH_SENTINELS.items():
        out[col] = _clean(t[col], sentinel)
    out["BRICKUNIQ"] = np.asarray(t["BRICKUNIQ"]).astype(bool)

    df = pd.DataFrame(out)
    df.insert(0, "BRICKNAME", brickname)
    if unique_only:
        df = df[df["BRICKUNIQ"]].reset_index(drop=True)
    return df


def read_meta(brickname: str) -> pd.DataFrame:
    """Per-exposure metadata for a brick (scalar columns only)."""
    t = Table.read(meta_path(brickname))
    scalar_cols = [c for c in t.colnames if len(t[c].shape) == 1]
    df = t[scalar_cols].to_pandas()
    for col in ("FILTER", "FIELD", "BASE", "FILE"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8").str.strip()
    return df


def brick_seeing(brickname: str) -> dict[str, float]:
    """Median per-band seeing FWHM in arcsec from the exposures of a brick."""
    meta = read_meta(brickname)
    band = meta["FILTER"].str[0].str.upper()
    fwhm_arcsec = meta["FWHM"] * meta["PIXSCALE"]
    ok = (meta["FWHM"] > 0) & (meta["FWHM"] < FWHM_SENTINEL)
    return fwhm_arcsec[ok].groupby(band[ok]).median().to_dict()
