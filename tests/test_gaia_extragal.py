"""Unit tests for the Gaia DR3 extragalactic masks and MC-core geometry."""

import numpy as np
import pandas as pd

from morphocloud.labels import gaia_extragal, tiles
from morphocloud.labels.gaia_extragal import GALCAND, QSOCAND, in_mc_core

# a periphery position outside both MC cores
RA_OUT, DEC_OUT = 30.0, -55.0


def _galcand(n, **overrides):
    """Purer-by-default galcand-like frame at a periphery position."""
    df = pd.DataFrame({
        "ra": np.full(n, RA_OUT),
        "dec": np.full(n, DEC_OUT),
        "radius_sersic": np.full(n, np.nan),
        "classlabel_dsc_joint": ["galaxy"] * n,
        "vari_best_class_name": [""] * n,
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def _qsocand(n, **overrides):
    """Purer-by-default qsocand-like frame at a periphery position."""
    df = pd.DataFrame({
        "ra": np.full(n, RA_OUT),
        "dec": np.full(n, DEC_OUT),
        "gaia_crf_source": np.zeros(n),
        "classlabel_dsc_joint": ["quasar"] * n,
        "vari_best_class_name": [""] * n,
        "host_galaxy_flag": np.full(n, np.nan),
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def test_in_mc_core_circles():
    ra = [81.3, 81.3, 16.0, 16.0, RA_OUT]
    dec = [-68.7, -68.7 + 9.5, -72.8, -72.8 + 6.5, DEC_OUT]
    # centers inside, just past each radius outside, periphery outside
    assert in_mc_core(ra, dec).tolist() == [True, False, True, False, False]
    # just inside each radius (offsets in dec are exact great-circle dists)
    assert in_mc_core([81.3, 16.0], [-68.7 + 8.9, -72.8 + 5.9]).all()


def test_galaxy_mask_purer_union():
    df = _galcand(
        4,
        radius_sersic=[1500.0, np.nan, np.nan, np.nan],
        classlabel_dsc_joint=["", "galaxy", "", ""],
        vari_best_class_name=["", "", "GALAXY", "quasar"],
    )
    assert gaia_extragal.galaxy_mask(df).tolist() == [True, True, True, False]


def test_qso_mask_purer_union():
    df = _qsocand(
        5,
        gaia_crf_source=[1, 0, 0, 0, np.nan],
        classlabel_dsc_joint=["", "quasar", "", "", ""],
        vari_best_class_name=["", "", "AGN", "", "RR"],
        host_galaxy_flag=[np.nan, np.nan, np.nan, 0, np.nan],
    )
    assert gaia_extragal.qso_mask(df).tolist() == [True, True, True, True, False]


def test_masks_exclude_mc_cores():
    gal = _galcand(2, ra=[81.3, RA_OUT], dec=[-68.7, DEC_OUT])
    qso = _qsocand(2, ra=[16.0, RA_OUT], dec=[-72.8, DEC_OUT])
    assert gaia_extragal.galaxy_mask(gal).tolist() == [False, True]
    assert gaia_extragal.qso_mask(qso).tolist() == [False, True]


def test_masks_empty_frames():
    assert not gaia_extragal.galaxy_mask(_galcand(0)).any()
    assert not gaia_extragal.qso_mask(_qsocand(0)).any()


def test_tile_adql_uses_geometry_and_purer_cut():
    pix = int(tiles.pixel_of(RA_OUT, DEC_OUT))
    for src in (GALCAND, QSOCAND):
        adql = src.tile_adql(pix)
        assert "CONTAINS" in adql and "POLYGON" in adql
        assert f"({src.where})" in adql
        assert adql.startswith(f"SELECT {', '.join(src.columns)} FROM")
