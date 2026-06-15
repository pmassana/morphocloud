"""Smoke tests for the brick readers against the real DR1 catalogs."""

import numpy as np
import pytest

from morphocloud import bricks
from morphocloud.config import CORE_BANDS, DELVEMC_DATA

pytestmark = pytest.mark.skipif(
    DELVEMC_DATA is None or not DELVEMC_DATA.exists(),
    reason="DELVE-MC catalogs not on this machine",
)

# 0002m587 has Y-band columns; 0830m680 (dense, LMC) has no Y at all
BRICKS = ["0002m587", "0830m680"]


def test_brick_list():
    df = bricks.load_brick_list()
    assert len(df) == 35052
    assert df["BRICKNAME"].str.len().eq(8).all()


@pytest.mark.parametrize("brickname", BRICKS)
def test_normalized_schema(brickname):
    df = bricks.read_objects(brickname)
    for band in CORE_BANDS:
        for col in (f"{band}MAG", f"{band}ERR", f"{band}SCATTER", f"NDET{band}"):
            assert col in df.columns
    # sentinels are gone
    assert df["GMAG"].max(skipna=True) < 40
    assert not np.any(df["FWHM"] > 1e5)
    assert df["BRICKUNIQ"].all()
    # undetected sources have NaN mags
    nodet = df["NDETG"] == 0
    if nodet.any():
        assert df.loc[nodet, "GMAG"].isna().all()


def test_schemas_identical_across_bricks():
    cols = [list(bricks.read_objects(b).columns) for b in BRICKS]
    assert cols[0] == cols[1]


@pytest.mark.parametrize("brickname", BRICKS)
def test_brick_seeing(brickname):
    seeing = bricks.brick_seeing(brickname)
    assert seeing, "no per-band seeing measured"
    for band, fwhm in seeing.items():
        assert 0.5 < fwhm < 3.0, (band, fwhm)
