"""Feature extraction tests against the real y4t2 catalogs."""

import pandas as pd
import pytest

from morphocloud import bricks, features
from morphocloud.config import DELVEMC_DATA

pytestmark = pytest.mark.skipif(
    DELVEMC_DATA is None or not DELVEMC_DATA.exists(),
    reason="DELVE-MC catalogs not on this machine",
)

BRICKS = ["0002m587", "0830m680"]


@pytest.fixture(scope="module", params=BRICKS)
def brick_frames(request):
    objects = bricks.read_objects(request.param)
    return objects, features.brick_features(request.param, objects)


def _point_like(objects):
    return (
        (objects["PROB"] > 0.8)
        & (objects["SHARP"].abs() < 0.3)
        & (objects["RMAG"] < 20)
    )


def test_schema_and_alignment(brick_frames):
    objects, feats = brick_frames
    assert list(feats.columns) == list(features.FEATURE_COLUMNS)
    assert len(feats) == len(objects)
    assert feats.index.equals(objects.index)


def test_extinction_correction(brick_frames):
    objects, feats = brick_frames
    expected = objects["GMAG"] - features.EXTINCTION_COEFF["G"] * objects["EBV"]
    pd.testing.assert_series_equal(
        feats["GMAG0"], expected, check_names=False)
    # EBV > 0 everywhere, so corrected mags are strictly brighter
    detected = objects["GMAG"].notna()
    assert (feats.loc[detected, "GMAG0"] < objects.loc[detected, "GMAG"]).all()


def test_colors_consistent(brick_frames):
    _, feats = brick_frames
    pd.testing.assert_series_equal(
        feats["G_R"], feats["GMAG0"] - feats["RMAG0"], check_names=False)
    pd.testing.assert_series_equal(
        feats["G_I"], feats["GMAG0"] - feats["IMAG0"], check_names=False)


def test_seeing_and_fwhm_ratio(brick_frames):
    objects, feats = brick_frames
    seeing = feats["SEEING"].iloc[0]
    assert 0.5 < seeing < 3.0
    assert (feats["SEEING"] == seeing).all()
    # point sources have coadd FWHM near the exposure seeing
    ratio = feats.loc[_point_like(objects), "FWHM_RATIO"]
    assert 0.8 < ratio.median() < 2.0


def test_concentration_anchored(brick_frames):
    objects, feats = brick_frames
    # anchor population (bright point-like stars) sits at 0 by construction
    anchor = (
        (objects["PROB"] >= features.ANCHOR_PROB_MIN)
        & (objects["SHARP"].abs() <= features.ANCHOR_SHARP_MAX)
        & pd.concat(
            [objects[f"{b}MAG"].between(*features.ANCHOR_MAG_RANGE)
             for b in ("G", "R", "I", "Z")], axis=1,
        ).any(axis=1)
    )
    assert abs(feats.loc[anchor, "CONC"].median()) < 0.01
    # and the statistic separates extended from point-like morphology
    extended = feats.loc[feats["FWHM_RATIO"] > 2, "CONC"].median()
    point = feats.loc[feats["FWHM_RATIO"].between(0.8, 1.3), "CONC"].median()
    assert extended > point + 0.3


def test_concentration_without_anchors():
    # too few anchor stars in every band -> all-NaN, not a garbage zero point
    objects = pd.DataFrame({
        **{f"{b}MAG": [18.0, 19.0] for b in ("G", "R", "I", "Z")},
        "MAG_AUTO": [13.0, 14.0], "PROB": [0.9, 0.9], "SHARP": [0.0, 0.0],
    })
    assert features._concentration(objects).isna().all()
