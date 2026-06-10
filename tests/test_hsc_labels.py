"""Unit tests for the HSC CI masks and the blend-aware component matching."""

import numpy as np
import pandas as pd

from morphocloud.labels import hsc
from morphocloud.labels.assemble import MATCH_RADIUS_ARCSEC, _component_flags


def _truth(ci, optical=None, ci_sigma=None):
    """HSC-like frame; consistent CI and an F814W detection unless overridden."""
    df = pd.DataFrame({"ci": ci})
    df["ci_sigma"] = (0.05 * df["ci"]) if ci_sigma is None else ci_sigma
    for col in hsc.OPTICAL_MAGS:
        df[col] = np.nan
    df["a_f814w"] = np.where(
        np.ones(len(df), bool) if optical is None else optical, 24.0, np.nan)
    return df


def test_ci_masks_disjoint_and_nan_safe():
    df = _truth([0.9, hsc.CI_STAR_MAX, 1.3, 2.0, np.nan, -1.0])
    point = hsc.point_source_mask(df)
    ext = hsc.galaxy_mask(df)
    assert point.tolist() == [True, False, False, False, False, False]
    assert ext.tolist() == [False, False, False, True, False, False]
    assert not (point & ext).any()


def test_galaxy_mask_requires_optical():
    df = _truth([2.0, 2.0], optical=[True, False])
    assert hsc.galaxy_mask(df).tolist() == [True, False]


def test_galaxy_mask_crowding_guard():
    # inconsistent CI across images (crowding-inflated star) or NaN -> no label
    df = _truth([2.0, 2.0, 2.0], ci_sigma=[0.1, 0.66, np.nan])
    assert hsc.galaxy_mask(df).tolist() == [True, False, False]


def _flags(obj_pos, comp_pos, comp_ci):
    """Run _component_flags on synthetic positions near dec -60."""
    objects = pd.DataFrame({
        "RA": [p[0] for p in obj_pos], "DEC": [p[1] for p in obj_pos]})
    truth = _truth(comp_ci)
    truth["matchra"] = [p[0] for p in comp_pos]
    truth["matchdec"] = [p[1] for p in comp_pos]
    return _component_flags(
        objects, truth, hsc.point_source_mask(truth), hsc.galaxy_mask(truth),
        "matchra", "matchdec")


def test_component_flags_blend_rules():
    eps = 0.1 / 3600  # 0.1" in dec
    objs = [(10.0, -60.0), (10.1, -60.0), (10.2, -60.0),
            (10.3, -60.0), (10.4, -60.0), (10.5, -60.0)]
    comps = [
        (10.0, -60.0),              # obj0: single point source -> star flag
        (10.1, -60.0),              # obj1: single extended -> galaxy
        (10.2, -60.0),              # obj2: point + extended -> blend
        (10.2, -60.0 + eps),
        (10.4, -60.0),              # obj4: two point sources -> star flag
        (10.4, -60.0 + eps),
        (10.5, -60.0),              # obj5: ambiguous CI -> blend, no label
    ]
    ci = [1.0, 2.0, 1.0, 2.0, 1.0, 1.05, 1.4]
    star, galaxy, blend = _flags(objs, comps, ci)
    assert star.tolist() == [True, False, False, False, True, False]
    assert galaxy.tolist() == [False, True, False, False, False, False]
    assert blend.tolist() == [False, False, True, False, False, True]


def test_component_outside_radius_ignored():
    far = 2 * MATCH_RADIUS_ARCSEC / 3600
    star, galaxy, blend = _flags(
        [(10.0, -60.0)], [(10.0, -60.0 + far)], [1.0])
    assert not (star.any() or galaxy.any() or blend.any())


def test_empty_truth():
    star, galaxy, blend = _flags([(10.0, -60.0)], [], [])
    assert not (star.any() or galaxy.any() or blend.any())
