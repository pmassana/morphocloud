"""Path-free inference: engineer_features, input-agnostic predict, smooth cut.

These run against the released weights bundled in models/ and a small synthetic
catalog — no local DELVE-MC brick files needed.
"""

import numpy as np
import pandas as pd
import pytest

from morphocloud import features, infer

pytestmark = pytest.mark.skipif(
    not infer.DEFAULT_MODEL.exists(),
    reason="released weights not present (download the v1.0.0 release assets)",
)


@pytest.fixture(scope="module")
def synthetic_features():
    """A FEATURE_COLUMNS frame built from a synthetic raw catalog (no I/O)."""
    rng = np.random.default_rng(0)
    n = 200
    raw = {}
    for b in ("G", "R", "I", "Z"):
        raw[f"{b}MAG"] = rng.uniform(18.0, 24.0, n)
        raw[f"{b}ERR"] = rng.uniform(0.01, 0.2, n)
        raw[f"{b}SCATTER"] = rng.uniform(0.0, 0.1, n)
        raw[f"NDET{b}"] = rng.integers(1, 5, n)
    raw["EBV"] = rng.uniform(0.0, 0.1, n)
    for c in ("CHI", "SHARP", "PROB", "ELLIPTICITY", "ASEMI", "BSEMI"):
        raw[c] = rng.uniform(0.0, 1.0, n)
    raw["FWHM"] = rng.uniform(3.0, 8.0, n)
    raw["MAG_AUTO"] = raw["RMAG"]  # plausible coadd auto mag
    return features.engineer_features(pd.DataFrame(raw), seeing=4.0)


@pytest.fixture(scope="module")
def clf():
    return infer.StarGalaxyClassifier.load()


def test_engineer_features_is_path_free_and_aligned(synthetic_features):
    feats = synthetic_features
    assert list(feats.columns) == list(features.FEATURE_COLUMNS)
    assert len(feats) == 200
    # the documented raw-column contract really covers everything it consumes
    assert set(features.RAW_INPUT_COLUMNS) >= {"EBV", "FWHM", "MAG_AUTO"}


def test_predict_proba_in_unit_range(clf, synthetic_features):
    raw, p = clf.predict_proba(synthetic_features)
    assert p.shape == (len(synthetic_features),)
    assert (p >= 0).all() and (p <= 1).all()
    assert np.allclose(clf.predict(synthetic_features), p)


def test_predict_input_agnostic(clf, synthetic_features):
    """DataFrame, numpy structured array and astropy Table give the same P_STAR."""
    p_df = clf.predict(synthetic_features)
    p_rec = clf.predict(synthetic_features.to_records(index=False))
    Table = pytest.importorskip("astropy.table").Table
    p_tbl = clf.predict(Table.from_pandas(synthetic_features))
    assert np.allclose(p_df, p_rec)
    assert np.allclose(p_df, p_tbl)


def test_predict_rejects_unknown_input(clf):
    with pytest.raises(TypeError):
        clf.predict([1, 2, 3])


def test_smooth_threshold_curve(clf):
    tf = clf.smooth_threshold("leak1")
    r = np.linspace(20.0, 24.0, 20)
    thr = tf(r)
    assert thr.shape == r.shape
    assert (thr > 0).all() and (thr < 1).all()
    # strictness>0 tightens (higher cut) everywhere; <0 loosens
    assert (tf(r, strictness=1.0) > thr).all()
    assert (tf(r, strictness=-1.0) < thr).all()
    # outside the table range the flat fallback is used
    assert tf(5.0, flat=0.5) == 0.5
    # scalar in -> float out
    assert isinstance(tf(22.0), float)


def test_smooth_threshold_unknown_point(clf):
    with pytest.raises(ValueError):
        clf.smooth_threshold("nonsense")
