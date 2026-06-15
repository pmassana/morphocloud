"""Dataset assembly tests: split determinism, quality cuts, brick worker."""

import numpy as np
import pandas as pd
import pytest

from morphocloud import dataset, features
from morphocloud.config import DATA_DIR, DELVEMC_DATA
from morphocloud.labels import assemble, lsdr10

# bricks whose label tiles (hp32-08298 / 08329) are cached for offline tests:
# 0496m662 = deep periphery field, 0135m725 = SMC bar (densest HST pointing)
PERIPHERY, SMC_BAR = "0496m662", "0135m725"

needs_data = pytest.mark.skipif(
    not (
        DELVEMC_DATA is not None
        and DELVEMC_DATA.exists()
        and (DATA_DIR / "labels/hsc_v3/hp32-08298.parquet").exists()
        and (DATA_DIR / "labels/hsc_v3/hp32-08329.parquet").exists()
        and (DATA_DIR / "labels/ls_dr10_mask/hp32-08298.parquet").exists()
        and (DATA_DIR / "labels/ls_dr10_mask/hp32-08329.parquet").exists()
    ),
    reason="DELVE-MC catalogs or cached label tiles not on this machine",
)


def test_masked_mask_bits():
    # artifact bits (BRIGHT, SATUR, ALLMASK, WISEM, BAILOUT) flag a region;
    # NPRIMARY/MEDIUM/GALAXY/CLUSTER/SUB_BLOB do not. A non-artifact bit
    # alongside a real artifact bit still counts as masked.
    df = pd.DataFrame({"maskbits": [
        0, 1 << 0, 1 << 1, 1 << 11, 1 << 12, 1 << 13, 1 << 16,
        (1 << 13) | (1 << 1)]})
    assert lsdr10.masked_mask(df).tolist() == [
        False, False, True, False, False, False, False, True]


def test_in_artifact_mask_many_to_one():
    # three objects within 1" of one masked row are all flagged (region flag,
    # not the one-to-one label matcher); a far object is not
    objects = pd.DataFrame({
        "RA": [10.0, 10.00005, 10.0001, 20.0], "DEC": [0.0, 0.0, 0.0, 0.0]})
    masked = pd.DataFrame({"ra": [10.0], "dec": [0.0], "maskbits": [1 << 1]})
    assert assemble._in_artifact_mask(objects, masked).tolist() == [
        True, True, True, False]
    # a GALAXY-only masked row is not an artifact -> nothing flagged
    real = pd.DataFrame({"ra": [10.0], "dec": [0.0], "maskbits": [1 << 12]})
    assert not assemble._in_artifact_mask(objects, real).any()


def test_pixel_split_deterministic_fractions():
    splits = pd.Series([dataset._pixel_split(p) for p in range(10_000)])
    assert dataset._pixel_split(42) == dataset._pixel_split(42)
    frac = splits.value_counts(normalize=True)
    for name, expected in dataset.SPLIT_FRACTIONS:
        assert abs(frac[name] - expected) < 0.02, (name, frac[name])


def test_quality_mask():
    objects = pd.DataFrame({
        "NDETG": [1, 1, 0, 1], "GERR": [0.05, 0.05, np.nan, 0.05],
        "NDETR": [1, 0, 1, 1], "RERR": [0.05, np.nan, 0.05, 0.9],
        "NDETI": [0, 0, 0, 0], "IERR": [np.nan] * 4,
    })
    # 2 good bands / 1 good / 1 good / 1 good (RERR too large)
    assert dataset.quality_mask(objects).tolist() == [True, False, False, False]


@needs_data
def test_brick_split_consistent():
    split = dataset.brick_split(PERIPHERY)
    assert split in ("train", "val", "test")
    assert dataset.brick_split(PERIPHERY) == split


@needs_data
def test_brick_dataset_periphery():
    df = dataset.brick_dataset(PERIPHERY)
    assert len(df) > 0
    expected = [
        *dataset.ID_COLUMNS, "SPLIT", "HST_CONFLICT_RATE", "LABEL",
        *assemble.PROVENANCE_COLUMNS, "IN_MC_CORE", *features.FEATURE_COLUMNS,
    ]
    assert sorted(df.columns) == sorted(expected)
    assert df["LABEL"].isin([assemble.STAR, assemble.GALAXY]).all()
    assert (df["SPLIT"] == df["SPLIT"].iloc[0]).all()
    # every labelled row is backed by at least one voting truth survey
    # (HST_BLEND and GAIA_QSO are accounting-only flags, never votes)
    backing = df[list(assemble.PROVENANCE_COLUMNS)].drop(
        columns=["HST_BLEND", "GAIA_QSO"])
    assert backing.any(axis=1).all()
    # no star label co-claimed as galaxy survived (conflicts dropped)
    stars = df["LABEL"] == assemble.STAR
    galaxy_votes = ["GAIA_GALAXY", "LS_GALAXY", "DR3_GALAXY", "HST_GALAXY"]
    assert not df.loc[stars, galaxy_votes].any(axis=1).any()
    assert df["RMAG0"].notna().mean() > 0.5


@needs_data
def test_brick_dataset_hst_noise_guard():
    # the SMC-bar brick has a high HST<->Gaia conflict rate (validated
    # 2026-06-09: 127/599), so HST-only galaxy labels must be dropped
    df = dataset.brick_dataset(SMC_BAR)
    assert df["HST_CONFLICT_RATE"].iloc[0] > dataset.MAX_HST_CONFLICT
    hst_only = df["HST_GALAXY"] & ~df["LS_GALAXY"] & ~df["DR3_GALAXY"]
    assert not hst_only.any()
    # the guard off -> the same rows come back in
    df_raw = dataset.brick_dataset(SMC_BAR, max_hst_conflict=1.0)
    assert (df_raw["HST_GALAXY"] & ~df_raw["LS_GALAXY"]
            & ~df_raw["DR3_GALAXY"]).any()


@needs_data
def test_split_summary():
    df = dataset.brick_dataset(PERIPHERY)
    summary = dataset.split_summary(df)
    assert summary.loc[df["SPLIT"].iloc[0], "N"] == len(df)
    assert 0 <= summary["F_STAR"].iloc[0] <= 1
