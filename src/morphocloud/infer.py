"""Inference: a calibrated stellar probability from DELVE-MC catalog rows alone.

Loads the released XGBoost model plus its isotonic calibrator and turns a
brick's object catalog into a per-source P(star). Every input is a DELVE-MC
DR1 column or a per-brick exposure-metadata value (features.brick_features) -
no truth catalog, including DELVE DR3, is ever read - so inference runs
anywhere the object files are present.

The calibrator is stored as isotonic interpolation knots, so applying it is a
plain np.interp with no sklearn dependency at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import bricks, features
from .config import MODELS_DIR
from .features import quality_mask

DEFAULT_MODEL = MODELS_DIR / "baseline_lshsc_xgb.json"


def _to_frame(table) -> pd.DataFrame:
    """Normalize any tabular input to a pandas DataFrame.

    Accepts a pandas DataFrame (returned as-is), an astropy Table (duck-typed on
    ``to_pandas`` so astropy stays an install-time, not import-time, concern), or
    a numpy structured/record array. This lets inference run on whatever catalog
    representation the caller already has.
    """
    if isinstance(table, pd.DataFrame):
        return table
    if hasattr(table, "to_pandas"):  # astropy.table.Table and friends
        return table.to_pandas()
    if isinstance(table, np.ndarray) and table.dtype.names is not None:
        return pd.DataFrame(table)
    raise TypeError(
        "expected a pandas DataFrame, an astropy Table, or a numpy structured "
        f"array with named fields; got {type(table).__name__}"
    )


class StarGalaxyClassifier:
    """The released baseline: raw XGBoost score + isotonic calibration to P(star)."""

    # Operating-point -> threshold-table column. leak* cap the galaxy->star leak
    # (base-rate-free, the trustworthy controls); pur* are this test sample's
    # star-heavy purity points. See scripts/build_threshold_table.py.
    OPERATING_POINTS = {
        "leak0.5": "t_leak0p5", "leak1": "t_leak1", "leak2": "t_leak2",
        "pur95": "t_pur95", "pur99": "t_pur99",
    }

    def __init__(self, booster, feature_names, best_iteration, cal_x, cal_y,
                 thresholds=None):
        self.booster = booster
        self.features = list(feature_names)
        self.best_iteration = int(best_iteration)
        self.cal_x = np.asarray(cal_x, dtype=np.float64)
        self.cal_y = np.asarray(cal_y, dtype=np.float64)
        # Per-magnitude operating-point table (sibling .thresholds.csv), or None
        # if it wasn't bundled with the weights. classify_brick stays unchanged
        # and only emits P_STAR; threshold_for() reads this on demand.
        self.thresholds = thresholds

    @classmethod
    def load(cls, model_path=DEFAULT_MODEL):
        """Load model, feature list/best-iteration (.meta.json), calibrator and,
        if present, the per-magnitude threshold table (.thresholds.csv).

        When called with no path and the bundled/dev default isn't on disk, the
        released weights are downloaded once to a user cache (see weights.py)."""
        model_path = Path(model_path)
        if not model_path.exists() and model_path == DEFAULT_MODEL:
            from . import weights

            model_path = weights.fetch_weights()
        with open(model_path.with_suffix(".meta.json")) as fh:
            meta = json.load(fh)
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        with open(model_path.with_suffix(".calibrator.json")) as fh:
            cal = json.load(fh)
        thr_path = model_path.with_suffix(".thresholds.csv")
        thresholds = pd.read_csv(thr_path) if thr_path.exists() else None
        return cls(
            booster, meta["features"], meta["best_iteration"],
            cal["x_thresholds"], cal["y_thresholds"], thresholds,
        )

    def threshold_for(self, r_mag, target="leak1"):
        """Per-magnitude calibrated-P_STAR cut for an operating point.

        The new default model over-calls stars at a flat p>=0.5 faint-ward, so a
        usable star/galaxy decision is `P_STAR >= threshold_for(RMAG0)`. `r_mag`
        is the extinction-corrected r magnitude (scalar or array-like); `target`
        is one of OPERATING_POINTS (default 'leak1' = galaxy->star leak <=1%, the
        README headline). Returns NaN where r is outside the table's bins or the
        bin had too few truth sources to set a number.
        """
        if self.thresholds is None:
            raise RuntimeError(
                "no threshold table bundled with this model "
                "(expected a sibling .thresholds.csv next to the weights)"
            )
        if target not in self.OPERATING_POINTS:
            raise ValueError(
                f"unknown target {target!r}; choose from "
                f"{sorted(self.OPERATING_POINTS)}"
            )
        tab = self.thresholds
        col = self.OPERATING_POINTS[target]
        r_in = np.asarray(r_mag, dtype=np.float64)
        r = np.atleast_1d(r_in)
        r_lo, r_hi = tab["r_lo"].to_numpy(), tab["r_hi"].to_numpy()
        # idx[i] = row whose [r_lo, r_hi) bin contains r[i]; -1 if below all bins.
        idx = np.searchsorted(r_lo, r, side="right") - 1
        valid = (idx >= 0) & (r < r_hi[np.clip(idx, 0, len(tab) - 1)])
        out = np.full(r.shape, np.nan, dtype=np.float64)
        out[valid] = tab[col].to_numpy()[idx[valid]]
        return out if r_in.ndim else float(out[0])

    def smooth_threshold(self, operating_point="leak1", degree=2,
                         weight_by_counts=True, eps=1e-4):
        """A smooth P_STAR-vs-magnitude cut T(r), as a callable.

        The per-magnitude table gives a calibrated P_STAR cut per r-bin for an
        operating point, but the values are noisy bin-to-bin and pile up against
        1 at the faint end. This fits a degree-`degree` polynomial in LOGIT space
        (so the curve stays in (0, 1)) and returns a closure
        ``threshold_for(rmag0, strictness=0.0, flat=None)`` -> cut(s) shaped like
        `rmag0`. `strictness` is a single logit-space dial: > 0 tightens, < 0
        loosens the cut everywhere without saturating at the ceiling. Inside the
        table r-range the fitted curve is returned; outside it returns `flat`
        (or, if `flat is None`, the curve clamped at the nearest edge).

        This is the smooth replacement for the step lookup in `threshold_for`;
        use `P_STAR >= smooth_threshold('leak1')(RMAG0)` for a star/galaxy call.
        """
        if self.thresholds is None:
            raise RuntimeError(
                "no threshold table bundled with this model "
                "(expected a sibling .thresholds.csv next to the weights)"
            )
        if operating_point not in self.OPERATING_POINTS:
            raise ValueError(
                f"unknown operating_point {operating_point!r}; choose from "
                f"{sorted(self.OPERATING_POINTS)}"
            )
        tab = self.thresholds
        col = self.OPERATING_POINTS[operating_point]
        r_mid = 0.5 * (tab["r_lo"].to_numpy() + tab["r_hi"].to_numpy())
        t = np.clip(tab[col].to_numpy(dtype=float), eps, 1.0 - eps)
        y = np.log(t / (1.0 - t))  # logit
        w = None
        if weight_by_counts and {"n_gal", "n_star"} <= set(tab.columns):
            w = np.sqrt(tab["n_gal"].to_numpy() + tab["n_star"].to_numpy())
        poly = np.poly1d(np.polyfit(r_mid, y, deg=degree, w=w))
        r_min, r_max = float(tab["r_lo"].min()), float(tab["r_hi"].max())

        def threshold_for(rmag0, strictness=0.0, flat=None):
            r = np.atleast_1d(np.asarray(rmag0, dtype=float))
            out = 1.0 / (1.0 + np.exp(-(poly(np.clip(r, r_min, r_max)) + strictness)))
            if flat is not None:
                outside = (r < r_min) | (r >= r_max) | ~np.isfinite(r)
                out = np.where(outside, flat, out)
            return out if np.ndim(rmag0) else float(out[0])

        return threshold_for

    def predict_proba(self, table):
        """Return (raw_score, calibrated P(star)) for a feature table.

        `table` is any input `_to_frame` accepts (pandas DataFrame, astropy
        Table, or numpy structured array) carrying the model's feature columns
        (features.FEATURE_COLUMNS). Missing features stay NaN - XGBoost routes
        them natively, exactly as in training. A plain DMatrix is exact here:
        quantization only affected training-time split-finding, not the learned
        tree thresholds.
        """
        feats = _to_frame(table)
        X = feats[self.features].to_numpy(dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=self.features)
        raw = self.booster.predict(
            dmat, iteration_range=(0, self.best_iteration + 1)
        )
        return raw, np.interp(raw, self.cal_x, self.cal_y)

    def predict(self, table) -> np.ndarray:
        """Calibrated P(star) for a feature table (see `predict_proba`)."""
        return self.predict_proba(table)[1]

    def classify_brick(self, brickname: str, unique_only: bool = False) -> pd.DataFrame:
        """Per-source classification table for one brick.

        Columns: BRICKNAME, OBJID, RA, DEC, BRICKUNIQ, RMAG0, P_STAR
        (calibrated), P_STAR_RAW, QUALITY_PASS. RMAG0 (extinction-corrected r) is
        carried so threshold_for() can be applied to the output directly.
        QUALITY_PASS marks the >=2-good-band cut the model was trained under:
        rows that fail it get a probability anyway but are outside the validated
        regime. unique_only=False keeps every source (BRICKUNIQ rides along for
        cross-brick dedup downstream).
        """
        objects = bricks.read_objects(brickname, unique_only=unique_only)
        feats = features.brick_features(brickname, objects)
        raw, prob = self.predict_proba(feats)
        out = objects[["BRICKNAME", "OBJID", "RA", "DEC", "BRICKUNIQ"]].copy()
        out["RMAG0"] = feats["RMAG0"].to_numpy(dtype=np.float32)
        out["P_STAR"] = prob.astype(np.float32)
        out["P_STAR_RAW"] = raw.astype(np.float32)
        out["QUALITY_PASS"] = quality_mask(objects).to_numpy()
        return out.reset_index(drop=True)
