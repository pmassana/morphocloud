"""Inference: a calibrated stellar probability from DELVE-MC catalog rows alone.

Loads the released XGBoost baseline plus its isotonic calibrator and turns a
brick's object catalog into a per-source P(star). Every input is a DELVE-MC
y4t2 column or a per-brick exposure-metadata value (features.brick_features) -
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
from .dataset import quality_mask

DEFAULT_MODEL = MODELS_DIR / "baseline_xgb.json"


class StarGalaxyClassifier:
    """The released baseline: raw XGBoost score + isotonic calibration to P(star)."""

    def __init__(self, booster, feature_names, best_iteration, cal_x, cal_y):
        self.booster = booster
        self.features = list(feature_names)
        self.best_iteration = int(best_iteration)
        self.cal_x = np.asarray(cal_x, dtype=np.float64)
        self.cal_y = np.asarray(cal_y, dtype=np.float64)

    @classmethod
    def load(cls, model_path=DEFAULT_MODEL):
        """Load model, feature list/best-iteration (.meta.json) and calibrator."""
        model_path = Path(model_path)
        with open(model_path.with_suffix(".meta.json")) as fh:
            meta = json.load(fh)
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        with open(model_path.with_suffix(".calibrator.json")) as fh:
            cal = json.load(fh)
        return cls(
            booster, meta["features"], meta["best_iteration"],
            cal["x_thresholds"], cal["y_thresholds"],
        )

    def predict_proba(self, feats: pd.DataFrame):
        """Return (raw_score, calibrated P(star)) for a FEATURE_COLUMNS frame.

        Missing features stay NaN - XGBoost routes them natively, exactly as in
        training. A plain DMatrix is exact here: quantization only affected
        training-time split-finding, not the learned tree thresholds.
        """
        X = feats[self.features].to_numpy(dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=self.features)
        raw = self.booster.predict(
            dmat, iteration_range=(0, self.best_iteration + 1)
        )
        return raw, np.interp(raw, self.cal_x, self.cal_y)

    def classify_brick(self, brickname: str, unique_only: bool = False) -> pd.DataFrame:
        """Per-source classification table for one brick.

        Columns: BRICKNAME, OBJID, RA, DEC, BRICKUNIQ, P_STAR (calibrated),
        P_STAR_RAW, QUALITY_PASS. QUALITY_PASS marks the >=2-good-band cut the
        model was trained under: rows that fail it get a probability anyway but
        are outside the validated regime. unique_only=False keeps every source
        (BRICKUNIQ rides along for cross-brick dedup downstream).
        """
        objects = bricks.read_objects(brickname, unique_only=unique_only)
        feats = features.brick_features(brickname, objects)
        raw, prob = self.predict_proba(feats)
        out = objects[["BRICKNAME", "OBJID", "RA", "DEC", "BRICKUNIQ"]].copy()
        out["P_STAR"] = prob.astype(np.float32)
        out["P_STAR_RAW"] = raw.astype(np.float32)
        out["QUALITY_PASS"] = quality_mask(objects).to_numpy()
        return out.reset_index(drop=True)
