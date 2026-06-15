"""Out-of-core XGBoost star/galaxy classifier.

Streams the labelled parquet in row-group batches via an xgboost DataIter into
an ExtMemQuantileDMatrix, which bins each batch and spills it to an on-disk page
cache (released from RAM). Peak memory is one raw batch plus the resident
histogram pages, independent of table size -- needed since the LS/HSC-augmented
table is ~169M rows and the in-core QuantileDMatrix no longer fits 18 GB. Trains
a histogram GBDT on the photometric and morphological features, early-stopping
on the spatially disjoint val split.

Two column groups are deliberately NOT model inputs:
  - NDET<band>: counts how many times a source was observed, a survey-cadence /
    depth property tied to footprint geometry, not to whether it is a star or a
    galaxy. Used for the quality cut (dataset.quality_mask), never as a feature.
  - provenance flags (GAIA_STAR, DR3_STAR, LS_GALAXY, ...): encode which truth
    catalog labelled the row, so feeding them would let the model read the
    answer off catalog membership instead of learning morphology.

This is the "high-confidence first" baseline: the clean consensus LABEL as
assembled (conflicts already dropped), DR3 included so the faint end keeps
labels. The DR3-distillation ablation and probability calibration are separate,
later steps. Usage:

    python scripts/train_baseline.py [--rounds N] [--batch-size N]
                                     [--threads N] [--out PATH]
                                     [--dmatrix extmem|incore]

--threads sets the CPU cores used for both quantization and training
(-1, the default, uses all cores). --dmatrix picks the matrix backend:
extmem (default) bins to an on-disk cache and bounds RAM; incore holds the
whole binned matrix in RAM (faster when RAM >> dataset or the disk is slow).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

from morphocloud.config import DATA_DIR, MODELS_DIR
from morphocloud.features import FEATURE_COLUMNS

DATASET_PATH = DATA_DIR / "train" / "dataset.parquet"

# observation-count columns: a property of the survey, not the source
EXCLUDE_FEATURES = tuple(f"NDET{b}" for b in ("G", "R", "I", "Z"))
MODEL_FEATURES = tuple(c for c in FEATURE_COLUMNS if c not in EXCLUDE_FEATURES)

PARAMS = {
    "objective": "binary:logistic",   # positive class = star (LABEL == 1)
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",
    "max_depth": 8,
    "eta": 0.3,   # XGBoost default; 0.1 converged too slowly (marginal, unstable)
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20.0,
    "max_bin": 256,
}


class ParquetSplitIter(xgb.DataIter):
    """Yields (features, label) batches for one SPLIT, streamed from parquet.

    Reads only the feature columns plus LABEL and SPLIT and filters each
    row-group batch to the requested split, so the full table never lands in
    memory at once.
    """

    def __init__(self, path, split, features, batch_size=2_000_000,
                 progress=True, cache_prefix=None, release_data=True):
        self._pf = pq.ParquetFile(path)
        self._split = split
        self._features = list(features)
        self._columns = self._features + ["LABEL", "SPLIT"]
        self._batch_size = batch_size
        self._batches = None
        self._progress = progress
        self._rows = 0       # split rows emitted in the current pass
        self._t0 = None      # wall-clock start of the current pass
        # Two modes set release_data/cache_prefix from train():
        #  - extmem (ExtMemQuantileDMatrix): cache_prefix set, release_data=True
        #    -> each batch is binned, spilled to the on-disk cache (on_host=False
        #    forces the filesystem) and freed from RAM. Peak memory is one raw
        #    batch + resident histogram pages, independent of table size.
        #  - in-core (QuantileDMatrix): cache_prefix=None, release_data=False
        #    -> the default (True) frees this iterator's batches once cuts are
        #    built, which silently breaks the val matrix built with ref=dtrain
        #    (training then makes zero splits).
        super().__init__(cache_prefix=cache_prefix, release_data=release_data,
                         on_host=False)

    def reset(self):
        self._batches = self._pf.iter_batches(
            columns=self._columns, batch_size=self._batch_size
        )
        self._rows = 0
        self._t0 = time.time()

    def next(self, input_data):
        if self._batches is None:
            self.reset()
        for rb in self._batches:
            split = rb.column("SPLIT").to_numpy(zero_copy_only=False)
            mask = split == self._split
            if not mask.any():
                continue
            cols = [
                rb.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                for c in self._features
            ]
            X = np.column_stack(cols)[mask]
            y = rb.column("LABEL").to_numpy(zero_copy_only=False).astype(np.float32)
            y = y[mask]
            # feature names ride along with each batch (the DataIter ctor in
            # this xgboost version has no feature_names argument)
            input_data(data=X, label=y, feature_names=self._features)
            self._rows += len(y)
            if self._progress:
                dt = time.time() - self._t0
                rate = self._rows / dt / 1e6 if dt else 0.0
                print(
                    f"  [{self._split}] {self._rows / 1e6:6.1f}M rows "
                    f"[{dt:4.0f}s, {rate:.1f}M/s]",
                    flush=True,
                )
            return True
        return False


def train(rounds, batch_size, out_path, nthread=-1, extmem=True):
    cores = nthread if nthread > 0 else "all"
    print(f"features ({len(MODEL_FEATURES)}): {', '.join(MODEL_FEATURES)}")
    print(f"excluded as inputs: {', '.join(EXCLUDE_FEATURES)} + provenance flags")
    print(f"cpu cores: {cores}  |  dmatrix: {'extmem' if extmem else 'incore'}\n")

    t0 = time.time()
    max_bin = PARAMS["max_bin"]
    cache_dir = None
    if extmem:
        # RAM-bounded: bin each batch and spill to an on-disk page cache. Best
        # for small RAM / fast SSD. release_data=True frees raw batches.
        cache_dir = DATASET_PATH.parent / "extmem_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        train_it = ParquetSplitIter(DATASET_PATH, "train", MODEL_FEATURES,
                                    batch_size, cache_prefix=str(cache_dir / "train"))
        val_it = ParquetSplitIter(DATASET_PATH, "val", MODEL_FEATURES,
                                  batch_size, cache_prefix=str(cache_dir / "val"))
        print("building train ExtMemQuantileDMatrix (binned cache -> disk)...", flush=True)
        dtrain = xgb.ExtMemQuantileDMatrix(train_it, nthread=nthread, max_bin=max_bin)
        print("building val ExtMemQuantileDMatrix...", flush=True)
        dval = xgb.ExtMemQuantileDMatrix(val_it, ref=dtrain, nthread=nthread,
                                         max_bin=max_bin)
    else:
        # In-core: the whole binned matrix lives in RAM (faster, no disk I/O per
        # round). Use when RAM >> dataset or the disk is slow (e.g. a 64 GB
        # server with an HDD). release_data=False is required here -- the default
        # frees the iterator's batches once cuts are built, which silently breaks
        # the val matrix built with ref=dtrain (training then makes zero splits).
        train_it = ParquetSplitIter(DATASET_PATH, "train", MODEL_FEATURES,
                                    batch_size, release_data=False)
        val_it = ParquetSplitIter(DATASET_PATH, "val", MODEL_FEATURES,
                                  batch_size, release_data=False)
        print("quantizing train split (in-core, streamed)...", flush=True)
        dtrain = xgb.QuantileDMatrix(train_it, nthread=nthread, max_bin=max_bin)
        print("quantizing val split (in-core, streamed)...", flush=True)
        dval = xgb.QuantileDMatrix(val_it, ref=dtrain, nthread=nthread,
                                   max_bin=max_bin)

    # class balance read off the built matrices (reliable; the iterator's own
    # tallies are unsafe because xgboost may reset it after the final pass)
    ytr = dtrain.get_label()
    yval = dval.get_label()
    n_train_star, n_train_gal = int((ytr == 1).sum()), int((ytr == 0).sum())
    n_val_star, n_val_gal = int((yval == 1).sum()), int((yval == 0).sum())
    spw = n_train_gal / n_train_star  # n_galaxy / n_star, positive class = star
    params = {**PARAMS, "scale_pos_weight": spw, "nthread": nthread}
    print(
        f"\ntrain: {n_train_star:,} star / {n_train_gal:,} galaxy "
        f"(scale_pos_weight={spw:.4f})"
    )
    print(f"val:   {n_val_star:,} star / {n_val_gal:,} galaxy")
    print(f"matrices ready in {time.time() - t0:.0f}s\n")

    evals_result: dict = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=10,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bst.save_model(out_path)
    meta = {
        "features": list(MODEL_FEATURES),
        "excluded_features": list(EXCLUDE_FEATURES),
        "label": "LABEL (1=star, 0=galaxy)",
        "params": params,
        "best_iteration": int(bst.best_iteration),
        "best_val_auc": float(evals_result["val"]["auc"][bst.best_iteration]),
        "n_train_star": n_train_star,
        "n_train_galaxy": n_train_gal,
        "n_val_star": n_val_star,
        "n_val_galaxy": n_val_gal,
    }
    meta_path = out_path.replace(".json", ".meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(
        f"\nbest iter {bst.best_iteration}  "
        f"val AUC {meta['best_val_auc']:.5f}  logloss "
        f"{evals_result['val']['logloss'][bst.best_iteration]:.5f}"
    )
    print(f"model -> {out_path}")
    print(f"meta  -> {meta_path}")
    print("\ntop features by gain:")
    gain = bst.get_score(importance_type="gain")
    for name, score in sorted(gain.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {name:14s} {score:12.1f}")

    if cache_dir is not None:
        shutil.rmtree(cache_dir, ignore_errors=True)  # reclaim the on-disk cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=2_000_000)
    ap.add_argument(
        "--threads", type=int, default=-1, dest="nthread",
        help="CPU cores for quantization and training (-1 = all available)",
    )
    ap.add_argument(
        "--out", default=str(MODELS_DIR / "baseline_xgb.json"),
        help="model output path (.json)",
    )
    ap.add_argument(
        "--dmatrix", choices=["extmem", "incore"], default="extmem",
        help="extmem (default): RAM-bounded, bins to an on-disk page cache -- "
             "best for small RAM and/or a fast SSD. incore: holds the whole "
             "binned matrix in RAM (faster, no per-round disk I/O) -- use when "
             "RAM >> dataset or the disk is slow (e.g. a 64 GB server with HDD).",
    )
    args = ap.parse_args()
    train(args.rounds, args.batch_size, args.out, args.nthread,
          extmem=(args.dmatrix == "extmem"))


if __name__ == "__main__":
    main()
