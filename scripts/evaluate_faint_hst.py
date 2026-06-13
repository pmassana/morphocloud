"""Faint-end star/galaxy evaluation on held-out HSC (HST) truth.

Scores the new (LS+HSC-label) model against the existing baseline on the part of
the catalog HST can adjudicate, in the spatial *test* split (held out from
training) and outside the crowded MC cores (where HSC CI is reliable).

Truth = HSC classification:
- galaxy  : HST_GALAXY  (ci>1.6 + guard)  -- reliable
- star    : HST_STAR    (isolated point-like) -- ~few% galaxy-leaky even in clean
            fields, so the *galaxy*-side metric (galaxy->star leak) is the
            trustworthy one; star completeness is reported with that caveat.

Both models output P(star); compared at raw threshold 0.5 and via ROC AUC per
magnitude bin. Usage:
    python scripts/evaluate_faint_hst.py \
        --new models/baseline_lshsc_xgb.json --base models/baseline_xgb.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from morphocloud.config import DATA_DIR

DATASET = DATA_DIR / "train" / "dataset.parquet"
REPORTS = DATA_DIR.parent / "reports"


def load_model(path):
    with open(path.replace(".json", ".meta.json")) as fh:
        meta = json.load(fh)
    bst = xgb.Booster()
    bst.load_model(path)
    return bst, meta["features"], meta["best_iteration"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="models/baseline_lshsc_xgb.json")
    ap.add_argument("--base", default="models/baseline_xgb.json")
    args = ap.parse_args()

    bst_new, feats, it_new = load_model(args.new)
    bst_base, feats_b, it_base = load_model(args.base)
    cols = sorted(set(feats) | set(feats_b) |
                  {"SPLIT", "HST_STAR", "HST_GALAXY", "IN_MC_CORE", "RMAG0"})
    # push down: test split AND an HSC label
    df = pd.read_parquet(DATASET, columns=cols, filters=[
        [("SPLIT", "==", "test"), ("HST_GALAXY", "==", True)],
        [("SPLIT", "==", "test"), ("HST_STAR", "==", True)],
    ])
    df = df[~df["IN_MC_CORE"]]
    # truth: galaxy if HSC galaxy; else star. (mutually exclusive by construction)
    df = df[df["HST_GALAXY"] | df["HST_STAR"]]
    y = np.where(df["HST_GALAXY"], 0, 1)  # 1 = star
    print(f"eval rows (test, non-core, HSC-labelled): {len(df):,} "
          f"({int((y==1).sum()):,} star / {int((y==0).sum()):,} galaxy)")

    def pstar(bst, fe, it):
        d = xgb.DMatrix(df[fe].to_numpy(np.float32), feature_names=fe)
        return bst.predict(d, iteration_range=(0, it + 1))
    p_new = pstar(bst_new, feats, it_new)
    p_base = pstar(bst_base, feats_b, it_base)

    bins = np.arange(20, 25.01, 0.5)
    rmag = df["RMAG0"].to_numpy()
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rmag >= lo) & (rmag < hi)
        if m.sum() < 20:
            continue
        ym, gm = y[m], (y[m] == 0)  # gm: true galaxies
        rec = {"r": (lo + hi) / 2, "n": int(m.sum()),
               "n_gal": int(gm.sum()), "n_star": int((ym == 1).sum())}
        for tag, p in [("new", p_new[m]), ("base", p_base[m])]:
            pred_star = p >= 0.5
            # galaxy->star leak (reliable): of true galaxies, frac called star
            rec[f"leak_{tag}"] = (float(pred_star[gm].mean())
                                  if gm.any() else np.nan)
            # star completeness: of true stars, frac called star (caveated truth)
            rec[f"compl_{tag}"] = (float(pred_star[ym == 1].mean())
                                   if (ym == 1).any() else np.nan)
            # star-sample purity among HSC-labelled
            ps = pred_star
            rec[f"pur_{tag}"] = (float((ym[ps] == 1).mean())
                                 if ps.any() else np.nan)
            rec[f"auc_{tag}"] = (roc_auc_score(ym, p)
                                 if len(np.unique(ym)) == 2 else np.nan)
        rows.append(rec)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== faint-end metrics vs r (test, HSC truth) ===")
    print(tab.round(3).to_string(index=False))

    # plot the two headline curves: galaxy->star leak and star completeness
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(tab["r"], tab["leak_base"], "o-", label="baseline")
    ax[0].plot(tab["r"], tab["leak_new"], "s-", label="new (LS+HSC)")
    ax[0].set_title("galaxy->star leak (HSC galaxies called star)\nLOWER is better")
    ax[0].set_xlabel("r"); ax[0].set_ylabel("contamination fraction"); ax[0].legend()
    ax[1].plot(tab["r"], tab["compl_base"], "o-", label="baseline")
    ax[1].plot(tab["r"], tab["compl_new"], "s-", label="new (LS+HSC)")
    ax[1].set_title("star completeness (HSC stars called star)\nHIGHER is better")
    ax[1].set_xlabel("r"); ax[1].set_ylabel("completeness"); ax[1].legend()
    fig.tight_layout()
    out = REPORTS / "faint_hst_eval.png"
    fig.savefig(out)
    print(f"\nplot -> {out}")
    tab.to_csv(REPORTS / "faint_hst_eval.csv", index=False)


if __name__ == "__main__":
    main()
