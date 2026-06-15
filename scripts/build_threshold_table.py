"""Build the per-magnitude operating-point threshold table for the release.

The LS+HSC model is a better faint discriminator than the old baseline, but at a
flat calibrated p>=0.5 the star-heavy training base rate makes it over-call stars
faint-ward (galaxy->star leak climbs past r~23). This script turns that AUC gain
into a usable operating point: for each r-mag bin it tabulates the
calibrated-probability threshold that hits several targets, so the release can
pick a point per science need instead of shipping a single 0.5 cut.

Truth = held-out HSC (HST) classification, the only valid *faint* truth, on the
spatial TEST split outside the MC cores (where HSC CI is reliable):
  - galaxy : HST_GALAXY (ci>1.6 + guard) -- reliable bright AND faint.
  - star   : HST_STAR (isolated point-like) -- ~few% galaxy-leaky, so star-side
             numbers (completeness) carry that caveat; the *leak* columns rest
             only on the reliable galaxy truth and are the trustworthy ones.

Per r-bin we emit two families of thresholds on the calibrated P(star):

  leak-target   t@leak{0.5,1,2}%  -- lowest threshold whose galaxy->star leak
      (FPR = fraction of true galaxies called star) is <= the target. This is
      base-rate-independent: it controls galaxy contamination directly, the side
      we can trust. `compl@leak*` is the star completeness (TPR) you keep there.

  purity-target t@pur{95,99}%     -- lowest threshold reaching the target star
      PURITY on this sample. NOTE: purity = TP/(TP+FP) depends on the star:galaxy
      base rate, and this HSC selection is star-heavy, so read these as the
      "test-set" operating point. For a real catalog bin with star fraction pi,
      recover purity from the base-rate-free columns:
          purity(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR)
      with TPR = compl@leak, FPR = the leak target. `f_star` (this sample) and
      the full-TEST base-rate proxy are printed so pi can be plugged in.

Output: reports/threshold_table.csv (+ console table). Usage:
    python scripts/build_threshold_table.py --model models/baseline_lshsc_xgb.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from morphocloud.config import DATA_DIR

DATASET = DATA_DIR / "train" / "dataset.parquet"
REPORTS = DATA_DIR.parent / "reports"

LEAK_TARGETS = (0.005, 0.01, 0.02)      # galaxy->star FPR caps
PURITY_TARGETS = (0.95, 0.99)           # star-purity targets (this sample's pi)
RMAG_EDGES = np.arange(20.0, 25.01, 0.5)
MIN_GAL, MIN_STAR = 30, 30               # min per-class counts to emit a number


def load_model(path):
    with open(path.replace(".json", ".meta.json")) as fh:
        meta = json.load(fh)
    bst = xgb.Booster()
    bst.load_model(path)
    cal = None
    cal_path = path.replace(".json", ".calibrator.json")
    if os.path.exists(cal_path):
        c = json.load(open(cal_path))
        cal = (np.asarray(c["x_thresholds"]), np.asarray(c["y_thresholds"]))
    return bst, meta["features"], int(meta["best_iteration"]), cal


def threshold_for_leak(p_gal, target):
    """Lowest threshold t with fraction(p_gal >= t) <= target, and that fraction.

    Take the (1-target) quantile of the galaxy scores from above: at most
    `target` of the galaxies sit at/above it. Returns (t, actual_leak).
    """
    if len(p_gal) < MIN_GAL:
        return np.nan, np.nan
    t = float(np.quantile(p_gal, 1.0 - target, method="higher"))
    return t, float(np.mean(p_gal >= t))


def threshold_for_purity(p_star, p_gal, target):
    """Lowest calibrated-prob threshold reaching `target` star purity in a bin.

    Sort the pooled (star+galaxy) scores descending; purity at depth k is the
    running star fraction. Returns (t, completeness_at_t) for the deepest point
    still at/above target, or (nan, nan) if unreachable.
    """
    if len(p_star) < MIN_STAR or len(p_gal) < MIN_GAL:
        return np.nan, np.nan
    p = np.concatenate([p_star, p_gal])
    is_star = np.concatenate([np.ones(len(p_star)), np.zeros(len(p_gal))]).astype(int)
    order = np.argsort(-p)
    s = is_star[order]
    tp = np.cumsum(s)
    purity = tp / np.arange(1, len(s) + 1)
    ok = np.where(purity >= target)[0]
    if len(ok) == 0:
        return np.nan, np.nan
    i = ok[-1]
    return float(p[order][i]), float(tp[i] / s.sum())


def catalog_base_rate():
    """Full-TEST star fraction per r-bin, from the existing eval CSV (proxy pi)."""
    path = REPORTS / "purity_completeness.csv"
    if not path.exists():
        return {}
    pc = pd.read_csv(path)
    pc = pc[pc["subset"] == "full"]
    return {round(0.5 * (lo + hi), 2): f
            for lo, hi, f in zip(pc["rmag_lo"], pc["rmag_hi"], pc["f_star"])}


def plot_table(tab, path):
    """Threshold (left) and the star completeness it costs (right) vs r-mag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = 0.5 * (tab["r_lo"] + tab["r_hi"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for tgt, tag in zip(LEAK_TARGETS, ("0p5", "1", "2")):
        ax[0].plot(r, tab[f"t_leak{tag}"], "o-", label=f"leak<={tgt*100:g}%")
        ax[1].plot(r, tab[f"compl_leak{tag}"], "o-", label=f"leak<={tgt*100:g}%")
    ax[0].plot(r, tab["t_pur99"], "s--", color="k", label="purity>=99% (sample pi)")
    ax[0].set(title="calibrated P_star threshold vs r", xlabel="r",
              ylabel="threshold", ylim=(0, 1.02))
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set(title="star completeness kept at that threshold\n(HSC stars, ~few% leaky)",
              xlabel="r", ylabel="completeness (TPR)", ylim=(0, 1.02))
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"plot  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/baseline_lshsc_xgb.json")
    args = ap.parse_args()

    bst, feats, best_it, cal = load_model(args.model)
    if cal is None:
        raise SystemExit("threshold table needs the calibrated probability; "
                         "run evaluate_baseline.py first to make the calibrator")

    cols = sorted(set(feats) | {"SPLIT", "HST_STAR", "HST_GALAXY",
                                "IN_MC_CORE", "RMAG0"})
    df = pd.read_parquet(DATASET, columns=cols, filters=[
        [("SPLIT", "==", "test"), ("HST_GALAXY", "==", True)],
        [("SPLIT", "==", "test"), ("HST_STAR", "==", True)],
    ])
    df = df[~df["IN_MC_CORE"] & (df["HST_GALAXY"] | df["HST_STAR"])]
    y = np.where(df["HST_GALAXY"], 0, 1)  # 1 = star
    print(f"model {args.model}  (calibrated)")
    print(f"eval rows (test, non-core, HSC truth): {len(df):,} "
          f"({int((y == 1).sum()):,} star / {int((y == 0).sum()):,} galaxy)\n")

    d = xgb.DMatrix(df[feats].to_numpy(np.float32), feature_names=feats)
    p = np.interp(bst.predict(d, iteration_range=(0, best_it + 1)), cal[0], cal[1])
    rmag = df["RMAG0"].to_numpy()
    base = catalog_base_rate()

    rows = []
    for lo, hi in zip(RMAG_EDGES[:-1], RMAG_EDGES[1:]):
        m = (rmag >= lo) & (rmag < hi)
        if m.sum() < MIN_GAL + MIN_STAR:
            continue
        ym, pm = y[m], p[m]
        p_star, p_gal = pm[ym == 1], pm[ym == 0]
        r_mid = round((lo + hi) / 2, 2)
        rec = {"r_lo": lo, "r_hi": hi, "n_gal": int((ym == 0).sum()),
               "n_star": int((ym == 1).sum()),
               "f_star_sample": round(float((ym == 1).mean()), 4),
               "pi_catalog": base.get(r_mid, np.nan),
               "auc": (roc_auc_score(ym, pm)
                       if len(np.unique(ym)) == 2 else np.nan)}
        for tgt in LEAK_TARGETS:
            tag = f"{tgt * 100:g}".replace(".", "p")
            t, leak = threshold_for_leak(p_gal, tgt)
            rec[f"t_leak{tag}"] = t
            rec[f"compl_leak{tag}"] = (float(np.mean(p_star >= t))
                                       if np.isfinite(t) else np.nan)
        for tgt in PURITY_TARGETS:
            t, comp = threshold_for_purity(p_star, p_gal, tgt)
            rec[f"t_pur{int(tgt * 100)}"] = t
            rec[f"compl_pur{int(tgt * 100)}"] = comp
        rows.append(rec)

    tab = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "threshold_table.csv"
    tab.to_csv(out, index=False)
    plot_table(tab, REPORTS / "threshold_table.png")

    # Sibling artifact next to the weights: travels with the model (same stem as
    # .meta/.calibrator) so infer.StarGalaxyClassifier.load() can pick it up and
    # it can't drift from the calibrated outputs it was derived from.
    sib = Path(args.model.replace(".json", ".thresholds.csv"))
    tab.to_csv(sib, index=False)

    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("=== per-magnitude threshold table (calibrated P_star) ===")
    print(tab.round(3).to_string(index=False))
    print(f"\ntable -> {out}\n      -> {sib}  (bundled with weights)")
    print("\nleak* = galaxy->star FPR cap (trustworthy); compl = star TPR kept.")
    print("pur*  = star purity at THIS sample's base rate (star-heavy); for a")
    print("catalog bin with star fraction pi use")
    print("    purity = pi*TPR / (pi*TPR + (1-pi)*FPR),  TPR=compl_leak, FPR=leak.")


if __name__ == "__main__":
    main()
