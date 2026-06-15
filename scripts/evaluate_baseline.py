"""Calibrate and evaluate the XGBoost star/galaxy model.

Two steps in one streamed pass over the held-out splits:

  CALIBRATION. The trainer used scale_pos_weight, so the raw model
    scores are tilted away from posterior probabilities. We fit an isotonic
    regression on the spatially-disjoint VAL split (raw score -> observed star
    fraction) and serialize it as interpolation knots, so the released
    inference path is a dependency-light np.interp with no sklearn at runtime.

  EVALUATION on the untouched, spatially-disjoint TEST split:
    - purity (precision) & completeness (recall) for the star class vs r-mag,
      colour and seeing, at the calibrated p=0.5 operating point;
    - ROC-AUC / PR-AUC / Brier, raw vs calibrated;
    - reference baselines on the SAME rows: pipeline PROB, the classic |SHARP|
      and CHI morphology scores, and the DELVE DR3 spread_model call (on its
      cross-match overlap);
    - feature importance by gain.

  Because the assembled TEST label is partly distilled from DR3 (which the
  model trained on), every metric is reported twice: on the FULL test set and
  on the EXTERNAL-TRUTH subset (rows carrying a Gaia / LS / HSC vote, i.e. no
  DR3-only labels) - the honest, non-circular number.

Outputs: models/baseline_xgb.calibrator.json, reports/eval_summary.json,
reports/purity_completeness.csv and reports/*.png. Usage:

    python scripts/evaluate_baseline.py [--batch-size N] [--model PATH]
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from morphocloud.config import DATA_DIR, MODELS_DIR

DATASET_PATH = DATA_DIR / "train" / "dataset.parquet"
REPORTS_DIR = MODELS_DIR.parent / "reports"

# truth flags that come from a real external catalog (NOT DR3, which the model
# distilled). A test row with any of these set is "external truth": its label
# does not rest on the distilled DR3 classifier, so metrics on it are honest.
EXTERNAL_TRUTH_FLAGS = ("GAIA_STAR", "GAIA_GALAXY", "LS_GALAXY", "HST_GALAXY")
# extra columns pulled per row for stratification and reference baselines
EXTRA_COLS = (
    "RMAG0", "G_R", "SEEING", "SHARP", "CHI", "PROB",
    "DR3_STAR", "DR3_GALAXY",
) + EXTERNAL_TRUTH_FLAGS

# stratification bin edges
RMAG_EDGES = np.arange(16.0, 24.5, 0.5)
GR_EDGES = np.arange(-0.5, 2.25, 0.25)
SEEING_EDGES = np.array([0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.3])


def stream_split(booster, best_iter, path, split, features, extra, batch_size):
    """Stream one SPLIT, returning raw star scores, labels and `extra` columns.

    Predicts per row-group batch with a plain DMatrix (exact: quantization only
    affected training split-finding, not the tree thresholds), so the test
    table never lands in memory whole.
    """
    pf = pq.ParquetFile(path)
    columns = list(features) + ["LABEL", "SPLIT"] + list(extra)
    out = {k: [] for k in ("raw", "label", *extra)}
    n = 0
    t0 = time.time()
    for rb in pf.iter_batches(columns=columns, batch_size=batch_size):
        split_arr = rb.column("SPLIT").to_numpy(zero_copy_only=False)
        mask = split_arr == split
        if not mask.any():
            continue
        X = np.column_stack(
            [rb.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
             for c in features]
        )[mask]
        d = xgb.DMatrix(X, feature_names=list(features))
        out["raw"].append(booster.predict(d, iteration_range=(0, best_iter + 1)))
        out["label"].append(
            rb.column("LABEL").to_numpy(zero_copy_only=False)[mask].astype(np.int8)
        )
        for c in extra:
            out[c].append(
                rb.column(c).to_numpy(zero_copy_only=False)[mask].astype(np.float32)
            )
        n += int(mask.sum())
        print(f"  [{split}] {n/1e6:5.1f}M rows [{time.time()-t0:4.0f}s]", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def fit_calibrator(raw, label):
    """Isotonic raw-score -> star-probability map, returned as interp knots."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, label)
    return iso, np.asarray(iso.X_thresholds_), np.asarray(iso.y_thresholds_)


def apply_calibrator(raw, x_knots, y_knots):
    """Reproduce isotonic prediction with clipping via np.interp (no sklearn)."""
    return np.interp(raw, x_knots, y_knots)


def score_metrics(label, scores):
    """ROC-AUC and PR-AUC (positive class = star) for each named score."""
    return {
        name: {
            "roc_auc": float(roc_auc_score(label, s)),
            "pr_auc": float(average_precision_score(label, s)),
        }
        for name, s in scores.items()
    }


def confusion_at(label, pred_star):
    """Star/galaxy purity, completeness and accuracy at a fixed threshold."""
    star = label == 1
    tp = int((pred_star & star).sum())
    fp = int((pred_star & ~star).sum())
    fn = int((~pred_star & star).sum())
    tn = int((~pred_star & ~star).sum())
    saf = lambda a, b: float(a / b) if b else float("nan")
    return {
        "star_purity": saf(tp, tp + fp),
        "star_completeness": saf(tp, tp + fn),
        "galaxy_purity": saf(tn, tn + fn),
        "galaxy_completeness": saf(tn, tn + fp),
        "accuracy": saf(tp + tn, tp + fp + fn + tn),
        "n_star": tp + fn,
        "n_galaxy": tn + fp,
    }


def purity_completeness_vs(label, pred_star, binvals, edges):
    """Per-bin star purity & completeness over `edges` of `binvals`."""
    star = label == 1
    rows = []
    idx = np.digitize(binvals, edges)
    for b in range(1, len(edges)):
        m = idx == b
        if not m.any():
            continue
        c = confusion_at(label[m], pred_star[m])
        rows.append({
            "lo": float(edges[b - 1]), "hi": float(edges[b]),
            "n": int(m.sum()), "f_star": float(star[m].mean()),
            "star_purity": c["star_purity"],
            "star_completeness": c["star_completeness"],
        })
    return rows


def threshold_for_purity(label, prob, target=0.99):
    """Lowest calibrated-prob threshold reaching `target` star purity."""
    order = np.argsort(-prob)
    star = (label[order] == 1).astype(np.int64)
    tp = np.cumsum(star)
    purity = tp / np.arange(1, len(star) + 1)
    ok = np.where(purity >= target)[0]
    if len(ok) == 0:
        return float("nan"), float("nan")
    i = ok[-1]  # deepest point still at/above target purity
    return float(prob[order][i]), float(tp[i] / star.sum())


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_reliability(label, raw, cal, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0, 1, 21)
    mid = 0.5 * (edges[:-1] + edges[1:])
    for s, name, c in ((raw, "raw", "C1"), (cal, "calibrated", "C0")):
        idx = np.clip(np.digitize(s, edges) - 1, 0, 19)
        obs = np.array([label[idx == b].mean() if (idx == b).any() else np.nan
                        for b in range(20)])
        ax.plot(mid, obs, "o-", color=c, label=name, ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.set(xlabel="predicted star probability", ylabel="observed star fraction",
           title="Reliability (test split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_purity_completeness(rows_full, rows_ext, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for rows, ls, tag in ((rows_full, "-", "full"), (rows_ext, "--", "ext-truth")):
        x = [0.5 * (r["lo"] + r["hi"]) for r in rows]
        ax.plot(x, [r["star_purity"] for r in rows], "o" + ls, color="C0",
                label=f"purity ({tag})", ms=4)
        ax.plot(x, [r["star_completeness"] for r in rows], "s" + ls, color="C3",
                label=f"completeness ({tag})", ms=4)
    ax.set(xlabel="r-mag (extinction-corrected)", ylabel="star metric",
           title="Star purity & completeness vs r-mag (calibrated p>=0.5)",
           ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_roc(label, scores, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(label, s)
        # thin dense curves for a light PNG
        k = max(1, len(fpr) // 2000)
        ax.plot(fpr[::k], tpr[::k], lw=1.3,
                label=f"{name} (AUC {roc_auc_score(label, s):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="false positive rate", ylabel="true positive rate (star)",
           title="ROC vs reference baselines (ext-truth test)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_importance(booster, path):
    gain = booster.get_score(importance_type="gain")
    items = sorted(gain.items(), key=lambda kv: kv[1])[-15:]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh([k for k, _ in items], [v for _, v in items], color="C0")
    ax.set(xlabel="gain", title="Feature importance (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=2_000_000)
    ap.add_argument("--model", default=str(MODELS_DIR / "baseline_xgb.json"))
    args = ap.parse_args()

    with open(args.model.replace(".json", ".meta.json")) as fh:
        meta = json.load(fh)
    features = meta["features"]
    best_iter = int(meta["best_iteration"])
    booster = xgb.Booster()
    booster.load_model(args.model)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"model {args.model}  best_iter={best_iter}  {len(features)} features\n")

    # ---- fit isotonic calibrator on the VAL split ----------------------------
    print("streaming VAL split for calibration...")
    val = stream_split(booster, best_iter, DATASET_PATH, "val", features,
                       (), args.batch_size)
    iso, x_knots, y_knots = fit_calibrator(val["raw"], val["label"])
    cal_path = args.model.replace(".json", ".calibrator.json")
    with open(cal_path, "w") as fh:
        json.dump({
            "method": "isotonic",
            "fit_split": "val",
            "x_thresholds": x_knots.tolist(),
            "y_thresholds": y_knots.tolist(),
            "note": "calibrated star probability = np.interp(raw, x, y)",
        }, fh)
    val_brier_raw = float(brier_score_loss(val["label"], val["raw"]))
    val_brier_cal = float(brier_score_loss(val["label"], iso.predict(val["raw"])))
    print(f"  calibrator -> {cal_path}")
    print(f"  val Brier  raw {val_brier_raw:.5f} -> cal {val_brier_cal:.5f}\n")

    # ---- evaluate on the held-out TEST split ---------------------------------
    print("streaming TEST split for evaluation...")
    test = stream_split(booster, best_iter, DATASET_PATH, "test", features,
                        EXTRA_COLS, args.batch_size)
    label = test["label"]
    cal = apply_calibrator(test["raw"], x_knots, y_knots)
    pred_star = cal >= 0.5

    external = np.zeros(len(label), dtype=bool)
    for f in EXTERNAL_TRUTH_FLAGS:
        external |= test[f] > 0.5
    dr3_overlap = (test["DR3_STAR"] > 0.5) | (test["DR3_GALAXY"] > 0.5)
    print(f"  test rows {len(label):,}  external-truth {external.sum():,}  "
          f"DR3-overlap {dr3_overlap.sum():,}\n")

    summary = {
        "model": args.model,
        "best_iteration": best_iter,
        "n_test": int(len(label)),
        "n_external_truth": int(external.sum()),
        "val_brier": {"raw": val_brier_raw, "calibrated": val_brier_cal},
        "subsets": {},
    }
    pc_rows = {}
    for tag, m in (("full", np.ones(len(label), bool)), ("external", external)):
        lab = label[m]
        scores = {
            "model": cal[m],
            "pipeline_PROB": test["PROB"][m],
            "neg_abs_SHARP": -np.abs(test["SHARP"][m]),
            "neg_CHI": -test["CHI"][m],
        }
        thr99, comp99 = threshold_for_purity(lab, cal[m], 0.99)
        subset = {
            "n": int(m.sum()),
            "f_star": float((lab == 1).mean()),
            "auc": score_metrics(lab, scores),
            "test_brier": {
                "raw": float(brier_score_loss(lab, test["raw"][m])),
                "calibrated": float(brier_score_loss(lab, cal[m])),
            },
            "confusion_p0.5": confusion_at(lab, pred_star[m]),
            "star_purity99": {"threshold": thr99, "completeness": comp99},
        }
        # DR3 spread_model baseline on its cross-match overlap within this subset
        dm = m & dr3_overlap
        if dm.any():
            subset["dr3_spread_model"] = {
                **confusion_at(label[dm], test["DR3_STAR"][dm] > 0.5),
                "vs": "test LABEL on DR3 overlap",
            }
        summary["subsets"][tag] = subset
        pc_rows[tag] = purity_completeness_vs(
            lab, pred_star[m], test["RMAG0"][m], RMAG_EDGES)

    # color / seeing stratification on the external-truth subset
    summary["external_vs_color"] = purity_completeness_vs(
        label[external], pred_star[external], test["G_R"][external], GR_EDGES)
    summary["external_vs_seeing"] = purity_completeness_vs(
        label[external], pred_star[external], test["SEEING"][external], SEEING_EDGES)

    with open(REPORTS_DIR / "eval_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # flat CSV of the r-mag purity/completeness curves
    csv_lines = ["subset,rmag_lo,rmag_hi,n,f_star,star_purity,star_completeness"]
    for tag, rows in pc_rows.items():
        for r in rows:
            csv_lines.append(
                f"{tag},{r['lo']},{r['hi']},{r['n']},{r['f_star']:.4f},"
                f"{r['star_purity']:.4f},{r['star_completeness']:.4f}")
    (REPORTS_DIR / "purity_completeness.csv").write_text("\n".join(csv_lines) + "\n")

    # figures
    fig_reliability(label, test["raw"], cal, REPORTS_DIR / "reliability.png")
    fig_purity_completeness(pc_rows["full"], pc_rows["external"],
                            REPORTS_DIR / "purity_completeness_rmag.png")
    ext_scores = {
        "model": cal[external],
        "pipeline_PROB": test["PROB"][external],
        "neg_abs_SHARP": -np.abs(test["SHARP"][external]),
        "neg_CHI": -test["CHI"][external],
    }
    fig_roc(label[external], ext_scores, REPORTS_DIR / "roc.png")
    fig_importance(booster, REPORTS_DIR / "feature_importance.png")

    # console digest
    ext = summary["subsets"]["external"]
    full = summary["subsets"]["full"]
    print("=" * 64)
    print(f"{'metric':22s}{'full':>14s}{'ext-truth':>14s}")
    print(f"{'model ROC-AUC':22s}{full['auc']['model']['roc_auc']:>14.5f}"
          f"{ext['auc']['model']['roc_auc']:>14.5f}")
    print(f"{'model PR-AUC':22s}{full['auc']['model']['pr_auc']:>14.5f}"
          f"{ext['auc']['model']['pr_auc']:>14.5f}")
    print(f"{'PROB ROC-AUC':22s}{full['auc']['pipeline_PROB']['roc_auc']:>14.5f}"
          f"{ext['auc']['pipeline_PROB']['roc_auc']:>14.5f}")
    print(f"{'-|SHARP| ROC-AUC':22s}{full['auc']['neg_abs_SHARP']['roc_auc']:>14.5f}"
          f"{ext['auc']['neg_abs_SHARP']['roc_auc']:>14.5f}")
    print(f"{'-CHI ROC-AUC':22s}{full['auc']['neg_CHI']['roc_auc']:>14.5f}"
          f"{ext['auc']['neg_CHI']['roc_auc']:>14.5f}")
    print(f"{'test Brier (cal)':22s}{full['test_brier']['calibrated']:>14.5f}"
          f"{ext['test_brier']['calibrated']:>14.5f}")
    print("-" * 64)
    c = ext["confusion_p0.5"]
    print(f"ext-truth @ p>=0.5: star purity {c['star_purity']:.4f}  "
          f"completeness {c['star_completeness']:.4f}")
    if "dr3_spread_model" in ext:
        d = ext["dr3_spread_model"]
        print(f"DR3 spread_model : star purity {d['star_purity']:.4f}  "
              f"completeness {d['star_completeness']:.4f}  (overlap baseline)")
    print(f"\nsummary -> {REPORTS_DIR / 'eval_summary.json'}")
    print(f"figures -> {REPORTS_DIR}/*.png")


if __name__ == "__main__":
    main()
