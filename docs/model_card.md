# Model card — morphocloud Tier 1 star/galaxy baseline

**Model:** `models/baseline_xgb.json` (+ `baseline_xgb.meta.json`, `baseline_xgb.calibrator.json`)
**Version:** Tier 1 baseline, trained 2026-06-11
**Type:** Gradient-boosted decision trees (XGBoost, `binary:logistic`) with isotonic
probability calibration
**Output:** `P_STAR` ∈ [0, 1] — calibrated probability that a DELVE-MC source is a star

---

## 1. Intended use

Produce a calibrated stellar probability for DELVE-MC (y4t2) catalog sources from
**catalog photometry and morphology alone**, as a fast, releasable baseline for
star/galaxy separation across the DELVE-MC footprint.

- **In scope:** point-source vs extended classification of DELVE-MC sources that pass
  the quality cut (≥2 of g/r/i detected with err < 0.5), in the magnitude/colour/seeing
  range where the model is validated (§6).
- **Out of scope:** physical QSO/AGN identification (QSOs are point sources here and are
  *not* separated — deferred to a future variable-source classifier); sources fainter
  than the validated regime; surveys other than DELVE-MC y4t2 without re-validation.

Inference reads **only DELVE-MC columns** (object catalog + per-brick exposure
metadata). No truth catalog — including DELVE DR3 — is ever an input.

## 2. Inputs / features (25)

Derived in `morphocloud.features.brick_features`:

- **Photometry:** extinction-corrected `g,r,i,z` mags (`*MAG0`, using the catalog `EBV`
  and DES Fitzpatrick-99 coefficients), per-band `ERR`, `SCATTER`; colours `g−r, r−i,
  i−z, g−i`.
- **Morphology:** `CHI`, `SHARP`, pipeline `PROB`, `ELLIPTICITY`, `ASEMI`, `BSEMI`;
  `FWHM_RATIO` (coadd FWHM ÷ per-brick seeing); `SEEING`; `CONC` (MAG_AUTO − PSF
  concentration, per-brick anchored on bright point sources).

**Deliberately excluded as inputs:** `NDET{g,r,i,z}` (an observation count = survey
cadence/depth/footprint geometry, not a source property — used only for the quality
cut) and all truth-provenance flags (would leak catalog membership). Missing values are
kept as NaN and routed natively by XGBoost.

## 3. Training data & label provenance

93.3 M quality-cut, labelled DELVE-MC sources (`data/train/dataset.parquet`). Labels are
a conflict-free consensus of:

| Source | Vote | Notes / bias |
|---|---|---|
| Gaia DR3 (high-conf. point sources) | star | **bright/blue-limited (~r ≲ 21)** — the only external *star* source |
| LS DR10 (`tractor`, type≠PSF) | galaxy | absent/Gaia-forced inside the MC cores |
| HSC v3 (HST) | galaxy | **only galaxy source inside the MCs**; shreds big galaxies → galaxy-only, conflict-rate QC per brick |
| Gaia DR3 extragalactic (I/356, "purer" unions) | galaxy | bright-biased, mostly redundant outside cores; QSO table is provenance-only, never a vote |
| DELVE DR3 `spread_model` (two-sided high-S/N cut) | star **and** galaxy | **distilled classifier, not truth** — keeps faint-end labels (~⅓ of each class are DR3-only) |

Sources in LS DR10 artifact-mask regions (bright-star ghosts/saturation) are excluded.
Conflicting and unlabelled sources are dropped. Per-source provenance flags are retained
in the dataset (for ablation/eval) but never used as features.

**Known label biases:** clean *faint* labels are scarce (see §6.1); external star truth
runs out at r ≈ 21; HSC galaxy purity degrades in crowded MC-bar fields (mitigated by the
per-brick conflict-rate cut); DR3 labels distil another classifier rather than provide
ground truth.

## 4. Training procedure

- Out-of-core: a streamed `QuantileDMatrix` over the 93.3 M-row parquet (the table never
  fully loads). `tree_method=hist`, `max_depth=8`, `eta=0.1`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=20`, `max_bin=256`.
- Class imbalance: `scale_pos_weight = n_gal/n_star ≈ 1.18` (positive class = star).
- 400 boosting rounds, early-stopping (patience 30) on the **spatially disjoint** val
  split. *(The final run reached the round cap without early-stopping; AUC had saturated,
  so the model was taken as-is — see §8.)*
- **Splits are by sky region** (HEALPix nside=16 superpixels), not i.i.d., to prevent
  spatial leakage. Row fractions land train 65% / val 17% / test 18%; the test split is
  ~1.2 mag brighter and bluer than train, so **metrics must be read stratified, not as
  i.i.d. accuracy**.

## 5. Calibration

Isotonic regression fit on the val split (raw score → observed star fraction),
serialized as interpolation knots in `baseline_xgb.calibrator.json`. Runtime application
is `np.interp(raw, x, y)` — no sklearn dependency. Calibration corrects the
`scale_pos_weight` tilt: test Brier 0.0091 (full) / 0.0026 (external-truth); the
reliability curve tracks the diagonal (`reports/reliability.png`).

## 6. Evaluation (held-out, spatially-disjoint test split, 16.7 M sources)

Because the test label is partly distilled from DR3 (which the model trained on), every
metric is reported on both the **full** test set and the **external-truth** subset (rows
with a Gaia/LS/HSC vote, i.e. no DR3-only labels).

| Metric | Full | External-truth |
|---|---|---|
| Model ROC-AUC | 0.9977 | 0.9997 |
| Model PR-AUC | 0.9984 | 0.9998 |
| Pipeline `PROB` ROC-AUC | 0.900 | 0.893 |
| Classic `−\|SHARP\|` ROC-AUC | 0.943 | 0.935 |
| Calibrated Brier | 0.0091 | 0.0026 |

The model decisively beats the pipeline `PROB` and the classic `SHARP` cut. `CHI` alone
is non-discriminative (expected — it is a goodness-of-fit statistic). At a calibrated
threshold of **0.75 you get 99% star purity at 99.0% completeness** (full set).
Performance is robust across seeing (star purity > 0.987 even at 1.1–1.3″) and colour
(dipping only to ~0.89 at red `g−r > 1.5`, where galaxies dominate).
Figures: `reports/{reliability,purity_completeness_rmag,roc,feature_importance}.png`;
full tables: `reports/eval_summary.json`, `reports/purity_completeness.csv`.

### 6.1 Validated regime and the faint-end caveat

Star purity vs r-mag (calibrated p ≥ 0.5, full set): ~0.99 at r < 19, ~0.97 at r ≈ 20.5,
**~0.93–0.95 down to r ≈ 24**, with completeness ≥ 0.91 throughout. **However, the faint
end (r ≳ 21) is only validated against DR3**, because external star truth (Gaia) runs out
there — past r ≈ 21 the external-truth sample is essentially pure galaxies, so its "star
purity" is undefined-in-practice, *not* a model failure. **Treat r ≲ 21 as the
externally-validated regime; 21 ≲ r ≲ 24 as DR3-validated only.**

### 6.2 On the DR3 `spread_model` reference baseline

A direct DR3 comparison on this dataset reads purity = completeness = 1.000, but that is
an **artifact**: conflict-dropping during label assembly forces DR3 and the external
catalogs to agree on every surviving row. It is *not* an informative baseline and is
reported here only to flag the circularity; a real DR3 comparison requires the
dropped-conflict rows.

## 7. How to run inference

```python
from morphocloud.infer import StarGalaxyClassifier
clf = StarGalaxyClassifier.load()              # loads model + calibrator
df = clf.classify_brick("0002m587")            # BRICKNAME, OBJID, RA, DEC, P_STAR, ...
```

CLI: `python scripts/predict_brick.py BRICK [BRICK ...] --out-dir preds/`
(`--format fits|parquet`). Output carries `P_STAR` (calibrated), `P_STAR_RAW`,
`BRICKUNIQ`, and `QUALITY_PASS` (the ≥2-good-band training cut — rows that fail it still
get a probability but lie outside the validated regime).

## 8. Limitations & open items

- **Faint-star truth gap (the core limitation):** no clean external star truth past
  r ≈ 21; faint-end performance currently rests on the distilled DR3 labels (§6.1).
- **DR3 distillation not yet ablated:** a DR3-free model + comparison (to quantify how
  much the faint end depends on distilled labels) is planned but **not yet run**.
- **MC cores under-masked:** the LS artifact mask is sparse exactly where DELVE-MC is
  densest (LS is Gaia-forced-only in the cores); a geometric bright-star mask is deferred.
- **No artifact/OOD class:** spurious detections are excluded from training, not modelled;
  use the quality flag (and a future OOD score) for inference rejection.
- **Training reached the round cap without early-stopping;** AUC had saturated, so the
  gain from training longer is expected to be marginal.

## 9. Files

| File | Contents |
|---|---|
| `models/baseline_xgb.json` | XGBoost model |
| `models/baseline_xgb.meta.json` | feature list, params, training counts, best iteration |
| `models/baseline_xgb.calibrator.json` | isotonic calibration knots |
| `src/morphocloud/infer.py`, `scripts/predict_brick.py` | inference library + CLI |
| `scripts/train_baseline.py`, `scripts/evaluate_baseline.py` | training, calibration + evaluation |
| `reports/` | evaluation summary, curves, figures |
