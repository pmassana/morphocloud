# morphocloud

A reliable **star/galaxy classifier for the DELVE-MC survey**. morphocloud produces a
calibrated *stellar probability* for each source from catalog photometry and morphology
alone — no images, no external catalog at inference time — so it runs anywhere the
DELVE-MC (y4t2) brick catalogs are present.

The current deliverable is **Tier 1**: a gradient-boosted decision tree (XGBoost) baseline
with calibrated probabilities. It is the strong, fast, releasable starting point, ahead of
any image-based or foundation-model work (Tier 2/3).

## What it does

Given a DELVE-MC source, the model returns `P_STAR` ∈ [0, 1], a calibrated probability that
the source is a star (vs. an extended galaxy). It learns from 25 catalog features —
extinction-corrected `griz` photometry and colours, plus DAOPHOT/ALLFRAME morphology
(`CHI`, `SHARP`, pipeline `PROB`, shape parameters, seeing-normalized FWHM, and a
MAG_AUTO−PSF concentration proxy). Truth-catalog columns (including DELVE DR3) are **never**
inputs, so inference depends only on the DELVE-MC catalog itself.

**Performance** (held-out, spatially-disjoint test split): ROC-AUC **0.998**, calibrated
Brier 0.009; **99% star purity at 99% completeness** at a calibrated threshold of 0.75. It
decisively beats the classic `SHARP` cut (AUC 0.94) and the pipeline `PROB` (0.90). The
classifier is externally validated to **r ≈ 21** and DR3-validated to **r ≈ 24** — see
[`docs/model_card.md`](docs/model_card.md) for the full evaluation, biases, and the
faint-end caveat.

## Installation

Requires Python ≥ 3.11. From the repo root:

```bash
conda create -n morphocloud python=3.11
conda activate morphocloud
pip install -e .          # add ".[dev]" for pytest + ruff
```

> On macOS, XGBoost needs OpenMP: `conda install llvm-openmp`.

Two things inference needs that are **not** in the repo:
- **The DELVE-MC y4t2 brick catalogs** (object FITS files). Their location is set in
  [`src/morphocloud/config.py`](src/morphocloud/config.py) (`DELVEMC_DATA`).
- **The trained model artifacts** `models/baseline_xgb.json`,
  `baseline_xgb.meta.json`, `baseline_xgb.calibrator.json` (gitignored — obtain the
  released weights, or reproduce them with the pipeline below).

## Usage — classify a brick

Python:

```python
from morphocloud.infer import StarGalaxyClassifier

clf = StarGalaxyClassifier.load()        # loads model + isotonic calibrator
df = clf.classify_brick("0002m587")      # one row per source
```

Command line:

```bash
python scripts/predict_brick.py 0002m587 --out-dir preds/          # FITS
python scripts/predict_brick.py 0002m587 0003m587 --format parquet
```

Output columns: `BRICKNAME`, `OBJID`, `RA`, `DEC`, `BRICKUNIQ`, `P_STAR` (calibrated),
`P_STAR_RAW`, and `QUALITY_PASS`. `QUALITY_PASS` flags the ≥2-good-band cut the model was
trained under; sources that fail it still get a probability but lie outside the validated
regime.

## Reproducing the pipeline

The full Tier 1 pipeline, end to end (details and locked decisions in
[`docs/plan.md`](docs/plan.md)):

```bash
# 1. fetch truth labels by sky tile (Gaia DR3, LS DR10, DELVE DR3, HSC v3, Gaia extragal)
python scripts/fetch_labels.py gaia_dr3        # ... and the other sources

# 2-3. assemble the labelled, quality-cut, spatially-split dataset
python scripts/build_dataset.py --jobs 10

# 4. train the out-of-core XGBoost baseline
python scripts/train_baseline.py --threads 10 | tee logs/train_baseline.log

# 5-6. calibrate (isotonic) and evaluate on the held-out test split
python scripts/evaluate_baseline.py            # writes reports/
```

Labels are streamed/fetched on demand by sky chunk — the DELVE-MC catalog is never copied
(local storage budget ~100 GB). The dataset (~93 M rows) trains out-of-core, so the full
table never lives in RAM.

## Repository layout

| Path | Contents |
|---|---|
| `src/morphocloud/` | package: brick readers, features, label assembly, dataset, TAP clients, inference |
| `scripts/` | CLIs: `fetch_labels`, `build_dataset`, `train_baseline`, `evaluate_baseline`, `predict_brick` |
| `docs/plan.md` | Tier 1 plan, data conventions, and locked decisions |
| `docs/model_card.md` | model card: intended use, training data, biases, evaluation, limitations |
| `tests/` | unit tests (`pytest`) |
| `models/`, `reports/`, `data/` | model artifacts, evaluation outputs, datasets (all gitignored) |

## Status & roadmap

Tier 1 (catalog-based GBDT baseline) is complete: dataset → training → calibration →
evaluation → packaging. Known open item: a **DR3-distillation ablation** to quantify how
much the faint end relies on the distilled DR3 labels. Tier 2/3 (image-based and
transformer/foundation-model classifiers) are future work.

## License

[MIT](LICENSE).
