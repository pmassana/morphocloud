# morphocloud

A reliable **star/galaxy classifier for the DELVE-MC survey**. morphocloud produces a
calibrated *stellar probability* for each source from catalog photometry and morphology
alone — no images, no external catalog at inference time — so it runs anywhere the
DELVE-MC DR1 brick catalogs are present.

It is a gradient-boosted decision tree (XGBoost) classifier with calibrated probabilities:
strong, fast, and runnable directly on catalog data.

## What it does

Given a DELVE-MC source, the model returns `P_STAR` ∈ [0, 1], a calibrated probability that
the source is a star (vs. an extended galaxy). Truth-catalog columns are **never** inputs,
so inference depends only on the DELVE-MC catalog itself.

It separates stars from galaxies essentially perfectly at the bright end (**star purity
0.998 at completeness 0.999 for r < 20.5** on independent HST/Gaia/Legacy-Surveys truth) and
stays useful deep into the faint regime (reliable separation to **r ≈ 23**), decisively
beating the classic `SHARP` cut and the pipeline `PROB`.

## How it works, evaluation & caveats

The full feature set, training data and procedure; the evaluation (purity/completeness vs
magnitude, ROC, feature importance, and the independent HST faint-end test); the
magnitude-dependent operating thresholds; and the validity ranges and known biases are all
documented in the **[model card](docs/model_card.md)**. Read it before using `P_STAR` for a
high-purity sample — past r ≈ 20.5 a magnitude-dependent threshold is required (the usage
example below shows how).

## Installation

Requires Python ≥ 3.11. The base install pulls **only the inference dependencies**
(`numpy`, `pandas`, `astropy`, `xgboost`):

```bash
conda create -n morphocloud python=3.11
conda activate morphocloud
pip install morphocloud            # or `pip install -e .` from a clone
```

The fetch and train pipelines are opt-in extras:

```bash
pip install "morphocloud[fetch]"   # TAP truth-label fetching (pyvo, scipy, healpy)
pip install "morphocloud[train]"   # dataset assembly + training (scikit-learn, matplotlib, …)
pip install "morphocloud[dev]"     # pytest + ruff
pip install "morphocloud[all]"     # everything
```

> On macOS, XGBoost needs OpenMP: `conda install llvm-openmp`.

**Get the model weights.** Download the four `baseline_lshsc_xgb.*` files
(`.json`, `.meta.json`, `.calibrator.json`, `.thresholds.csv`) from the
[v1.0.0 release](https://github.com/pmassana/morphocloud/releases/tag/v1.0.0) into a
directory, then either set `MORPHOCLOUD_MODELS_DIR` to it or pass the path to
`StarGalaxyClassifier.load(model_path=…)`.

**Local brick workflows only** (`classify_brick`, fetching, training) read the DELVE-MC
catalogs from disk. Point morphocloud at them via environment variables — nothing
machine-specific ships in the package:

```bash
export MORPHOCLOUD_DELVEMC_DATA=/path/to/delvemc_dr1
export MORPHOCLOUD_BRICK_LIST=/path/to/delvemc_bricks_0.25deg.fits.gz
```

`MORPHOCLOUD_DELVEMC_DATA` is a directory holding, for each brick `<BRICKNAME>`, the
object catalog `<BRICKNAME>_object.fits.gz` and the per-exposure metadata
`<BRICKNAME>_meta.fits` (e.g. `0002m587_object.fits.gz`, `0002m587_meta.fits`).
In-memory inference (below) needs none of these.

## Usage — inference on any catalog

Inference runs on **any** table that carries the model features — a pandas `DataFrame`,
an astropy `Table`, or a numpy structured array — so no on-disk brick layout is required.
Build the feature columns from your raw DELVE-MC columns with `engineer_features`
(it's path-free: catalog + the brick's median seeing in arcsec), then score:

```python
from morphocloud.features import engineer_features, RAW_INPUT_COLUMNS, FEATURE_COLUMNS
from morphocloud.infer import StarGalaxyClassifier

clf = StarGalaxyClassifier.load(model_path="weights/baseline_lshsc_xgb.json")

# `catalog` is any DataFrame / astropy Table holding RAW_INPUT_COLUMNS
feats = engineer_features(catalog, seeing=1.1)   # seeing in arcsec (FWHM normalizer)
p_star = clf.predict(feats)                       # calibrated P(star), one per row
```

If you already have the `FEATURE_COLUMNS` engineered, skip `engineer_features` and pass
your table straight to `clf.predict(table)` — a DataFrame, astropy `Table`, and structured
array all give identical results. `RAW_INPUT_COLUMNS` / `FEATURE_COLUMNS` document the
contracts.

**Turning `P_STAR` into a star/galaxy decision (smooth threshold).** The star-heavy
training base rate makes a flat `P_STAR >= 0.5` over-call stars faint-ward, so cut on a
magnitude-dependent threshold instead. `smooth_threshold` fits a smooth curve `T(r)` (in
logit space, from the bundled per-magnitude table) and returns a callable with a single
`strictness` dial — tighten (`>0`) or loosen (`<0`) the cut everywhere:

```python
threshold_for = clf.smooth_threshold(operating_point="leak1")   # galaxy->star leak <=1%
rmag0 = feats["RMAG0"].to_numpy()                                # extinction-corrected r
star = p_star >= threshold_for(rmag0, strictness=0.0, flat=0.5)  # flat 0.5 outside table r
```

Operating points: `leak0.5` / `leak1` / `leak2` (galaxy→star leak caps, base-rate-free and
the trustworthy controls) and `pur95` / `pur99` (star-purity points at the star-heavy test
base rate). `threshold_for(rmag0, target=…)` gives the raw step-table lookup if you prefer
it to the smooth curve.

## Usage — classify a local brick

With `MORPHOCLOUD_DELVEMC_DATA` set, score a brick straight from its files:

```python
from morphocloud.infer import StarGalaxyClassifier

clf = StarGalaxyClassifier.load()        # loads model + isotonic calibrator
df = clf.classify_brick("0002m587")      # one row per source
```

```bash
python scripts/predict_brick.py 0002m587 --out-dir preds/          # FITS
python scripts/predict_brick.py 0002m587 0003m587 --format parquet
```

Output columns: `BRICKNAME`, `OBJID`, `RA`, `DEC`, `BRICKUNIQ`, `RMAG0`
(extinction-corrected r), `P_STAR` (calibrated), `P_STAR_RAW`, and `QUALITY_PASS`.

Apply `smooth_threshold` (or the `threshold_for` step lookup) to `df["RMAG0"]` and
`df["P_STAR"]` exactly as in *inference on any catalog* above to get a star/galaxy
decision. Per-bin numbers and the recover-purity-at-your-π formula are in
[`docs/model_card.md`](docs/model_card.md) §6.2.

## Reproducing the pipeline

```bash
# 1. fetch truth labels by sky tile (Gaia DR3, LS DR10 galaxies + PSF stars,
#    DELVE DR3, HSC v3, Gaia extragal)
python scripts/fetch_labels.py gaia_dr3        # ... and the other sources

# 2-3. assemble the labelled, quality-cut, spatially-split dataset
python scripts/build_dataset.py --jobs 10 && python scripts/build_dataset.py merge

# 4. train the out-of-core XGBoost model (extmem default; incore for big-RAM hosts)
python scripts/train_baseline.py --threads 10 --dmatrix extmem | tee logs/train.log

# 5-6. calibrate (isotonic) + evaluate on the held-out test split  -> reports/
python scripts/evaluate_baseline.py --model models/baseline_xgb.json
python scripts/evaluate_faint_hst.py           # faint-end vs baseline on HSC truth
```

Labels are streamed/fetched on demand by sky chunk — the DELVE-MC catalog is never copied
(local storage budget ~100 GB).

## Repository layout

| Path | Contents |
|---|---|
| `src/morphocloud/` | package: brick readers, features, label assembly, dataset, TAP clients, inference |
| `scripts/` | CLIs: `fetch_labels`, `build_dataset`, `train_baseline`, `evaluate_baseline`, `evaluate_faint_hst`, `predict_brick` |
| `docs/model_card.md` | model card: intended use, training data, biases, evaluation, limitations |
| `docs/figures/` | evaluation figures (shown in the model card) |
| `tests/` | unit tests (`pytest`) |
| `models/`, `reports/`, `data/` | model artifacts, evaluation outputs, datasets (all gitignored) |

## License

[MIT](LICENSE).
