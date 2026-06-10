# Tier 1 implementation plan — gradient-boosted star–galaxy classifier

Status: agreed 2026-06-09. Decisions locked: bulk labels from **Gaia DR3 (stars) + Legacy
Surveys DR10 (galaxies)**; **HSC v3 (HST)** promoted to a fourth label source for the
faint MC interior (2026-06-09, after the coverage census below), with HST fields split
between training and an independent validation holdout; JWST stays deferred. The pipeline `PROB` column **is** included as an input feature (with a PROB-only cut
reported as a reference baseline). **DELVE DR3** (DESDM reduction, Astro Data Lab) is a
third label provenance (conservative high-S/N `spread_model` cuts only) and the headline
external baseline — but its columns are **never model inputs**: inference must run from
DELVE-MC catalog values alone.

## What we verified about the y4t2 data (2026-06-09)

- 35,041 of 35,052 footprint bricks have `BRICKNAME_object.fits.gz` (254 GB total,
  median ~13k rows/brick, up to ~170k near the LMC center → ~500M sources overall).
- Morphology columns are DAOPHOT/PHOTRED heritage: `CHI`, `SHARP`, `PROB` (per-exposure
  star/galaxy probability averaged over exposures), plus coadd shape parameters
  `MAG_AUTO`, `ASEMI`, `BSEMI`, `THETA`, `ELLIPTICITY`, `FWHM`. There is **no**
  `spread_model` — the original roadmap was written assuming a SourceExtractor/DESDM
  reduction.
- `EBV` (SFD) is already a per-source column → no external dust map needed.
- Missing-value sentinels: `99.99` (mags, errors, CHI, SHARP, PROB), `999999.0` (FWHM);
  `NDET<band> == 0` marks no detection. `BRICKUNIQ == 1` selects the unique-area subset
  (deduplication across overlapping bricks).
- **Column sets vary by brick**: g/r/i/z generally present, u and Y only sometimes.
  Readers must normalize schemas.
- `BRICKNAME_meta.fits`: one row per exposure/chip with `FWHM`, `ALF_DEPTH`,
  `CALIB_DEPTH`, zero-point terms → per-brick seeing/depth summary features.

## Truth-catalog facts verified live (2026-06-09)

- **Astrometry**: DELVE-MC is Gaia-calibrated — median offset vs Gaia DR3 ~0.01",
  90th-percentile separation 0.15" (tested on a dense LMC brick and a sparse periphery
  brick). Cross-match radius fixed at **0.5"**.
- **Coverage holes**: LS DR10 and DELVE DR3 both have *zero* sources at the LMC and SMC
  centers. DELVE DR3 excludes the inner MC region entirely (also empty at the LMC
  outskirt test point, 3.4° from center). Near the MCs, LS DR10 rows are purely
  Gaia-forced PSF entries (`ref_cat='GE'`) — no independent detection ran there, so no
  galaxy labels exist near the Clouds from either catalog. Gaia covers everything.
  Quantified per brick in `data/labels/coverage/brick_coverage.parquet`
  (built by `scripts/build_coverage_maps.py`): **DELVE DR3 reaches 96.3% of the
  35,052 footprint bricks; LS DR10 provides >=10 galaxy labels in 81.2%; only 2.9%
  of bricks (the inner MCs) have neither** — there, Gaia (and later HSC) carry the
  labels. South of dec -44.8 the label pools are ~77M LS galaxies and ~693M DR3
  sources, so training-set size is limited by what we choose to use, not supply.
- **Data Lab tables**: `gaia_dr3.gaia_source`, `ls_dr10.tractor`,
  `delve_dr3.coadd_objects` (no precomputed extended_class in DR3; we build our own
  spread_model cuts; `flags_gold == 0` is the cleanliness cut).
- **Data Lab TAP quirks**: ADQL POINT/CIRCLE geometry is *not* translated and bare q3c
  predicates don't parse — use plain ra/dec range conditions (indexed, fast; see
  `tap.box_condition`). `LIMIT` is silently ignored — use `TOP n`. Missing floats are
  stored as NaN, *not* NULL, so `IS NOT NULL` does not filter them.
- **HSC v3 via MAST VO TAP** (`https://mast.stsci.edu/vo-tap/api/v0.1/hsc`, anonymous,
  verified 2026-06-09): summary table `dbo.summagaper2catview`, lowercase columns
  (`matchra`, `matchdec`, `ci`, `ci_sigma`, `numimages`, per-instrument magnitudes
  `a_*`/`w2_*`/`w3_*` = ACS/WFPC2/WFC3). Quirks: `SUM(CASE …)` is rejected (strict
  ADQL 2.1 parser); results are **hard-capped at 100,000 rows** (sync and async,
  `maxrec` above it silently ignored) and the overflow flag is unreliable (also set
  on complete results) — so completeness must be verified against `COUNT(*)` and big
  fetches split (`LabelSource.max_rows` recursively halves the box in dec);
  `GROUP BY FLOOR(expr)` runs server-side and is fast (full southern-cap census in
  seconds).
- **HSC coverage census** (`scripts/build_hst_coverage.py` →
  `data/labels/coverage/hsc_*.parquet`, 2026-06-09): 14.7M HSC sources fall in
  footprint bricks (10.6M with `numimages >= 2`), concentrated in 186 of 718 tiles and
  752 of 35,052 bricks (2.1% — pencil-beam, but dense exactly on the Clouds). Of the
  1,002 bricks blind to both LS DR10 and DELVE DR3 (the inner MCs), **270 (27%) have
  HSC sources — 7.7M with `numimages >= 2`**, of which ~1.6M pass a provisional
  `ci > 1.3` extended cut (a blend-inflated upper bound on the galaxy-label pool, to
  be replaced by validated per-instrument CI thresholds).

## Storage strategy (~100 GB budget, catalog excluded)

Stream everything per brick / per sky chunk; never duplicate the DELVE-MC catalog.

- `data/labels/` — truth tables fetched per HEALPix nside=32 tile (718 tiles cover the
  footprint) via anonymous Data Lab TAP, cached as parquet. Label cuts run server-side
  (`LabelSource.where`), which cuts fetch time ~4x: measured ~4 min/tile in the
  periphery (Gaia 25 s, DR3 130 s, LS 76 s), so a full-footprint fetch is roughly a
  day serial — resumable via `scripts/fetch_labels.py`, parallelizable across a few
  workers if needed. Estimated a few GB total. `hsc_v3` (MAST) adds ~8 s for the
  ~530 empty tiles and minutes for dense MC tiles (100k-row split fetches; the 800k-row
  SMC tile took ~6 min, 35 MB parquet) — est. 2–3 GB over the 186 non-empty tiles.
- `data/train/` — the final labelled feature table (single parquet, est. < 5 GB).
- `models/` — serialized XGBoost + calibrator (MB scale).
- Inference output — per-brick parquet of `OBJID`, calibrated probability
  (est. 10–20 GB for the full survey). Everything fits comfortably.

## Pipeline stages

### 1. Label assembly (`labels/`, `crossmatch.py`)

- **Stars ← Gaia DR3**: point sources with clean astrometry (`ruwe < 1.4`, low
  `astrometric_excess_noise` significance, low `ipd_frac_multi_peak`). Keeps both MC
  members (small parallax, MC-consistent PM) and Galactic foreground. Document the
  bright/blue selection bias (Gaia G ≲ 20.5–21).
- **Galaxies ← LS DR10 tractor**: `type` in (DEV, EXP, SER) with conservative quality
  cuts (Δχ² margin over PSF model, clean `maskbits`, minimum `nobs`). Tractor is
  unreliable in the crowded LMC/SMC centers, so galaxy labels will concentrate in the
  periphery — recorded as a known bias.
- **Stars & galaxies ← DELVE DR3 (DESDM, Astro Data Lab)**: covers the vast majority of
  the footprint and provides `spread_model`/`spreaderr_model`, which the community uses
  as `|spread_model| < 0.003 + spreaderr_model` to select stars. We use it as a
  *teacher label source with conservative two-sided cuts at high S/N only*:
  - stars: `|spread_model| + 3·spreaderr_model < 0.003`-style tight cut;
  - galaxies: `spread_model − 3·spreaderr_model > 0.005`-style tight cut;
  - the ambiguous band in between stays unlabelled, and labels are restricted to the
    magnitude/S-N range where the cut is demonstrably reliable (validated against the
    Gaia/LS DR10 overlap before use).
  These labels distill another *morphology classifier*, not ground truth — they inherit
  spread_model's faint-end errors, so they carry their own provenance flag, get
  ablation-tested (train with/without; evaluate on Gaia/LS/HST-labelled holdout), and can
  be down-weighted relative to Gaia/LS labels.
- **Galaxies ← HSC v3 (HST, MAST)** — implemented & validated 2026-06-09: the only
  galaxy source inside the MCs and the only label set independent of DECam imaging.
  Cut (`labels/hsc.py`): `ci > 1.6` AND `ci_sigma < 0.2·ci` AND an optical broad-band
  detection (IR-only sources excluded — WFC3/IR shifts the CI scale) AND
  `numimages >= 2` (server-side). Matching is blend-aware and many-to-one
  (`assemble._component_flags`): each HSC component is assigned to its nearest DELVE
  object within 0.5", and the object is `HST_GALAXY` only if *every* component passes
  the galaxy cut; mixed/ambiguous components → `HST_BLEND`, unlabelled but counted.
  **No star labels from HSC**: validation showed HSC shreds big galaxies
  (LS `shape_r` > 1.5") into point-like knots (ci ~ 1.1), so point-like CI ≠ star at
  faint magnitudes. Validation against Gaia stars and LS galaxies (tiles
  hp32-08298 = deep periphery field, hp32-08329 = SMC interior):
  - Unsaturated stars (G > 18.5): ci p50 ≈ 1.02, p95 ≈ 1.2 (periphery) – 1.5
    (crowded interior). Leak past ci = 1.6 is <1% / 2.2% respectively; the
    `ci_sigma` guard cuts the worst case to ~0.9% (leaked stars have ci_sigma
    p50 = 0.66 — crowding-inflated CI is inconsistent across images — vs 0.08 for
    real galaxies; the guard keeps 100% of LS-confirmed ci > 1.6 galaxies).
  - Saturated bright stars (G ≲ 18.5) inflate to ci ~ 1.3–4; they carry Gaia/DR3
    star labels, so assemble's conflict logic drops them automatically.
  - End-to-end: 0496m662 (periphery): 17 `HST_GALAXY`, 11 LS/DR3-confirmed, 0 star
    conflicts. 0090m732 (blind interior): 10 net-new galaxy labels where every other
    galaxy source is empty. 0135m725 (SMC bar, densest pointing): 599 claims with
    127 Gaia-star conflicts (auto-dropped) → in the densest bar fields the surviving
    sub-Gaia labels are estimated only ~50% pure. **Dataset assembly (stage 3) must
    use the per-brick Gaia-conflict rate of `HST_GALAXY` as a label-noise monitor
    and drop/down-weight HST labels in bricks where it is high.**
  Bias to document: HST pointings oversample clusters/dense fields, and shredding
  makes the galaxy labels lean compact. **Split policy (2026-06-09)**: HST data is
  split by field/pointing between training and validation; interior fields go to
  training (nothing else labels the inner MCs), except a small holdout of interior
  pointings so release metrics can be quoted for the inner MCs, plus periphery
  fields for validation.
- Backend (implemented 2026-06-09): `tap.py` keeps a service registry (Data Lab +
  MAST, one cached `TAPService` each); `LabelSource.service` selects the endpoint and
  `LabelSource.max_rows` works around MAST's 100k-row hard cap (COUNT-verified,
  recursively dec-split box fetches). `tiles.py` unchanged — the box-condition ADQL
  is service-agnostic.
- **DR3 columns are never input features.** The released classifier must run from
  DELVE-MC catalog values alone, so train-time inputs are restricted to the same schema.
  DR3 enters through labels and evaluation only. (Possible later experiment, not v1:
  teacher–student distillation — a teacher trained on the cross-match overlap with both
  schemas produces soft targets for the MC-only student.)
- First actions when this stage starts: (a) TAP count queries to confirm LS DR10 and
  DELVE DR3 coverage of the MC footprint, (b) measure the DELVE-MC↔Gaia astrometric
  offset before fixing the match radius (start at 0.5").
- Per-source provenance (truth survey + source id) is stored; sources claimed by multiple
  truth sets with conflicting labels are dropped and counted (the Gaia×DR3 and LS×DR3
  disagreement rates double as a per-magnitude label-noise estimate for DR3).
- Label-independence caveat: LS DR10 south and DELVE DR3 partly share DECam imaging with
  DELVE-MC itself, so morphology labels are not fully independent of our features; the
  future HST/JWST set is the only truly independent validation.

### 2. Features (`features.py`)

- Extinction-corrected g, r, i, z via catalog `EBV` × DECam SFD coefficients, **plus
  explicit colors** g−r, r−i, i−z (and wide-baseline g−i). Both go in: boosted trees use
  axis-aligned splits, so a color–color locus is learnable with a couple of shallow
  splits when colors are explicit features, but must be approximated by many deep splits
  if the model only sees raw magnitudes. Magnitudes stay in too — they carry their own
  signal (the galaxy fraction and morphology reliability are strongly mag-dependent),
  and XGBoost is robust to the redundancy.
- Per-band `ERR`, `SCATTER`, `NDET`.
- `CHI`, `SHARP`, `PROB`, `ELLIPTICITY`, `ASEMI`/`BSEMI`.
- `FWHM` normalized by per-brick seeing (from `_meta.fits` exposures), and
  `MAG_AUTO` − PSF mag as a concentration proxy.
- u/Y dropped (too patchy). Missing values → NaN; XGBoost handles them natively.

### 3. Dataset assembly (`dataset.py`)

- Per-brick worker: quality cuts (`BRICKUNIQ == 1`, detected in ≥2 of g/r/i, sane
  errors) → cross-match against label tables → append labelled rows to parquet.
- **Spatial splits**: assign whole bricks to train/val/test via coarse HEALPix
  super-pixels so split boundaries are contiguous sky regions (no spatial leakage).
  Track magnitude/color/seeing/crowding distributions per split.

### 4–7. Train, calibrate, evaluate, package

- XGBoost binary logloss, `scale_pos_weight` for imbalance, early stopping on the
  spatial validation split; hyperparameters via CV over spatial folds.
- Isotonic calibration on a held-out spatial split.
- Evaluation: purity/completeness vs magnitude, color, seeing, crowding; ROC/PR;
  feature importance (+SHAP). Reference baselines: `SHARP` cut, `CHI` cut, `PROB` cut,
  and — on the DR3 cross-match overlap — the classic
  `|spread_model| < 0.003 + spreaderr_model` cut, the headline external comparison.
- Release: serialized model + calibrator, `morphocloud-infer` CLI streaming over bricks,
  model card documenting truth provenance, biases, validity range.

### Later (not blocking v1)

- ~~HST faint-end set from the Hubble Source Catalog v3~~ — promoted into stage 1
  label assembly (2026-06-09) after the coverage census showed HSC reaches 27% of the
  label-blind inner-MC bricks; see the HSC bullet above.
- JWST deep fields (e.g., the Time-Domain Field at the South Ecliptic Pole) as a
  further faint-end check.
- Tiers 2/3 (image-based models) on a remote server with more storage.
