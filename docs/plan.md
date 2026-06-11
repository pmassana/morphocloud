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
- **Gaia DR3 backend — CDS Vizier (primary since 2026-06-10 evening)**, `vizier`
  service (`https://tapvizier.cds.unistra.fr/TAPVizieR/tap`), table `"I/355/gaiadr3"`
  = the same DR3 source catalogue (identical `source_id` and values; cross-validated
  against an ESA tile — pixel 8299 gave the *identical* 13,674 `source_id` set).
  Switched here after the ESA archive dropped into a 503 maintenance/capacity outage
  that stalled the run for ~2 h; ESA is expected to stay unstable through the Gaia
  **DR4** run-up, so Vizier is the default and ESA the fallback. Vizier specifics:
  it **renames the columns** (`Source`, `RA_ICRS`/`DE_ICRS` at epoch 2016.0, `Plx`,
  `Gmag`/`BPmag`/`RPmag`, `RUWE`, `epsi`=excess_noise, `sepsi`=excess_noise_sig,
  `IPDfmp`=ipd_frac_multi_peak), so `GAIA.rename` maps them back to the canonical
  lowercase names and `GAIA.cast` pins 6 cols to float32 → cached tiles are
  byte-schema-identical to the ESA tiles already on disk (they mix without
  conversion). Crucially, **Vizier needs real ADQL geometry**: a plain ra/dec-box
  `COUNT(*)` is unindexed and ~25× slower (38 s vs 1.5 s), so the tile box is sent
  as `CONTAINS(POINT, POLYGON(...))` via `LabelSource.geometry=True` (`id_col` dedup
  cleans the shared dec-edge of a split). It is fast (~13k rows/s: a 130k-row dense
  tile in ~57 s), `max_rows=500k`, COUNT-verified, `--shard I/N` parallelized.
  The same service also fetches the **Gaia DR3 extragalactic tables** `"I/356/galcand"`
  and `"I/356/qsocand"` (`gaia_galcand`/`gaia_qsocand` caches, `labels/gaia_extragal.py`):
  purer-union cuts run server-side, so tiles are tiny (~419k galaxy + ~271k QSO purer
  rows footprint-wide; raw would be 924k + 3.2M, dominated by MC contamination).
  Caveat: a *negated* CONTAINS (e.g. "outside a circle") is unindexed and crawls —
  exclusion geometry must be applied client-side or via inclusive counts.
- **ESA Gaia archive TAP** (`esa_gaia` service, `GAIA_ESA`; primary 2026-06-10 day
  after Data Lab async downloads degraded to ~20–50 KB/s, demoted to fallback that
  evening): table `gaiadr3.gaia_source`, canonical lowercase columns (no rename).
  Measured quirks: sync queries are capped at 60 s execution (counts only — never
  tile fetches); async jobs execute reliably but **linearly in result size at
  ~350 rows/s** (range vs ADQL-geometry conditions makes no difference; counts are
  index-fast either way); the job-status endpoint drops kept-alive connections
  routinely while a job executes, so polling must tolerate failures (`tap._run_async`,
  which replaced pyvo's fragile `run_async`). `max_rows=300k`.
- **Gaia (both backends)**: raw Gaia is too big to fetch (504M rows south of dec −44;
  5.4M in the LMC-center tile alone) → the `point_source_mask` cuts also run
  server-side (2.5× fewer rows in the dense interior; ~2.1M in the LMC-center tile),
  the local mask reapplies after the fetch so cached pre-cut tiles stay valid, and
  `fetch_labels.py --shard I/N` parallelizes tiles across workers.
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
  footprint) via anonymous TAP (Gaia from Vizier, LS/DR3 from Data Lab, HSC from MAST),
  cached as parquet. Label cuts run server-side (`LabelSource.where`) for all three
  DECam-era sources *and* Gaia (added 2026-06-10 — mandatory at ESA's ~350 rows/s).
  Periphery tiles measured 2026-06-09/10: Gaia ~15–30 s (Vizier; ~2 min on ESA),
  DR3 130 s, LS 76 s. The
  dense MC-interior tiles dominate the total (the LMC-center tile alone is ~2.1M Gaia
  candidates ≈ 1.7 h); expect a few days serial, or run `scripts/fetch_labels.py`
  sharded (`--shard I/N`) with one process per archive — resumable either way. Estimated a few GB total. `hsc_v3` (MAST) adds ~8 s for the
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
- **Galaxies ← Gaia DR3 galaxy_candidates / QSO flags ← qso_candidates (Vizier
  I/356)** — implemented 2026-06-10 (`labels/gaia_extragal.py`). The raw tables are
  completeness-driven and unusable here (south of dec −44.875: 924k "galaxies" /
  3.2M "QSOs", with 89k / 1.09M within 3° of the LMC center — misclassified
  Magellanic stars; Gaia's DSC did no sky-position filtering). Only the release
  paper's **purer** sub-samples are fetched (Gaia Collaboration, Bailer-Jones et
  al. 2023, A&A 674, A41 — both union definitions verified 2026-06-10 to reproduce
  the published counts *exactly* on Vizier: 2,891,132 galaxies / 1,942,825 quasars):
  galaxies `radius_sersic IS NOT NULL OR classlabel_dsc_joint='galaxy' OR
  vari_best_class_name='GALAXY'`; quasars `gaia_crf_source=1 OR host_galaxy_flag<6
  OR classlabel_dsc_joint='quasar' OR vari_best_class_name='AGN'`. The paper's
  ~95% purity excludes "generous regions" around the MCs — **LMC 9° around ICRS
  (81.3, −68.7), SMC 6° around (16.0, −72.8)** (its appendix ADQL, verbatim) — and
  the purer samples are still contaminated inside (purer-QSO density at the LMC is
  ~10× the sky value; RR-Lyrae-classed "quasars" pass even the DSC-joint cut), so
  both masks exclude the cores. Net labels: ~388k of 419k purer galaxies and 229k
  of 271k purer QSOs survive the circles; the circles cover 16.9% of footprint
  bricks, where Gaia-astrometry star labels and HST galaxy labels (crowding-robust)
  still apply. Roles (**decided 2026-06-10**): `GAIA_GALAXY` is a galaxy *vote* —
  the only wide-area galaxy source inside the MCs (though thin: ~700 purer labels
  in the LMC core pre-cut, bright/extended-biased, Gaia G ≲ 21); `GAIA_QSO` is a
  **provenance/evaluation flag only, never a vote** — the Tier 1 target is
  morphological and QSOs are point sources (86% of purer southern QSOs pass the
  `gaia_dr3` star cut, so their STAR labels are correct for this target). The flag
  supports QSO-contamination metrics (12k purer QSOs behind the LMC) and downstream
  masking; *physical* star-vs-extragalactic separation needs variability features
  (per-band `SCATTER` is only a weak proxy) and is **deferred to a future
  variable-source classifier** with time-series inputs. `IN_MC_CORE` (the same two
  circles) is carried through assembly into the dataset for split evaluation of
  the crowded cores.
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

### 2. Features (`features.py`) — implemented 2026-06-10

- Extinction-corrected g, r, i, z via catalog `EBV` × DECam SFD coefficients, **plus
  explicit colors** g−r, r−i, i−z (and wide-baseline g−i). Both go in: boosted trees use
  axis-aligned splits, so a color–color locus is learnable with a couple of shallow
  splits when colors are explicit features, but must be approximated by many deep splits
  if the model only sees raw magnitudes. Magnitudes stay in too — they carry their own
  signal (the galaxy fraction and morphology reliability are strongly mag-dependent),
  and XGBoost is robust to the redundancy. Coefficients: DES DR1 (Abbott et al. 2018)
  Fitzpatrick-99, A/E(B−V) = 3.186/2.140/1.569/1.196 — the set the DELVE DRs use.
- Per-band `ERR`, `SCATTER`, `NDET`.
- `CHI`, `SHARP`, `PROB`, `ELLIPTICITY`, `ASEMI`/`BSEMI`.
- `FWHM_RATIO` = coadd `FWHM` / per-brick seeing, plus the seeing itself (`SEEING`).
  Verified 2026-06-10: catalog `FWHM` is already in **arcsec** (point sources sit at
  the exposure seeing); per-brick seeing = median over the core-band medians from
  `_meta.fits` (the coadd mixes all bands, so one band-agnostic normalizer).
- `CONC` concentration proxy from `MAG_AUTO` − PSF mag. **Found 2026-06-10:
  `MAG_AUTO` is on the coadd's instrumental zero point** (offsets of −4.4 to −5.9 mag
  that vary per brick), so the raw difference is not survey-comparable. Two-stage
  per-brick anchoring on bright point-like stars (`PROB ≥ 0.8`, `|SHARP| ≤ 0.3`,
  16–20.5 mag, ≥20 per band): per-band median subtracted, then the per-source median
  over bands re-centered on the anchors → CONC ≡ 0 for the brick's point sources by
  construction, positive = extended (validated: extended `FWHM_RATIO > 2` sources sit
  +0.4 to +2.1 mag above point sources). Stars keep a real mag-dependent CONC trend in
  crowded bricks (saturation/neighbor flux) — fine for a tree that also sees mags.
- u/Y dropped (too patchy). Missing values → NaN; XGBoost handles them natively.

### 3. Dataset assembly (`dataset.py`) — implemented 2026-06-10

- Per-brick worker `brick_dataset`: quality cuts (`BRICKUNIQ == 1` via the reader;
  detected with error < 0.5 in ≥2 of g/r/i) → `assemble.brick_labels` →
  `features.brick_features` → labelled rows only (conflicts/unlabelled excluded).
- **HST label-noise guard** (the stage-3 QC required by the stage-1 validation):
  per brick, the fraction of `HST_GALAXY` claims colliding with star labels is
  computed and stored (`HST_CONFLICT_RATE`); where it exceeds 0.2, galaxy labels
  backed *only* by HST are dropped. Validated: the SMC-bar brick 0135m725 (rate
  0.21) loses its HST-only labels, 0090m732 (rate 0.17) keeps its 10 net-new
  interior galaxies.
- **Spatial splits**: whole bricks assigned by their center to HEALPix nside=16
  nest superpixels (~13.4 deg², ~210 bricks each), each superpixel hashed
  deterministically (md5, machine-independent) into train/val/test = 70/15/15.
  `split_summary` tracks per-split counts, star fraction, and r-mag/g−r/seeing
  quantiles (written to `data/train/split_summary.csv` at merge time).
- `scripts/build_dataset.py`: resumable shard-per-brick builder (atomic writes,
  skip-if-exists, same pattern as fetch_labels) + `merge` step streaming all shards
  into `data/train/dataset.parquet`. Run only after the label fetch has cached all
  tiles (missing tiles are fetched on demand; must not race fetch_labels.py).
- **Open**: the HST *field-level* train/val split override (interior pointings →
  training, small interior holdout + periphery → validation) needs the full HSC
  fetch to enumerate pointings; the generic superpixel split stands in until then.

### 3b. Label QA (`notebooks/qa_labels.ipynb`) — added 2026-06-10

Visual + quantitative QA of the assembled labels before training. Gitignored
(carries ~10 MB of embedded cutouts; regenerable). Run with the `morphocloud`
Jupyter kernel; products land under `data/qa/`.

- **Pool**: `dataset.brick_dataset` over the bricks whose gaia+ls+hsc tiles are all
  cached (no on-demand fetches) → cached to `data/qa/labelled_pool.parquet`.
- **Stratified sampling** by provenance source (gaia/ls/hst/dr3) × r-mag bin, so
  faint/rare labels are inspected, not just bright easy ones.
- **Contact sheets** of fast Legacy-viewer RGB cutouts (`ls-dr10`, non-blank inside
  the MCs); green=star / red=galaxy border, `PROB`/`SHARP`/mag in the title.
- **Catalog-only cross-check**: label vs pipeline `PROB`/`SHARP`/`CONC`; flags the
  worst disagreements (initial run: 31.7% of star labels have pipeline PROB<0.5,
  15.7% of galaxy labels PROB>0.5 — most likely DELVE `PROB` miscalibration given the
  clean SHARP separation, star 0.24 vs galaxy 1.9, but worth eyeballing).
- **CONFLICT / HST_BLEND** review (rebuilds the rows `brick_dataset` drops).
- **Native-depth SIA** (`ls_dr10` g/r/z; `delve_dr3` SIA is empty in the inner MCs) —
  per-object `deep_look` and a **viewer-vs-SIA side-by-side** for the suspects.
- Writes `data/qa/inspection_*.csv` scaffolds (empty `verdict` column) to record calls.

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
