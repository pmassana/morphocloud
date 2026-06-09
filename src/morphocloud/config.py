"""Paths and catalog conventions for the DELVE-MC y4t2 release."""

from pathlib import Path

# Local DELVE-MC brick catalogs (not part of this repo's storage budget)
DELVEMC_DATA = Path("/Users/pol.massana/Documents/DELVE/DELVEMC_data/delvemc_y4t2")

# Brick definitions for the full footprint (0.25 x 0.25 deg bricks)
BRICK_LIST = Path(
    "/Users/pol.massana/Documents/DELVE/codes/delvered/data/delvemc_bricks_0.25deg.fits.gz"
)

# Repo-local data products (gitignored, lives inside the ~100GB budget)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Catalog conventions (verified against y4t2 object files)
# - photometry columns: <BAND>MAG/<BAND>ERR/<BAND>SCATTER/NDET<BAND>, BAND in
#   present-by-brick subsets of {U, G, R, I, Z, Y}; g/r/i are near-complete,
#   z mostly present, u/Y patchy. Column sets VARY per brick.
MAG_SENTINEL = 99.0  # mags, PROB, CHI, SHARP use 99.99 for missing
FWHM_SENTINEL = 999_990.0  # FWHM uses 999999.0 for missing
CORE_BANDS = ("G", "R", "I", "Z")
