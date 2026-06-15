"""Paths and catalog conventions for the DELVE-MC y4t2 release.

The local-data paths are read from the environment so no machine-specific path
ships in the package. Inference on a catalog you already have in memory needs
none of them; only the on-disk brick workflows (fetching, dataset assembly,
``bricks``/``classify_brick``) do. Set, in your shell profile:

    export MORPHOCLOUD_DELVEMC_DATA=/path/to/delvemc_y4t2
    export MORPHOCLOUD_BRICK_LIST=/path/to/delvemc_bricks_0.25deg.fits.gz
    export MORPHOCLOUD_MODELS_DIR=/path/to/weights   # optional; default: repo models/
"""

import os
from pathlib import Path


def _env_path(var: str) -> Path | None:
    val = os.environ.get(var)
    return Path(val) if val else None


# Local DELVE-MC brick catalogs (not part of this repo's storage budget)
DELVEMC_DATA = _env_path("MORPHOCLOUD_DELVEMC_DATA")

# Brick definitions for the full footprint (0.25 x 0.25 deg bricks)
BRICK_LIST = _env_path("MORPHOCLOUD_BRICK_LIST")

# Repo-local data products (gitignored, lives inside the ~100GB budget)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
# Model weights: override for pip-installed use (released assets live wherever
# you download them); defaults to the repo's models/ dir for local development.
MODELS_DIR = _env_path("MORPHOCLOUD_MODELS_DIR") or (
    Path(__file__).resolve().parents[2] / "models"
)


def require_delvemc_data() -> Path:
    """The local brick-catalog dir, or a clear error if it isn't configured."""
    if DELVEMC_DATA is None:
        raise RuntimeError(
            "set MORPHOCLOUD_DELVEMC_DATA to your DELVE-MC y4t2 brick directory "
            "(only the on-disk brick workflows need it; in-memory inference does not)"
        )
    return DELVEMC_DATA


def require_brick_list() -> Path:
    """The footprint brick-list file, or a clear error if it isn't configured."""
    if BRICK_LIST is None:
        raise RuntimeError(
            "set MORPHOCLOUD_BRICK_LIST to the delvemc_bricks_0.25deg.fits.gz path"
        )
    return BRICK_LIST

# Catalog conventions (verified against y4t2 object files)
# - photometry columns: <BAND>MAG/<BAND>ERR/<BAND>SCATTER/NDET<BAND>, BAND in
#   present-by-brick subsets of {U, G, R, I, Z, Y}; g/r/i are near-complete,
#   z mostly present, u/Y patchy. Column sets VARY per brick.
MAG_SENTINEL = 99.0  # mags, PROB, CHI, SHARP use 99.99 for missing
FWHM_SENTINEL = 999_990.0  # FWHM uses 999999.0 for missing
CORE_BANDS = ("G", "R", "I", "Z")
