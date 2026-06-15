"""Shared test setup.

Import xgboost before anything else so its OpenMP runtime loads first. On macOS
conda envs, loading scipy/healpy (pulled in by the label tests) before xgboost
makes a later ``Booster.load_model`` segfault on an OpenMP-runtime clash. Order
only matters when those packages share a process — inference-only installs
(numpy/pandas/astropy/xgboost) never hit it.
"""

import xgboost  # noqa: F401  (imported for its side effect: OpenMP load order)
