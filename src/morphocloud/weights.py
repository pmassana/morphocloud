"""Resolve and, if needed, download the released model weights.

The wheel ships code only; the ~10 MB weights live as assets on the GitHub
release. ``fetch_weights`` downloads the four ``baseline_lshsc_xgb.*`` files to a
user cache on first use and returns the path to the booster ``.json``, verifying
each file against a pinned SHA-256. Local development is unaffected: when the
repo-local ``models/`` dir (or ``MORPHOCLOUD_MODELS_DIR``) already holds the
weights, ``infer`` uses those and nothing here runs.

Stdlib only (``urllib``) so inference adds no network dependency.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_STEM = "baseline_lshsc_xgb"
RELEASE_TAG = "v1.0.0"
_BASE_URL = (
    f"https://github.com/pmassana/morphocloud/releases/download/{RELEASE_TAG}/"
)

# suffix -> expected SHA-256 of the released asset. The booster .json is the
# entry point; the other three are loaded as siblings by StarGalaxyClassifier.
WEIGHT_FILES = {
    ".json": "b1c41a453924364564b6b15ea66e3198302ac9b6fb844f69a085780de41ec327",
    ".calibrator.json": "03bd204e165bc966a5ffef998f15f3aa70695fa4804ff54045c37b4dd9f2104d",
    ".meta.json": "fd38731f9503aa2dace5b0de5186f614262ccc964f48b75efb86ac286cde8d4e",
    ".thresholds.csv": "e2cf068aa901056a0132dd8d3d19b26e1b6560225473927e6849e38b59428f41",
}


def _cache_dir() -> Path:
    """Where downloaded weights live: ``MORPHOCLOUD_MODELS_DIR`` if the user set
    it (honour their explicit choice), else an XDG-style per-version cache."""
    explicit = os.environ.get("MORPHOCLOUD_MODELS_DIR")
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "morphocloud" / RELEASE_TAG


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically (temp file + rename)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "morphocloud"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def fetch_weights(dest_dir: Path | None = None, *, force: bool = False) -> Path:
    """Ensure the four weight files are present locally; return the booster path.

    Files already present with the expected SHA-256 are kept; anything missing,
    truncated, or mismatched is (re)downloaded from the GitHub release. Pass
    ``force=True`` to re-download regardless, or ``dest_dir`` to override the
    cache location.
    """
    dest_dir = Path(dest_dir) if dest_dir is not None else _cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for suffix, sha in WEIGHT_FILES.items():
        dest = dest_dir / f"{MODEL_STEM}{suffix}"
        if not force and dest.exists() and _sha256(dest) == sha:
            continue
        url = f"{_BASE_URL}{MODEL_STEM}{suffix}"
        print(f"morphocloud: downloading {dest.name} -> {dest_dir}", file=sys.stderr)
        _download(url, dest)
        got = _sha256(dest)
        if got != sha:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {dest.name}: expected {sha}, got {got}. "
                "The release asset may have changed; report this upstream."
            )
    return dest_dir / f"{MODEL_STEM}.json"
