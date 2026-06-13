"""How many HSC point-like (candidate star) sources survive a 0.5" isolation cut.

Motivation: the Tier 1 model's star purity falls after r~23 because Gaia is the
only faint star-label source. HSC could add faint stars inside/near the MCs, but
HST resolves blends DECam sees as one source, so an HSC "star" sitting inside a
crowded clump would be a bad DECam star label. The proposed guard: drop any
HSC point-like source that has another HSC detection within 0.5" (DECam's PSF /
match radius) — it is blended from the ground.

This script quantifies how many point-like sources that cut removes, tile by
tile over the cached HSC catalog, with no DELVE cross-match yet (it measures the
HSC candidate pool itself). Edge effect: nearest-neighbor is computed within a
tile, so sources within 0.5" of a tile border could have an unseen neighbor
across it -- negligible at these tile sizes, and it only makes the "kept" count
a slight over-estimate.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from morphocloud.labels import hsc
from morphocloud.labels.gaia_extragal import in_mc_core

ISO_ARCSEC = 0.5
TILES = sorted(glob.glob("data/labels/hsc_v3/hp32-*.parquet"))

# F606W ~ r-band; collect the optical F606W / F814W magnitude per source,
# preferring ACS, then WFC3, then WFPC2.
F606 = ("a_f606w", "w3_f606w", "w2_f606w")
F814 = ("a_f814w", "w3_f814w", "w2_f814w")


def _coalesce(df: pd.DataFrame, cols) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for c in cols:
        out = np.where(np.isnan(out), df[c].to_numpy(), out)
    return out


def _nn_arcsec(ra: np.ndarray, dec: np.ndarray, ra_all, dec_all) -> np.ndarray:
    """Nearest-neighbor sep (arcsec) of each (ra,dec) to the full source set,
    excluding self. Local tangent-plane KDTree (small-angle, fine at <1')."""
    dec0 = np.radians(np.median(dec_all))
    x_all = ra_all * np.cos(dec0)
    tree = cKDTree(np.column_stack([x_all, dec_all]))
    q = np.column_stack([ra * np.cos(dec0), dec])
    # k=2: nearest is self (dist 0); take the second.
    dist, _ = tree.query(q, k=2)
    return dist[:, 1] * 3600.0  # deg -> arcsec


def main() -> None:
    rows = []
    for i, path in enumerate(TILES):
        df = pd.read_parquet(path)
        if len(df) == 0:
            continue
        cand = hsc.point_source_mask(df)          # 0 < ci < 1.2
        optical = df[list(hsc.OPTICAL_MAGS)].notna().any(axis=1)
        cand &= optical                            # drop IR-only sources
        n_cand = int(cand.sum())
        if n_cand == 0:
            continue

        sub = df[cand]
        nn = _nn_arcsec(
            sub["matchra"].to_numpy(), sub["matchdec"].to_numpy(),
            df["matchra"].to_numpy(), df["matchdec"].to_numpy(),
        )
        blended = nn < ISO_ARCSEC
        core = in_mc_core(sub["matchra"], sub["matchdec"])
        r606 = _coalesce(sub, F606)
        r814 = _coalesce(sub, F814)

        rows.append(pd.DataFrame({
            "nn": nn, "blended": blended, "core": core,
            "r606": r606, "r814": r814,
        }))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(TILES)} tiles")

    res = pd.concat(rows, ignore_index=True)
    res.to_parquet("data/labels/coverage/hsc_star_isolation.parquet")

    n = len(res)
    blended = res["blended"]
    print("\n=== HSC point-like (candidate star) sources, 0.5\" isolation cut ===")
    print(f"total point-like candidates (optical, numimages>=2): {n:,}")
    print(f"  blended (neighbor <0.5\", EXCLUDED): {blended.sum():,} "
          f"({100*blended.mean():.1f}%)")
    print(f"  isolated (KEPT):                    {(~blended).sum():,} "
          f"({100*(~blended).mean():.1f}%)")

    print("\n-- by region --")
    for label, m in [("MC core", res["core"]), ("outside core", ~res["core"])]:
        sub = res[m]
        if len(sub) == 0:
            continue
        print(f"{label:>13}: {len(sub):>10,} candidates | "
              f"excluded {sub['blended'].sum():>9,} ({100*sub['blended'].mean():4.1f}%) | "
              f"kept {(~sub['blended']).sum():>9,}")

    bins = [0, 20, 21, 22, 23, 24, 25, 99]
    exc = res[blended]
    kept = res[~blended]
    for band in ("r814", "r606"):
        name = {"r814": "F814W (~i/z)", "r606": "F606W (~r)"}[band]
        print(f"\n-- candidates by {name} magnitude --")
        ktab = kept.groupby(pd.cut(kept[band], bins), observed=True).size()
        etab = exc.groupby(pd.cut(exc[band], bins), observed=True).size()
        for b in ktab.index:
            print(f"  {str(b):>12}: kept {ktab.get(b,0):>9,} | "
                  f"excluded {etab.get(b,0):>9,}")
        print(f"  no {band} detection: kept {kept[band].isna().sum():>9,} | "
              f"excluded {exc[band].isna().sum():>9,}")


if __name__ == "__main__":
    main()
