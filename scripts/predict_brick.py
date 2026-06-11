"""Tier 1 release: classify DELVE-MC brick(s) into a calibrated P(star).

Thin CLI over morphocloud.infer.StarGalaxyClassifier. Writes one output table
per brick (FITS by default, parquet with --format parquet) carrying the
calibrated stellar probability and the quality flag. Inference reads only the
DELVE-MC object catalogs and per-brick metadata - no truth catalog.

    python scripts/predict_brick.py BRICK [BRICK ...] --out-dir preds/
    python scripts/predict_brick.py 0001m740 --format parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from astropy.table import Table

from morphocloud.infer import DEFAULT_MODEL, StarGalaxyClassifier


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bricks", nargs="+", help="brick name(s), e.g. 0001m740")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out-dir", default="preds")
    ap.add_argument("--format", choices=("fits", "parquet"), default="fits")
    ap.add_argument(
        "--unique-only", action="store_true",
        help="keep only BRICKUNIQ==1 sources (default: classify every source)",
    )
    args = ap.parse_args()

    clf = StarGalaxyClassifier.load(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model {args.model}  ({len(clf.features)} features)\n")

    for brick in args.bricks:
        df = clf.classify_brick(brick, unique_only=args.unique_only)
        path = out_dir / f"{brick}_class.{args.format}"
        if args.format == "parquet":
            df.to_parquet(path, index=False)
        else:
            Table.from_pandas(df).write(path, overwrite=True)
        n_star = int((df["P_STAR"] >= 0.5).sum())
        print(
            f"{brick}: {len(df):>7,} sources  "
            f"{n_star:>7,} star (P>=0.5)  {df['QUALITY_PASS'].sum():>7,} pass cut "
            f"-> {path}"
        )


if __name__ == "__main__":
    main()
