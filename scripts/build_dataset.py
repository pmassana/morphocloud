"""Build the labelled training table, one parquet shard per brick.

Resumable: existing shards are skipped. Run only after fetch_labels.py has
cached every tile — missing tiles are fetched on demand, which is slow and
must not run concurrently with fetch_labels.py (same cache files). Usage:

    python scripts/build_dataset.py [max_bricks]   # build per-brick shards
    python scripts/build_dataset.py merge          # single parquet + summary
"""

import sys
import time

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from morphocloud import dataset
from morphocloud.bricks import available_bricks
from morphocloud.config import DATA_DIR

SHARD_DIR = DATA_DIR / "train" / "shards"
DATASET_PATH = DATA_DIR / "train" / "dataset.parquet"
SUMMARY_PATH = DATA_DIR / "train" / "split_summary.csv"


def build(max_bricks=None):
    names = available_bricks()[:max_bricks]
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    done = sum((SHARD_DIR / f"{b}.parquet").exists() for b in names)
    print(f"{len(names)} bricks, {done} shards already built")
    for i, brick in enumerate(names):
        path = SHARD_DIR / f"{brick}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        df = dataset.brick_dataset(brick)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.rename(path)
        print(f"[{i + 1}/{len(names)}] {brick}: {len(df):,} rows "
              f"({df['SPLIT'].iloc[0] if len(df) else '-'}) "
              f"[{time.time() - t0:.0f}s]", flush=True)


def merge():
    shards = sorted(SHARD_DIR.glob("*.parquet"))
    print(f"merging {len(shards)} shards")
    schema = ds.dataset(shards[0]).schema
    with pq.ParquetWriter(DATASET_PATH, schema) as writer:
        for batch in ds.dataset(shards, schema=schema).to_batches():
            if batch.num_rows:
                writer.write_batch(batch)
    df = pd.read_parquet(
        DATASET_PATH,
        columns=["SPLIT", "BRICKNAME", "LABEL", "RMAG0", "G_R", "SEEING"],
    )
    summary = dataset.split_summary(df)
    summary.to_csv(SUMMARY_PATH)
    print(f"{len(df):,} rows -> {DATASET_PATH}")
    print(summary.round(3))


if __name__ == "__main__":
    if "merge" in sys.argv[1:]:
        merge()
    else:
        max_bricks = int(sys.argv[1]) if len(sys.argv) > 1 else None
        build(max_bricks)
