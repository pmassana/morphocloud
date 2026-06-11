"""Build the labelled training table, one parquet shard per brick.

Resumable: existing shards are skipped. Run only after fetch_labels.py has
cached every tile — missing tiles are fetched on demand, which is slow and
must not run concurrently with fetch_labels.py (same cache files). Usage:

    python scripts/build_dataset.py [max_bricks] [--jobs N]  # per-brick shards
    python scripts/build_dataset.py merge                    # single parquet + summary

Building is embarrassingly parallel once every label tile is cached (no
network, one independent shard per brick, atomic .tmp->rename writes): pass
--jobs N to fan the bricks across N worker processes.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from morphocloud import dataset
from morphocloud.bricks import available_bricks
from morphocloud.config import DATA_DIR

SHARD_DIR = DATA_DIR / "train" / "shards"
DATASET_PATH = DATA_DIR / "train" / "dataset.parquet"
SUMMARY_PATH = DATA_DIR / "train" / "split_summary.csv"


def _build_one(brick):
    """Build a single brick shard. Returns ("ok", brick, n_rows, split),
    ("skip", brick) if already built, or ("error", brick, message) — a single
    bad brick must never abort the pool. The write is atomic."""
    path = SHARD_DIR / f"{brick}.parquet"
    if path.exists():
        return ("skip", brick)
    try:
        df = dataset.brick_dataset(brick)
        tmp = path.with_suffix(f".parquet.{os.getpid()}.tmp")
        df.to_parquet(tmp)
        tmp.rename(path)
        return ("ok", brick, len(df), df["SPLIT"].iloc[0] if len(df) else "-")
    except Exception as e:
        return ("error", brick, f"{type(e).__name__}: {e}")


def build(max_bricks=None, jobs=1):
    names = available_bricks()[:max_bricks]
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    done = sum((SHARD_DIR / f"{b}.parquet").exists() for b in names)
    print(f"{len(names)} bricks, {done} shards already built, jobs={jobs}")
    todo = [b for b in names if not (SHARD_DIR / f"{b}.parquet").exists()]
    n = len(todo)
    t0 = time.time()
    if jobs <= 1:
        results = map(_build_one, todo)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(_build_one, todo, chunksize=8)
    errors = []
    for i, res in enumerate(results):
        if res[0] == "error":
            errors.append((res[1], res[2]))
            print(f"[{i + 1}/{n}] {res[1]} FAILED: {res[2]}", flush=True)
            continue
        if res[0] == "skip":
            continue
        _, brick, nrows, split = res
        if (i + 1) % 50 == 0 or i + 1 == n:
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / rate / 60
            print(f"[{i + 1}/{n}] {brick}: {nrows:,} rows ({split}) "
                  f"[{rate:.1f} bricks/s, ETA {eta:.0f} min]", flush=True)
    if jobs > 1:
        pool.shutdown()
    print(f"done: {n - len(errors)}/{n} built, {len(errors)} failed")
    for brick, msg in errors:
        print(f"  FAILED {brick}: {msg}")


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
    args = sys.argv[1:]
    if "merge" in args:
        merge()
    else:
        jobs = 1
        if "--jobs" in args:
            i = args.index("--jobs")
            jobs = int(args[i + 1])
            args = args[:i] + args[i + 2:]
        max_bricks = int(args[0]) if args else None
        build(max_bricks, jobs=jobs)
