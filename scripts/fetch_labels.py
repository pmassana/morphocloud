"""Fetch and cache all truth-label tiles for the footprint.

Resumable: cached tiles are skipped, so this can be re-run after
interruptions. Usage:

    python scripts/fetch_labels.py [max_tiles] [source ...] [--shard I/N]

with source in {gaia_dr3, gaia_galcand, gaia_qsocand, ls_dr10, delve_dr3,
hsc_v3} (default: all).
--shard I/N makes this worker handle every Nth tile starting at I, so N
parallel workers cover the footprint without ever racing on the same tile
(run each source list in its own process too — the archives are
independent). Example 3-way Gaia split alongside the Data Lab sources:

    python scripts/fetch_labels.py gaia_dr3 --shard 0/3 &
    python scripts/fetch_labels.py gaia_dr3 --shard 1/3 &
    python scripts/fetch_labels.py gaia_dr3 --shard 2/3 &
    python scripts/fetch_labels.py ls_dr10 delve_dr3 hsc_v3 &
"""

import sys
import time

from morphocloud.bricks import load_brick_list
from morphocloud.labels import tiles
from morphocloud.labels.delvedr3 import DELVEDR3
from morphocloud.labels.gaia import GAIA
from morphocloud.labels.gaia_extragal import GALCAND, QSOCAND
from morphocloud.labels.hsc import HSC
from morphocloud.labels.lsdr10 import LSDR10

SOURCES = {s.name: s for s in (GAIA, GALCAND, QSOCAND, LSDR10, DELVEDR3, HSC)}
RETRIES = 3


def main(argv):
    args = argv[1:]
    shard, nshards = 0, 1
    if "--shard" in args:
        i = args.index("--shard")
        shard, nshards = (int(x) for x in args[i + 1].split("/"))
        args = args[:i] + args[i + 2:]
    max_tiles = int(args[0]) if args and args[0].isdigit() else None
    names = [a for a in args if a in SOURCES] or list(SOURCES)

    pixels = tiles.footprint_pixels(load_brick_list())[:max_tiles]
    pixels = pixels[shard::nshards]
    print(f"{len(pixels)} tiles x {names}", flush=True)

    for name in names:
        src = SOURCES[name]
        done = sum(src.cache_path(p).exists() for p in pixels)
        print(f"\n=== {name}: {done}/{len(pixels)} already cached ===", flush=True)
        for i, pix in enumerate(pixels):
            if src.cache_path(pix).exists():
                continue
            print(f"[{i+1}/{len(pixels)}] {tiles.tile_id(pix)} ...", flush=True)
            for attempt in range(RETRIES):
                try:
                    t0 = time.time()
                    df = src.fetch_tile(pix)
                    print(f"[{i+1}/{len(pixels)}] {tiles.tile_id(pix)}: "
                          f"{len(df):,} rows [{time.time()-t0:.0f}s]", flush=True)
                    break
                except Exception as e:
                    wait = 30 * (attempt + 1)
                    print(f"[{i+1}/{len(pixels)}] {tiles.tile_id(pix)} failed "
                          f"({type(e).__name__}: {str(e)[:80]}), "
                          f"retry in {wait}s", flush=True)
                    time.sleep(wait)
            else:
                print(f"[{i+1}/{len(pixels)}] {tiles.tile_id(pix)}: GAVE UP", flush=True)


if __name__ == "__main__":
    main(sys.argv)
