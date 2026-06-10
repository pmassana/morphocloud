"""Fetch and cache all truth-label tiles for the footprint.

Resumable: cached tiles are skipped, so this can be re-run after
interruptions. Usage:

    python scripts/fetch_labels.py [max_tiles] [source ...]

with source in {gaia_dr3, ls_dr10, delve_dr3, hsc_v3} (default: all four).
"""

import sys
import time

from morphocloud.bricks import load_brick_list
from morphocloud.labels import tiles
from morphocloud.labels.delvedr3 import DELVEDR3
from morphocloud.labels.gaia import GAIA
from morphocloud.labels.hsc import HSC
from morphocloud.labels.lsdr10 import LSDR10

SOURCES = {s.name: s for s in (GAIA, LSDR10, DELVEDR3, HSC)}
RETRIES = 3


def main(argv):
    max_tiles = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else None
    names = [a for a in argv[1:] if a in SOURCES] or list(SOURCES)

    pixels = tiles.footprint_pixels(load_brick_list())[:max_tiles]
    print(f"{len(pixels)} tiles x {names}")

    for name in names:
        src = SOURCES[name]
        done = sum(src.cache_path(p).exists() for p in pixels)
        print(f"\n=== {name}: {done}/{len(pixels)} already cached ===")
        for i, pix in enumerate(pixels):
            if src.cache_path(pix).exists():
                continue
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
