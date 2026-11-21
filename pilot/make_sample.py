"""Fix the chapter samples used by the LLM measurements (seeded, reproducible).

main    6 OT + 6 NT chapters drawn uniformly from chapters that exist in every
        full-Bible translation of the pilot (and in the OT-only / NT-only ones)
        have at most 50 verses in the KJV (keeps CPU runs short) and have the
        same verse count (+-2) in all of them (same versification).
famous  four of the best-known chapters, as a memorisation probe.
"""
from __future__ import annotations

import json
import random

from common import RESULTS, TRANSLATIONS, load_vpl

OT_ONLY = {"hboWLC"}
NT_ONLY = {"grcbyz", "aai"}
SEED = 20261121
MAX_VERSES = 50


def main():
    books = {tid: load_vpl(tid) for tid in TRANSLATIONS}
    kjv = books["eng-kjv2006"]
    order = list(kjv.keys())
    nt_start = next(i for i, (b, _) in enumerate(order) if b == "MAT")
    pools = {"ot": [], "nt": []}
    for i, key in enumerate(order):
        part = "nt" if i >= nt_start else "ot"
        skip = NT_ONLY if part == "ot" else OT_ONLY
        if len(kjv[key]) > MAX_VERSES:
            continue
        present = [tid for tid in TRANSLATIONS if tid not in skip]
        if not all(key in books[tid] for tid in present):
            continue
        # Same verse count (+-2) everywhere = proxy for identical versification,
        # so that "the same chapter" really is the same content in all languages.
        if all(abs(len(books[tid][key]) - len(kjv[key])) <= 2 for tid in present):
            pools[part].append(key)
    rng = random.Random(SEED)
    picked = set(rng.sample(pools["ot"], 6)) | set(rng.sample(pools["nt"], 6))
    sample = {
        "main": [k for k in order if k in picked],
        "famous": [("GEN", 1), ("PSA", 23), ("JOH", 1), ("1CO", 13)],
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "seed": SEED,
        "max_verses": MAX_VERSES,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sample.json").write_text(json.dumps(sample, indent=1) + "\n")
    print(json.dumps(sample))


if __name__ == "__main__":
    main()
