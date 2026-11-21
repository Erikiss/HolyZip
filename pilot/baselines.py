"""Classical compressors on the full verse-per-line files (their best case).

Writes pilot/results/baselines.json with compressed sizes in bytes.
"""
from __future__ import annotations

import bz2
import json
import lzma

import brotli
import pyppmd
import zstandard

from common import RESULTS, TRANSLATIONS, VPL


def xz(data: bytes) -> int:
    return len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))


def zstd(data: bytes) -> int:
    params = zstandard.ZstdCompressionParameters.from_level(22, window_log=27, enable_ldm=True)
    return len(zstandard.ZstdCompressor(compression_params=params).compress(data))


def ppmd(data: bytes) -> int:
    best = None
    for order in (6, 8, 12, 16):
        enc = pyppmd.Ppmd8Encoder(order, 1 << 30)
        n = len(enc.encode(data)) + len(enc.flush())
        best = n if best is None else min(best, n)
    return best


CODECS = {
    "bzip2 -9": lambda d: len(bz2.compress(d, 9)),
    "zstd -22 --long": zstd,
    "brotli -q11": lambda d: len(brotli.compress(d, quality=11, lgwin=24)),
    "xz -9e": xz,
    "PPMd (best order 6-16)": ppmd,
}


def main():
    out = {"per_translation": {}, "solid": {}}
    blobs = {}
    for tid in TRANSLATIONS:
        data = (VPL / f"{tid}.txt").read_bytes()
        blobs[tid] = data
        row = {"bytes": len(data)}
        for name, fn in CODECS.items():
            row[name] = fn(data)
        out["per_translation"][tid] = row
        best = min(v for k, v in row.items() if k != "bytes")
        print(f"{tid:12s} {len(data):>10,} B  best classical {8 * best / len(data):.3f} bits/byte", flush=True)

    # Does a classical compressor profit from seeing several translations at once?
    groups = {
        "KJV+WEB (same language)": ["eng-kjv2006", "engwebp"],
        "Luther1912+Elberfelder (same language)": ["deu1912", "deuelbbk"],
        "KJV+Luther1912 (different languages)": ["eng-kjv2006", "deu1912"],
        "all 17 translations": list(TRANSLATIONS),
    }
    for name, tids in groups.items():
        joined = b"".join(blobs[t] for t in tids)
        separate = {c: sum(out["per_translation"][t][c] for t in tids) for c in ("xz -9e", "PPMd (best order 6-16)")}
        solid = {"xz -9e": xz(joined), "PPMd (best order 6-16)": ppmd(joined)}
        out["solid"][name] = {"bytes": len(joined), "separate": separate, "solid": solid}
        print(name, {c: round(100 * (1 - solid[c] / separate[c]), 1) for c in solid}, "% saved by solid mode", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
