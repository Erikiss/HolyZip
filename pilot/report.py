"""Aggregate pilot results into Markdown tables (stdout)."""
from __future__ import annotations

import json
import random
from collections import Counter

from common import RESULTS, TRANSLATIONS, VPL

NT_BOOKS = set(
    "MAT MAR LUK JOH ACT ROM 1CO 2CO GAL EPH PHI COL 1TH 2TH 1TI 2TI TIT PHM HEB JAM 1PE 2PE 1JO 2JO 3JO JUD REV".split()
)
PPMD = "PPMd (best order 6-16)"


def load():
    rows = [json.loads(l) for l in (RESULTS / "llm_bits.jsonl").read_text().splitlines()]
    keys = Counter((r["model"], r["mode"], r["pivot"], r["sample"], r["tid"], r["book"], r["chapter"]) for r in rows)
    assert max(keys.values()) == 1, "duplicate records in llm_bits.jsonl"
    base = json.loads((RESULTS / "baselines.json").read_text())
    return rows, base


def select(rows, chapters=None, **where):
    sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if chapters is not None:
        sel = [r for r in sel if (r["book"], r["chapter"]) in chapters]
    return sel


def pooled(sel):
    return sum(r["bits"] for r in sel) / sum(r["bytes"] for r in sel)


def nt_share(tid: str) -> float:
    """Share of the file's bytes that belongs to the New Testament."""
    nt = total = 0
    with open(VPL / f"{tid}.txt", "rb") as fh:
        for line in fh:
            total += len(line)
            if line[:3].decode() in NT_BOOKS:
                nt += len(line)
    return nt / total


def weighted(sel, share_nt):
    """Bits/byte with OT and NT chapters weighted by their share of the whole file."""
    ot = [r for r in sel if r["book"] not in NT_BOOKS]
    nt = [r for r in sel if r["book"] in NT_BOOKS]
    if not ot or not nt:
        return pooled(sel)
    return (1 - share_nt) * pooled(ot) + share_nt * pooled(nt)


def interval(sel, share_nt, reps=4000):
    """95% bootstrap interval over chapters (resampled within OT and NT)."""
    rng = random.Random(1)
    ot = [r for r in sel if r["book"] not in NT_BOOKS]
    nt = [r for r in sel if r["book"] in NT_BOOKS]
    vals = []
    for _ in range(reps):
        draw = [rng.choice(g) for g in (ot, nt) if g for _ in g]
        vals.append(weighted(draw, share_nt))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps)]


def short(model: str) -> str:
    return model.split("/")[-1].replace("Qwen3.5-", "")


def main():
    rows, base = load()
    per = base["per_translation"]
    best = {t: min(v for k, v in r.items() if k != "bytes") for t, r in per.items()}
    models = sorted({r["model"] for r in rows}, key=lambda m: ("Base" not in m, m))
    small = models[0]
    unc = dict(mode="uncond", pivot="", sample="main")

    print("### A. Classical compressors, full file (bits per byte, lower is better)\n")
    codecs = [k for k in next(iter(per.values())) if k != "bytes"]
    print("| translation | language | bytes | " + " | ".join(codecs) + " |")
    print("|---|---|--:|" + "--:|" * len(codecs))
    for t, r in per.items():
        print(f"| {t} | {TRANSLATIONS[t][0]} | {r['bytes']:,} | " + " | ".join(f"{8 * r[c] / r['bytes']:.3f}" for c in codecs) + " |")

    print(f"\n### B. LLM arithmetic-coding cost without cross-lingual context ({short(small)}, random chapters)\n")
    print("| translation | language | chapters (OT+NT) | tokens/byte | LLM bits/byte, sample | LLM bits/byte, weighted to file "
          "[95% interval] | best of the five classical codecs, full file | LLM as share of that |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    for t in TRANSLATIONS:
        sel = select(rows, model=small, tid=t, **unc)
        if not sel:
            continue
        share = nt_share(t)
        w = weighted(sel, share)
        lo, hi = interval(sel, share)
        cl = 8 * best[t] / per[t]["bytes"]
        n_nt = sum(r["book"] in NT_BOOKS for r in sel)
        assert all(r["lossless_tokenization"] for r in sel), t
        print(f"| {t} | {TRANSLATIONS[t][0]} | {len(sel) - n_nt}+{n_nt} | "
              f"{sum(r['tokens'] for r in sel) / sum(r['bytes'] for r in sel):.3f} | {pooled(sel):.3f} | "
              f"{w:.3f} [{lo:.2f}–{hi:.2f}] | {cl:.3f} | {100 * w / cl:.0f}% [{100 * lo / cl:.0f}–{100 * hi / cl:.0f}] |")

    print("\n### C. Same verse of an already-decoded translation in the context (identical chapters, bits per byte)\n")
    print("| target | pivot in context | model | chapters | alone | with pivot | change | verses with pivot |")
    print("|---|---|---|--:|--:|--:|--:|--:|")
    for m in models:
        for pivot in sorted({r["pivot"] for r in rows if r["mode"] == "cond" and r["model"] == m}):
            for t in TRANSLATIONS:
                cond = select(rows, model=m, mode="cond", pivot=pivot, sample="main", tid=t)
                if not cond:
                    continue
                ch = {(r["book"], r["chapter"]) for r in cond}
                alone = select(rows, chapters=ch, model=m, tid=t, **unc)
                if len(alone) != len(cond):
                    continue
                u, c = pooled(alone), pooled(cond)
                print(f"| {t} ({TRANSLATIONS[t][0]}) | {pivot} | {short(m)} | {len(cond)} | {u:.3f} | {c:.3f} | "
                      f"{100 * (c / u - 1):+.0f}% | {sum(r['pivot_matched'] for r in cond)}/{sum(r['verses'] for r in cond)} |")

    print("\n### D. Model comparison on identical chapters, no cross-lingual context (bits per byte)\n")
    print("| translation | models compared | chapters | " + " | ".join(short(m) for m in models) + " |")
    print("|---|---|--:|" + "--:|" * len(models))
    for t in TRANSLATIONS:
        sets = {m: {(r["book"], r["chapter"]) for r in select(rows, model=m, tid=t, **unc)} for m in models}
        for group in ([m for m in models if "Base" in m], [small] + [m for m in models if "Base" not in m]):
            have = [m for m in group if sets[m]]
            if len(have) < 2:
                continue
            common = set.intersection(*(sets[m] for m in have))
            cells = [f"{pooled(select(rows, chapters=common, model=m, tid=t, **unc)):.3f}" if m in have else "–" for m in models]
            print(f"| {t} | {' vs '.join(short(m) for m in have)} | {len(common)} | " + " | ".join(cells) + " |")

    print("\n### E. Famous chapters vs. random chapters (bits per byte)\n")
    print("| translation | model | GEN 1 | PSA 23 | JOH 1 | 1CO 13 | famous pooled | random chapters | n random |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for m in models:
        for t in TRANSLATIONS:
            fam = select(rows, model=m, mode="uncond", pivot="", sample="famous", tid=t)
            rnd = select(rows, model=m, tid=t, **unc)
            if len(fam) != 4 or not rnd:
                continue
            by = {r["book"]: r["bits"] / r["bytes"] for r in fam}
            print(f"| {t} | {short(m)} | " + " | ".join(f"{by[b]:.3f}" for b in ("GEN", "PSA", "JOH", "1CO"))
                  + f" | {pooled(fam):.3f} | {pooled(rnd):.3f} | {len(rnd)} |")

    print("\n### F. Classical compressors with two translations in one stream (bytes)\n")
    print("| pair | xz separate | xz solid | PPMd separate | PPMd solid |")
    print("|---|--:|--:|--:|--:|")
    for name, g in base["solid"].items():
        if name.startswith("all"):
            continue
        s, o = g["separate"], g["solid"]
        print(f"| {name} | {s['xz -9e']:,} | {o['xz -9e']:,} | {s[PPMD]:,} | {o[PPMD]:,} |")


if __name__ == "__main__":
    main()
