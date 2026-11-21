"""Aggregate pilot results into Markdown tables (stdout)."""
from __future__ import annotations

import json
from collections import defaultdict

from common import RESULTS, TRANSLATIONS

BEST = "best classical"


def load():
    rows = [json.loads(l) for l in (RESULTS / "llm_bits.jsonl").read_text().splitlines()]
    base = json.loads((RESULTS / "baselines.json").read_text())
    return rows, base


def agg(rows, **where):
    """Pooled bits/byte over chapters (sum bits / sum bytes) plus bookkeeping."""
    sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if not sel:
        return None
    bits, nbytes = sum(r["bits"] for r in sel), sum(r["bytes"] for r in sel)
    per = sorted(r["bits"] / r["bytes"] for r in sel)
    return dict(
        bpb=bits / nbytes, chapters=len(sel), bytes=nbytes, tokens=sum(r["tokens"] for r in sel),
        lo=per[0], hi=per[-1], keys={(r["book"], r["chapter"]) for r in sel},
        lossless=all(r["lossless_tokenization"] for r in sel),
        matched=sum(r["pivot_matched"] for r in sel), verses=sum(r["verses"] for r in sel),
    )


def paired(rows, settings: list[dict]):
    """Pooled bits/byte of several settings on exactly the chapters all of them have."""
    aggs = [agg(rows, **w) for w in settings]
    if not all(aggs):
        return None
    common = set.intersection(*(a["keys"] for a in aggs))
    out = []
    for where in settings:
        sel = [r for r in rows if all(r[k] == v for k, v in where.items()) and (r["book"], r["chapter"]) in common]
        out.append(sum(r["bits"] for r in sel) / sum(r["bytes"] for r in sel))
    return out, len(common)


def short(model: str) -> str:
    return model.split("/")[-1].replace("Qwen3.5-", "")


def main():
    rows, base = load()
    per = base["per_translation"]
    best = {t: min(v for k, v in r.items() if k != "bytes") for t, r in per.items()}
    models = sorted({r["model"] for r in rows}, key=lambda m: ("Base" not in m, m))

    print("### A. Classical compressors, full file (bits per byte, lower is better)\n")
    codecs = [k for k in next(iter(per.values())) if k != "bytes"]
    print("| translation | language | bytes | " + " | ".join(codecs) + " |")
    print("|---|---|--:|" + "--:|" * len(codecs))
    for t, r in per.items():
        print(f"| {t} | {TRANSLATIONS[t][0]} | {r['bytes']:,} | " + " | ".join(f"{8 * r[c] / r['bytes']:.3f}" for c in codecs) + " |")

    small = models[0]
    print(f"\n### B. LLM arithmetic-coding cost, no cross-lingual context ({short(small)}, random-chapter sample)\n")
    print("| translation | language | chapters | sample bytes | tokens/byte | LLM bits/byte | chapter range | "
          "best classical (full file) | LLM size vs. best classical |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for t in TRANSLATIONS:
        a = agg(rows, model=small, mode="uncond", sample="main", tid=t)
        if not a:
            continue
        cl = 8 * best[t] / per[t]["bytes"]
        note = "" if a["lossless"] else " (tokenizer not lossless!)"
        print(f"| {t}{note} | {TRANSLATIONS[t][0]} | {a['chapters']} | {a['bytes']:,} | {a['tokens'] / a['bytes']:.3f} | "
              f"{a['bpb']:.3f} | {a['lo']:.2f}–{a['hi']:.2f} | {cl:.3f} | {100 * a['bpb'] / cl:.0f}% |")

    print("\n### C. Conditioning on an already-decoded translation of the same verse (paired chapters)\n")
    print("| target | pivot in context | model | chapters | alone | with pivot | saving | verses with pivot |")
    print("|---|---|---|--:|--:|--:|--:|--:|")
    for m in models:
        for pivot in sorted({r["pivot"] for r in rows if r["mode"] == "cond" and r["model"] == m}):
            for t in TRANSLATIONS:
                c = agg(rows, model=m, mode="cond", pivot=pivot, sample="main", tid=t)
                p = paired(rows, [dict(model=m, mode="uncond", pivot="", sample="main", tid=t),
                                  dict(model=m, mode="cond", pivot=pivot, sample="main", tid=t)])
                if not c or not p:
                    continue
                (u, cc), n = p
                print(f"| {t} ({TRANSLATIONS[t][0]}) | {pivot} | {short(m)} | {n} | {u:.3f} | {cc:.3f} | "
                      f"{100 * (1 - cc / u):.0f}% | {c['matched']}/{c['verses']} |")

    print("\n### D. Model comparison on identical chapters, no cross-lingual context (bits per byte)\n")
    print("| translation | chapters | " + " | ".join(short(m) for m in models) + " |")
    print("|---|--:|" + "--:|" * len(models))
    for t in TRANSLATIONS:
        have = [m for m in models if agg(rows, model=m, mode="uncond", sample="main", tid=t)]
        if len(have) < 2:
            continue
        vals, n = paired(rows, [dict(model=m, mode="uncond", sample="main", tid=t) for m in have])
        by = dict(zip(have, vals))
        print(f"| {t} | {n} | " + " | ".join(f"{by[m]:.3f}" if m in by else "–" for m in models) + " |")

    print("\n### E. Memorisation probe: famous chapters vs. random chapters (bits per byte)\n")
    print("| translation | model | famous (GEN 1, PSA 23, JOH 1, 1CO 13) | random sample |")
    print("|---|---|--:|--:|")
    for m in models:
        for t in TRANSLATIONS:
            f = agg(rows, model=m, mode="uncond", sample="famous", tid=t)
            r = agg(rows, model=m, mode="uncond", sample="main", tid=t)
            if f and r:
                print(f"| {t} | {short(m)} | {f['bpb']:.3f} | {r['bpb']:.3f} |")

    print("\n### F. Solid mode for classical compressors (several translations in one stream, bytes)\n")
    print("| group | xz separate | xz solid | PPMd separate | PPMd solid |")
    print("|---|--:|--:|--:|--:|")
    for name, g in base["solid"].items():
        s, o = g["separate"], g["solid"]
        print(f"| {name} | {s['xz -9e']:,} | {o['xz -9e']:,} | {s['PPMd (best order 6-16)']:,} | {o['PPMd (best order 6-16)']:,} |")


if __name__ == "__main__":
    main()
