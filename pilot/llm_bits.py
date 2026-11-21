"""Measure the ideal arithmetic-coding cost of Bible text under a causal LM.

For every sampled chapter we compute sum(-log2 p(token | context)) over the
tokens of that chapter. That sum is the size an arithmetic coder driven by the
model would produce (to within a few bytes), so bits / bytes is the achievable
compression rate *excluding* the model weights.

Two modes:
  uncond  context = the preceding lines of the same translation.
  cond    context additionally contains, before each verse, the same verse
          from an already-decoded pivot translation (e.g. the KJV). Only the
          target-translation tokens are charged.

Results are appended to pilot/results/llm_bits.jsonl (one record per chapter).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM

from common import RESULTS, TRANSLATIONS, load_tokenizer, load_vpl, verse_key

LN2 = math.log(2.0)
NT_BOOKS = set(
    "MAT MAR LUK JOH ACT ROM 1CO 2CO GAL EPH PHI COL 1TH 2TH 1TI 2TI TIT PHM HEB JAM 1PE 2PE 1JO 2JO 3JO JUD REV".split()
)
SAMPLE_FILE = RESULTS / "sample.json"
OUT = RESULTS / "llm_bits.jsonl"


def build_segments(tid, chapter, pivot, n_prefix):
    """Return [(text, counted)] for one chapter. Text pieces end with a newline."""
    target = load_vpl(tid)
    flat, start = [], {}
    for key, lines in target.items():
        start[key] = len(flat)
        flat.extend(line for _, line in lines)
    first = start[chapter]
    last = first + len(target[chapter])
    lo = max(0, first - n_prefix)
    pivot_map = {}
    if pivot:
        for lines in load_vpl(pivot).values():
            for _, line in lines:
                pivot_map[verse_key(line)] = line
    segs, matched = [], 0
    for i in range(lo, last):
        line = flat[i]
        if pivot:
            p = pivot_map.get(verse_key(line))
            if p is not None:
                segs.append((p + "\n", False))
                matched += i >= first
        segs.append((line + "\n", i >= first))
    return segs, matched, last - first


@torch.no_grad()
def score(model, tok, segs, start_id, head_chunk=128):
    ids, counted, lossless = [start_id], [False], True
    for text, is_counted in segs:
        piece = tok(text, add_special_tokens=False).input_ids
        if is_counted and tok.decode(piece) != text:
            lossless = False
        ids.extend(piece)
        counted.extend([is_counted] * len(piece))
    x = torch.tensor([ids])
    hidden = model.model(input_ids=x).last_hidden_state[0]  # [n, d]
    pos = [i for i in range(1, len(ids)) if counted[i]]
    bits = 0.0
    for a in range(0, len(pos), head_chunk):
        p = torch.tensor(pos[a : a + head_chunk])
        logits = model.lm_head(hidden[p - 1]).double()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = x[0, p]
        bits += -lp.gather(1, tgt[:, None]).sum().item() / LN2
    n_bytes = sum(len(t.encode("utf-8")) for t, c in segs if c)
    return bits, n_bytes, len(pos), len(ids), lossless


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["uncond", "cond"], default="uncond")
    ap.add_argument("--pivot", default="")
    ap.add_argument("--translations", required=True, help="comma-separated eBible ids")
    ap.add_argument("--sample", default="main", help="key in results/sample.json")
    ap.add_argument("--prefix-lines", type=int, default=None)
    ap.add_argument("--nt-only", action="store_true", help="restrict the sample to its New Testament chapters")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "cond" and not args.pivot:
        ap.error("--mode cond needs --pivot")
    n_prefix = args.prefix_lines if args.prefix_lines is not None else (6 if args.mode == "cond" else 12)

    torch.set_num_threads(args.threads)
    chapters = [tuple(c) for c in json.loads(SAMPLE_FILE.read_text())[args.sample]]
    if args.nt_only:
        chapters = [c for c in chapters if c[0] in NT_BOOKS]
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["mode"], r["pivot"], r["sample"], r["tid"], r["book"], r["chapter"]))

    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    start_id = tok.convert_tokens_to_ids("<|endoftext|>")

    for tid in args.translations.split(","):
        assert tid in TRANSLATIONS, tid
        have = load_vpl(tid)
        for book, chap in chapters:
            key = (args.model, args.mode, args.pivot, args.sample, tid, book, chap)
            if key in done or (book, chap) not in have:
                continue
            t0 = time.time()
            segs, matched, n_verses = build_segments(tid, (book, chap), args.pivot, n_prefix)
            bits, n_bytes, n_tok, n_ctx, lossless = score(model, tok, segs, start_id)
            rec = dict(
                model=args.model, mode=args.mode, pivot=args.pivot, sample=args.sample,
                tid=tid, book=book, chapter=chap, verses=n_verses, pivot_matched=matched,
                bits=round(bits, 2), bytes=n_bytes, tokens=n_tok, ctx_tokens=n_ctx,
                lossless_tokenization=lossless, secs=round(time.time() - t0, 2),
            )
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(
                f"{args.model.split('/')[-1]} {args.mode}{'|' + args.pivot if args.pivot else ''} "
                f"{tid} {book} {chap}: {bits / n_bytes:.3f} bits/byte "
                f"({n_tok} tok, {n_bytes} B, {rec['secs']}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
