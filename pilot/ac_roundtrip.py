"""Real lossless round trip with an LM-driven arithmetic coder + determinism probes.

1. Compress one chapter with the model run token-by-token (state cache),
   decompress with the same code path, and check the bytes are identical.
2. Probe how fragile that is: compare the integer frequency tables produced by
   (a) the same incremental path a second time,
   (b) one full-sequence ("teacher forced") forward pass - the fast way to
       encode, which a decoder cannot reproduce because it must go token by token,
   (c) the incremental path with a different thread count.
   Any position whose table differs would desynchronise a real decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from ac import Decoder, Encoder, freqs_from_probs
from common import RESULTS, load_tokenizer, load_vpl


class Stepper:
    """Feeds tokens one at a time and returns the next-token frequency table."""

    def __init__(self, model, start_id: int) -> None:
        self.model, self.past = model, None
        self.next_input = start_id

    @torch.no_grad()
    def table(self) -> np.ndarray:
        out = self.model(
            input_ids=torch.tensor([[self.next_input]]), past_key_values=self.past, use_cache=True
        )
        self.past = out.past_key_values
        probs = torch.softmax(out.logits[0, -1].double(), dim=-1).numpy()
        return freqs_from_probs(probs)

    def push(self, token: int) -> None:
        self.next_input = token


def compress(model, start_id, ids, trace=None):
    enc, step, ideal = Encoder(), Stepper(model, start_id), 0.0
    for t in ids:
        f = step.table()
        cum = np.cumsum(f)
        lo = int(cum[t - 1]) if t else 0
        enc.encode(lo, int(cum[t]), int(cum[-1]))
        ideal += -math.log2(f[t] / cum[-1])
        if trace is not None:
            trace.append((hashlib.blake2b(f.tobytes(), digest_size=16).digest(), lo, int(cum[t])))
        step.push(t)
    return struct.pack("<I", len(ids)) + enc.finish(), ideal


def decompress(model, start_id, blob):
    (n,) = struct.unpack("<I", blob[:4])
    dec, step, ids = Decoder(blob[4:]), Stepper(model, start_id), []
    for _ in range(n):
        t = dec.decode(np.cumsum(step.table()))
        ids.append(t)
        step.push(t)
    return ids


@torch.no_grad()
def teacher_forced_trace(model, start_id, ids, chunk=64):
    x = torch.tensor([[start_id] + ids[:-1]])
    hidden = model.model(input_ids=x).last_hidden_state[0]
    trace = []
    for a in range(0, len(ids), chunk):
        logits = model.lm_head(hidden[a : a + chunk]).double()
        probs = torch.softmax(logits, dim=-1).numpy()
        for j in range(probs.shape[0]):
            f = freqs_from_probs(probs[j])
            cum = np.cumsum(f)
            t = ids[a + j]
            trace.append(
                (hashlib.blake2b(f.tobytes(), digest_size=16).digest(), int(cum[t - 1]) if t else 0, int(cum[t]))
            )
    return trace


def diff(a, b):
    tables = [i for i, (x, y) in enumerate(zip(a, b)) if x[0] != y[0]]
    intervals = [i for i, (x, y) in enumerate(zip(a, b)) if x[1:] != y[1:]]
    return dict(
        positions=len(a),
        tables_differ=len(tables),
        first_table_mismatch=tables[0] if tables else None,
        target_interval_differs=len(intervals),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--tid", default="deu1912")
    ap.add_argument("--book", default="LUK")
    ap.add_argument("--chapter", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--alt-threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    start_id = tok.convert_tokens_to_ids("<|endoftext|>")

    text = "".join(line + "\n" for _, line in load_vpl(args.tid)[(args.book, args.chapter)])
    raw = text.encode("utf-8")
    ids = tok(text, add_special_tokens=False).input_ids
    assert tok.decode(ids).encode("utf-8") == raw, "tokenizer is not lossless on this text"

    trace_a: list = []
    t0 = time.time()
    blob, ideal_bits = compress(model, start_id, ids, trace_a)
    t_enc = time.time() - t0
    t0 = time.time()
    out_ids = decompress(model, start_id, blob)
    t_dec = time.time() - t0
    restored = tok.decode(out_ids).encode("utf-8")

    trace_b: list = []
    compress(model, start_id, ids, trace_b)
    trace_tf = teacher_forced_trace(model, start_id, ids)
    torch.set_num_threads(args.alt_threads)
    trace_c: list = []
    compress(model, start_id, ids, trace_c)

    res = dict(
        model=args.model, tid=args.tid, book=args.book, chapter=args.chapter,
        original_bytes=len(raw), tokens=len(ids), compressed_bytes=len(blob),
        ideal_bytes=round(ideal_bits / 8, 1), bits_per_byte=round(8 * len(blob) / len(raw), 4),
        lossless_roundtrip=restored == raw,
        sha256_original=hashlib.sha256(raw).hexdigest(),
        sha256_restored=hashlib.sha256(restored).hexdigest(),
        encode_tok_per_s=round(len(ids) / t_enc, 1), decode_tok_per_s=round(len(ids) / t_dec, 1),
        threads=args.threads,
        determinism=dict(
            same_path_repeat=diff(trace_a, trace_b),
            incremental_vs_teacher_forced=diff(trace_a, trace_tf),
            **{f"threads_{args.threads}_vs_{args.alt_threads}": diff(trace_a, trace_c)},
        ),
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"ac_roundtrip_{args.tid}_{args.book}{args.chapter}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
