"""Shared helpers for the HolyZip pilot: corpus loading and chapter sampling."""
from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VPL = DATA / "vpl"
RESULTS = Path(__file__).resolve().parent / "results"

# eBible.org translation ids used in the pilot. All are marked redistributable
# by eBible; the texts are downloaded on demand and never committed.
TRANSLATIONS = OrderedDict(
    [
        ("eng-kjv2006", ("English", "King James Version", "public domain")),
        ("engwebp", ("English", "World English Bible", "public domain")),
        ("deu1912", ("German", "Luther 1912", "public domain")),
        ("deuelbbk", ("German", "Elberfelder (revised, bibelkommentare.de)", "© 2019 VdcG, CC BY-NC-ND")),
        ("fraLSG", ("French", "Louis Segond 1910", "public domain")),
        ("spaRV1909", ("Spanish", "Reina-Valera 1909", "public domain")),
        ("latVUC", ("Latin", "Clementine Vulgate", "public domain")),
        ("russyn", ("Russian", "Synodal", "public domain")),
        ("cmn-cu89s", ("Chinese", "Union Version (simplified)", "public domain")),
        ("kor", ("Korean", "Korean Bible 1910", "public domain")),
        ("vie1934", ("Vietnamese", "Vietnamese Bible 1923", "public domain")),
        ("hin2017", ("Hindi", "Indian Revised Version", "© BCS, redistributable")),
        ("swhulb", ("Swahili", "Unlocked Literal Bible", "© Door43, redistributable")),
        ("yor", ("Yoruba", "Yoruba Bible", "© Biblica, redistributable")),
        ("hboWLC", ("Hebrew", "Westminster Leningrad Codex (OT)", "public domain")),
        ("grcbyz", ("Greek", "Byzantine NT", "public domain")),
        ("aai", ("Miniafia (PNG)", "Miniafia NT", "© Wycliffe, redistributable")),
    ]
)


def load_vpl(tid: str) -> "OrderedDict[tuple[str, int], list[tuple[int, str]]]":
    """Return {(book, chapter): [(line_no, raw_line_without_newline), ...]}.

    The raw line (e.g. ``GEN 1:1 In the beginning ...``) is kept byte-exact so
    that every size we report refers to bytes of the original file.
    """
    chapters: "OrderedDict[tuple[str, int], list[tuple[int, str]]]" = OrderedDict()
    path = VPL / f"{tid}.txt"
    with open(path, encoding="utf-8", newline="") as fh:
        for i, line in enumerate(fh.read().split("\n")):
            if not line:
                continue
            ref = line.split(" ", 2)
            book, cv = ref[0], ref[1]
            chap = int(cv.split(":")[0])
            chapters.setdefault((book, chap), []).append((i, line))
    return chapters


def verse_key(line: str) -> tuple[str, str]:
    book, cv, *_ = line.split(" ", 2)
    return book, cv


def sample_chapters(tids: list[str], n: int, seed: int = 20261121) -> list[tuple[str, int]]:
    """Pick ``n`` chapters at random (fixed seed) among chapters that exist in
    every translation in ``tids``; returned in canonical order."""
    common = None
    order = None
    for tid in tids:
        keys = list(load_vpl(tid).keys())
        if order is None:
            order = keys
        common = set(keys) if common is None else common & set(keys)
    pool = [k for k in order if k in common]
    rng = random.Random(seed)
    picked = set(rng.sample(pool, min(n, len(pool))))
    return [k for k in pool if k in picked]


def load_tokenizer(model_id: str):
    """Qwen tokenizer with its NFC normaliser switched off.

    The stock tokenizer normalises to NFC, which silently rewrites text that is
    not NFC and would make the codec lossy. In the pilot corpus this hits the
    pointed/cantillated Hebrew of the Westminster Leningrad Codex. Without the
    normaliser encode->decode is byte-exact; for text that already is NFC the
    token ids are unchanged.
    """
    from tokenizers import normalizers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.backend_tokenizer.normalizer = normalizers.Sequence([])
    return tok
