"""Download the pilot translations from eBible.org (verse-per-line format).

Only the ``<id>_vpl.txt`` member is read out of each archive; nothing else is
extracted. Texts land in ``data/vpl/<id>.txt`` (git-ignored).
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile

from common import DATA, TRANSLATIONS, VPL

URL = "https://ebible.org/Scriptures/{tid}_vpl.zip"


def fetch(tid: str) -> None:
    out = VPL / f"{tid}.txt"
    if out.exists():
        print(f"{tid}: cached ({out.stat().st_size:,} bytes)")
        return
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # eBible rejects urllib's default User-Agent, so send a generic one.
    req = urllib.request.Request(URL.format(tid=tid), headers={"User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    (raw / f"{tid}_vpl.zip").write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        member = f"{tid}_vpl.txt"
        text = zf.read(member)
    text.decode("utf-8")  # must be valid UTF-8
    VPL.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text)
    print(f"{tid}: {len(text):,} bytes sha256={hashlib.sha256(text).hexdigest()[:16]}")


if __name__ == "__main__":
    for tid in sys.argv[1:] or TRANSLATIONS:
        fetch(tid)
