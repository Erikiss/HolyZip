"""Exact integer arithmetic coder driven by per-step frequency tables.

The coder itself is fully deterministic integer arithmetic. The only source of
encoder/decoder disagreement is therefore the frequency table, i.e. the model
logits: both sides must produce bit-identical tables at every step.
"""
from __future__ import annotations

import numpy as np

STATE_BITS = 64
FULL = 1 << STATE_BITS
HALF = FULL >> 1
QUARTER = HALF >> 1
MASK = FULL - 1
TOTAL_BITS = 32  # frequency tables sum to at most 2**32


def freqs_from_probs(p: np.ndarray) -> np.ndarray:
    """Quantise float64 probabilities to integer frequencies >= 1.

    Every symbol keeps a frequency of at least 1, so any token stays codable;
    with a 248k vocabulary this floor costs < 0.0001 bit per token.
    """
    scale = (1 << TOTAL_BITS) - p.size
    return np.floor(p * scale).astype(np.int64) + 1


class Encoder:
    def __init__(self) -> None:
        self.low, self.high, self.pending = 0, MASK, 0
        self.bits: list[int] = []

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        if self.pending:
            self.bits.extend([1 - bit] * self.pending)
            self.pending = 0

    def encode(self, cum_lo: int, cum_hi: int, total: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + span * cum_hi // total - 1
        self.low = self.low + span * cum_lo // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


class Decoder:
    def __init__(self, data: bytes) -> None:
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.pos = 0
        self.low, self.high, self.code = 0, MASK, 0
        for _ in range(STATE_BITS):
            self.code = (self.code << 1) | self._next()

    def _next(self) -> int:
        bit = int(self.bits[self.pos]) if self.pos < len(self.bits) else 0
        self.pos += 1
        return bit

    def decode(self, cum: np.ndarray) -> int:
        """``cum`` is the inclusive cumulative frequency table (int64)."""
        total = int(cum[-1])
        span = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // span
        sym = int(np.searchsorted(cum, value, side="right"))
        cum_lo = int(cum[sym - 1]) if sym else 0
        cum_hi = int(cum[sym])
        self.high = self.low + span * cum_hi // total - 1
        self.low = self.low + span * cum_lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._next()
        return sym
