"""The byte-table gather must equal the sign-vector dot product.

patches.py never unpacks the codes: it scores 32 bytes per patch by
gathering from a (32, 256) table of partial sums. That is an
optimization standing in for an inner product, so it gets an exact
equivalence test — a silent bit-order mistake here would not crash,
it would just quietly rank badly, which is the failure mode this
project keeps paying for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.patches import _tables                          # noqa: E402


def test_byte_table_equals_sign_dot_product():
    rng = np.random.default_rng(0)
    dim = 256
    q = rng.standard_normal(dim).astype(np.float32)
    X = rng.standard_normal((64, dim)).astype(np.float32)
    codes = np.packbits((X > 0).astype(np.uint8), axis=-1)   # (64, 32)

    T = _tables(q, dim)
    ar = np.arange(dim // 8)
    got = T[ar, codes].sum(1)

    want = np.sign(X) @ q
    assert np.allclose(got, want, atol=1e-3), (got[:4], want[:4])


def test_table_shape_and_bit_order():
    # a code of all 1 bits means every dimension is +1
    dim = 64
    q = np.arange(dim, dtype=np.float32) - 32
    T = _tables(q, dim)
    ar = np.arange(dim // 8)
    ones = np.full((1, dim // 8), 255, np.uint8)
    zeros = np.zeros((1, dim // 8), np.uint8)
    assert np.isclose(T[ar, ones].sum(), q.sum(), atol=1e-3)
    assert np.isclose(T[ar, zeros].sum(), -q.sum(), atol=1e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
