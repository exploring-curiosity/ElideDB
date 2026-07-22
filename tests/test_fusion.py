"""Reciprocal rank fusion — the properties the retrieval fix depends on.

Run: ./myenv/bin/python -m pytest tests/test_fusion.py -q
     (or plain `./myenv/bin/python tests/test_fusion.py`)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.fusion import ranks_from_scores, rrf   # noqa: E402


def test_ranks_descending():
    r = ranks_from_scores([3.0, 1.0, 2.0])
    assert list(r) == [0.0, 2.0, 1.0]


def test_abstention_is_median_not_last():
    """NaN means 'no evidence'. It must not be treated as 'worst', or a
    ranker could veto every item it cannot score — which is every uncaptioned
    window for the lexical ranker."""
    r = ranks_from_scores([3.0, 1.0, np.nan, 2.0])
    known = [r[0], r[1], r[3]]
    assert r[2] == np.median(known)
    assert r[2] < max(known)          # strictly better than last


def test_all_abstain_is_neutral():
    r = ranks_from_scores([np.nan, np.nan])
    assert list(r) == [0.0, 0.0]


def test_consensus_beats_a_single_strong_signal():
    """THE property the 'crossing red car' fix rests on.

    A candidate that one ranker loves and another hates must lose to a
    candidate both rankers merely like. Under a weighted score sum the former
    wins whenever its strong signal has the wider spread; under RRF it cannot.
    """
    rng = np.random.default_rng(0)
    n = 200
    a, b = rng.random(n), rng.random(n)
    a[0], b[0] = 10.0, -10.0                  # rank 1 on a, rank last on b
    a[1] = np.sort(a)[-4] + 1e-9              # solid top-5 on both
    b[1] = np.sort(b)[-4] + 1e-9
    f = rrf({"a": a, "b": b})
    assert f[1] > f[0]
    fused_rank = (-f).argsort().argsort()
    assert fused_rank[1] == 0                 # consensus candidate wins
    assert fused_rank[0] > 5                  # one-trick candidate falls away


def test_weight_zero_disables_a_ranker():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    f = rrf({"a": a, "b": b}, {"a": 1.0, "b": 0.0})
    assert f[0] > f[1]


def test_mismatched_length_is_an_error():
    try:
        rrf({"a": np.zeros(3), "b": np.zeros(4)})
    except ValueError as e:
        assert "same candidate set" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
