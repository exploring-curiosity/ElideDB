"""Shared set-path selection — the fit-live contract.

Everything the fitter searches over must execute through these
functions and only these; the knee/obj-boost divergences (fit LOQO
0.21 vs live 0.16, measured) are the regression class this kills.

Run: ./myenv/bin/python tests/test_setpath.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.setpath import (confidence_cut, filter_mask,   # noqa: E402
                             rankfrac)


def test_rankfrac_nan_votes_neutral():
    r = rankfrac(np.array([3.0, np.nan, 1.0]))
    assert r[0] == 1.0 and r[1] == 0.5 and r[2] == 0.0


def test_filter_bottom_quantile_dies():
    ch = {"mot": np.array([0.9, 0.1, 0.5, 0.7])}
    alive = filter_mask(ch, ["mot"], 1 / 3)
    assert list(alive) == [True, False, True, True]


def test_filter_median_disagreement_cancels():
    ch = {"mot": np.array([0.9, 0.1, 0.5, 0.7]),
          "prf": np.array([0.1, 0.9, 0.5, 0.7])}
    alive = filter_mask(ch, ["mot", "prf"], 1 / 3)
    assert alive.all()


def test_filter_no_evidence_all_alive():
    ch = {"pe": np.array([0.9, 0.1]),
          "conj": np.array([np.nan, np.nan])}
    alive = filter_mask(ch, ["conj"], 1 / 3)
    assert alive.all()


def test_filter_zero_quantile_all_alive():
    ch = {"mot": np.array([0.9, 0.1, 0.5])}
    assert filter_mask(ch, ["mot"], 0.0).all()


def test_filter_all_nan_channel_excluded_from_median():
    """The one semantic change of the unification: a channel that is
    NaN for EVERY episode is excluded from the median (fit semantics),
    not included as a neutral 0.5 row (old live semantics). With the
    dead channel excluded, the single live channel decides alone."""
    ch = {"mot": np.array([0.9, 0.1, 0.5, 0.7]),
          "conj": np.array([np.nan, np.nan, np.nan, np.nan])}
    alive = filter_mask(ch, ["mot", "conj"], 1 / 3)
    assert list(alive) == [True, False, True, True]


def test_filter_empty_channels_no_crash():
    assert filter_mask({}, ["mot"], 0.5).shape == (0,)


def test_cut_alpha_zero_is_plain_topk():
    s = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert confidence_cut(s, 0.0, 3) == 3
    assert confidence_cut(s, 0.0, 10) == 5


def test_cut_drops_low_confidence_tail():
    # top-5 mean 0.654; alpha 0.9 -> floor 0.589 -> first 3 survive
    s = np.array([1.0, 0.99, 0.98, 0.2, 0.1])
    assert confidence_cut(s, 0.9, 10) == 3


def test_cut_confident_head_never_empty_for_sane_alpha():
    # s[0] >= top-5 mean always, so alpha <= 1 keeps at least one
    s = np.array([1.0, 0.1])
    assert confidence_cut(s, 0.95, 10) >= 1


def test_cut_bounded_by_kmax():
    s = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert confidence_cut(s, 0.5, 4) == 4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
