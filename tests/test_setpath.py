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

from elidedb.setpath import (confidence_cut, event_positions,  # noqa: E402
                             filter_mask, nms_keep, rankfrac)


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


def _toy_geometry():
    # two streams: A with 4 consecutive episodes, B with 2
    keys = [("A", 10, 19), ("A", 20, 29), ("A", 30, 39), ("A", 40, 49),
            ("B", 10, 19), ("B", 20, 29)]
    return event_positions(keys)


def test_nms_zero_radius_is_identity():
    sid, pos = _toy_geometry()
    order = np.array([3, 0, 1, 4])
    assert list(nms_keep(order, sid, pos, 0)) == [3, 0, 1, 4]


def test_nms_suppresses_adjacent_same_stream():
    sid, pos = _toy_geometry()
    # ranked: A0 best, A1 adjacent duplicate, B0 distinct, A3 far
    order = np.array([0, 1, 4, 3])
    assert list(nms_keep(order, sid, pos, 1)) == [0, 4, 3]


def test_nms_keeps_other_stream_same_position():
    sid, pos = _toy_geometry()
    order = np.array([0, 4])     # A pos0 and B pos0 — different events
    assert list(nms_keep(order, sid, pos, 2)) == [0, 4]


def test_nms_radius_two_reaches_further():
    sid, pos = _toy_geometry()
    order = np.array([0, 2, 3])  # A pos0, pos2, pos3
    assert list(nms_keep(order, sid, pos, 2)) == [0, 3]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
