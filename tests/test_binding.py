"""Binding primitives: mechanical atom decomposition + variant-max.

atoms_of is the LLM-free decomposition (regex over determiner phrases)
that the conjunctive channel and the binding filter depend on; its
contract is one atom per noun phrase, >=2 atoms or conj abstains.
variant_max replaces the bare np.nanmax over query-variant score
arrays: all-NaN rows (episodes a channel cannot score) must stay NaN
WITHOUT numpy's All-NaN-slice warning polluting bench stdout.

Run: ./myenv/bin/python tests/test_binding.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.fusion import variant_max      # noqa: E402
from elidedb.sig2 import atoms_of           # noqa: E402


def test_atoms_two_atoms_spoon_cloth():
    a = atoms_of("place the spoon on top of the cloth")
    assert a == ["the spoon", "the cloth"]


def test_atoms_attribute_kept():
    a = atoms_of("put the green object into the drawer")
    assert "the green object" in a and "the drawer" in a


def test_atoms_preposition_is_boundary():
    """The q09 bug: 'into' swallowed as filler produced the single
    corrupt atom 'the eggplant into the', so conj abstained on a
    two-object binding query. Closed-class words end a phrase."""
    a = atoms_of("put the eggplant into the drawer")
    assert a == ["the eggplant", "the drawer"]


def test_atoms_conjunction_is_boundary():
    a = atoms_of("pick up a vessel and put it on the stove")
    assert a == ["a vessel", "the stove"]


def test_atoms_single_noun_means_abstain():
    assert len(atoms_of("close the drawer")) < 2


def test_variant_max_all_nan_column_silent():
    vs = [np.array([1.0, np.nan]), np.array([2.0, np.nan])]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = variant_max(vs)
    assert out[0] == 2.0 and np.isnan(out[1])
    assert not any("All-NaN" in str(x.message) for x in w)


def test_variant_max_single_variant_identity():
    out = variant_max([np.array([1.0, np.nan, 3.0])])
    assert out[0] == 1.0 and np.isnan(out[1]) and out[2] == 3.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
