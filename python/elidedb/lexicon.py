"""Closed-class English lexicon shared by query routing and training.

Dependency-free ON PURPOSE: the query path imports this (via
rerank.directional_swap), and it previously reached the same tables
through fdnnv2, whose module level imports mlx. That worked on a Mac
and crashed every Linux deployment. Lexicon is language knowledge, not
model code; it lives where nothing heavier than a list comprehension
runs.

Verb/direction swaps: GENERIC english antonyms, no dataset nouns, per
the no-hardwire rule.
"""
from __future__ import annotations

# ---------------------------------------------------------------------
# DERIVED OPPOSITES. The 81 pairs below and the 16 `un-` bases were
# hand-authored, and fold/stack/cover are conspicuously this corpus's
# vocabulary - a task prior in a file that claims to hold generic
# English. The swap-contrast MECHANISM is sound and measured; the list
# was not derived.
#
# derived_swaps() replaces it: opposites are found by reflection in an
# embedding space over the vocabulary the CORPUS attested. On a driving
# log that finds accelerate/brake without anyone having thought of them;
# here it finds open/close. The hand list stays only as a cold-start
# fallback for an empty store, and is labelled as such rather than
# presented as knowledge.
# ---------------------------------------------------------------------


def derived_swaps(store, vec=None, min_count=2):
    """Opposite pairs this corpus supports, or () if it cannot say."""
    from .derive import attested, opposite_pairs
    try:
        if "labels" not in store.tables():
            return ()
        vals = store.table("labels").scan().column("value").to_pylist()
        vocab = attested(vals, min_count=min_count)
        if len(vocab) < 8:
            return ()
        if vec is None:
            from .sig2 import _text_vec as vec
        return tuple((a, b) for a, b, _ in opposite_pairs(vocab, vec))
    except Exception:
        return ()


# COLD-START FALLBACK ONLY - not knowledge, and not to be extended.
# Every entry here is a hardwiring violation that derived_swaps()
# supersedes as soon as a corpus has been ingested.
VERB_SWAPS = [
    ("open", "close"), ("opens", "closes"), ("opening", "closing"),
    ("opened", "closed"), ("into", "out of"), ("inside", "outside"),
    ("picks up", "puts down"), ("picking up", "putting down"),
    ("lifts", "lowers"), ("lifting", "lowering"),
    ("pushes", "pulls"), ("pushing", "pulling"),
    ("left", "right"), ("up", "down"), ("onto", "off"),
    ("toward", "away from"), ("front", "back"),
]

# The un- REVERSAL family: verbs whose opposite is their morphological
# negation. A query like "folding cloth" without a swap gets an
# ABSOLUTE contrast question, and encoders answer absolute questions
# with concept PRESENCE, not action direction (measured twice: open
# clips outscored close clips 0.36; tiger-in-drawer outscored
# fold-cloth +1.6 vs +0.17 because the drawer clip contains cloth).
# The swap-contrast cancels that bias by construction. Generic
# English, generated inflections, zero dataset words.
_UN_BASES = ["fold", "wrap", "roll", "stack", "cover", "screw", "plug",
             "zip", "tie", "load", "lock", "pack", "buckle", "hook",
             "fasten", "tangle"]


def _inflect(v):
    ing = (v[:-1] + "ing") if v.endswith("e") else (v + "ing")
    ed = (v + "d") if v.endswith("e") else (v + "ed")
    return [v, v + "s", ing, ed]


VERB_SWAPS += [(a, b) for base in _UN_BASES
               for a, b in zip(_inflect(base), _inflect("un" + base))]
