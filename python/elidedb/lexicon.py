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
