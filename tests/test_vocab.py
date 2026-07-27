"""Corpus vocabulary — the dictionary half is deterministic WordNet.

The no-hardwire split: WordNet hypernymy is English (allowed);
which hyponyms the corpus attests is data (exercised live via the
bench store, not unit-tested here).

Run: ./myenv/bin/python tests/test_vocab.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.vocab import hyponym_lemmas    # noqa: E402


def test_vessel_covers_cookware():
    lems = hyponym_lemmas("vessel")
    # dictionary fact, corpus-independent; if this fails, raise the
    # depth argument — do NOT hand-add lemmas
    assert "pot" in lems


def test_lemmas_single_word_lowercase():
    for lem in hyponym_lemmas("container")[:50]:
        assert lem.isalpha() and lem == lem.lower()


def test_unknown_word_empty():
    assert hyponym_lemmas("zzzzqqq") == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
