"""Corpus-attested vocabulary: the self-recognized replacement for the
_HYPONYMS hand table (the last dictionary-of-the-dataset in the code).

Split per the no-hardwire rule's dictionary-vs-dataset test:
  - WordNet hypernymy is ENGLISH (corpus-independent, allowed);
  - WHICH hyponyms exist in THIS corpus is DATA — attested by scoring
    each candidate lemma against the store's SigLIP2 frame space and
    keeping only lemmas the corpus matches better than the query's own
    word. A kitchen store attests pot/pan for "vessel"; a street store
    would attest boat — nothing in code prefers either.
Attestations cache per store in _vocab.json (first query per new word
pays ~2s of text encodes, then free)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def hyponym_lemmas(word, depth=2):
    """Single-word WordNet hyponym lemmas over every noun sense —
    dictionary knowledge only, no corpus involved."""
    from nltk.corpus import wordnet as wn
    try:
        senses = wn.synsets(word, pos="n")
    except LookupError:
        import nltk
        nltk.download("wordnet", quiet=True)
        senses = wn.synsets(word, pos="n")
    out = []
    for syn in senses:
        frontier = [syn]
        for _ in range(depth):
            frontier = [h for s in frontier for h in s.hyponyms()]
            for s in frontier:
                for lem in s.lemma_names():
                    if ("_" not in lem and lem.isalpha()
                            and lem.lower() != word):
                        out.append(lem.lower())
    return list(dict.fromkeys(out))


def _top5(sc):
    k = min(5, len(sc))
    return float(np.sort(sc)[-k:].mean())


def _attested(store, word, cap):
    """Hyponyms of `word` that THIS corpus scores above the word
    itself — the corpus prefers the specific term or the category
    word stands."""
    from .sig2 import _frame_scores
    # score EVERY candidate, not a prefix: WordNet enumerates a
    # word's senses in an order that has nothing to do with this
    # corpus (for "vessel" the watercraft sense's ~150 hyponyms
    # precede the container sense's, so a [:40] prefix silently
    # dropped pot/pan/bowl before they could be scored — measured,
    # see docs/logs). The corpus decides via score, not WordNet's
    # traversal order.
    cands = hyponym_lemmas(word)
    if not cands:
        return []
    base = _top5(_frame_scores(store, f"a photo of a {word}"))
    keep = []
    for c in cands:
        sc = _top5(_frame_scores(store, f"a photo of a {c}"))
        if sc > base:
            keep.append((sc, c))
    return [c for _, c in sorted(keep, reverse=True)[:cap]]


def corpus_variants(store, text, cap=3):
    """Query variants substituting the first noun the corpus attests
    better hyponyms for; [text] alone when nothing is attested.
    Mirrors the one-substituted-noun shape of the old _HYPONYMS loop
    so text channels can keep taking max over variants."""
    from .sig2 import atoms_of
    cache_p = Path(store.dir) / "_vocab.json"
    cache = (json.loads(cache_p.read_text()) if cache_p.exists()
             else {})
    tl = text.lower()
    for atom in atoms_of(tl):
        w = atom.split()[-1]
        if w not in cache:
            cache[w] = _attested(store, w, cap)
            cache_p.write_text(json.dumps(cache, indent=1))
        if cache[w]:
            return [text] + [tl.replace(w, c) for c in cache[w]]
    return [text]
