"""Zero-shot composite sweep: find the strongest OFF-THE-SHELF scoring
before wiring anything. All variants are pure math over tables already
on disk (SigLIP per-frame vectors, X-CLIP-large clip vectors) — no new
models, no metadata, no learning. Balanced sample, precision@10, the
14-query bench.

Variants:
  mean        recording = mean of frame vectors (the incumbent)
  fmax        recording = MAX frame cosine (the moment that matters)
  ftop5       mean of top-5 frame cosines (robust max)
  +subj       query also asked as subject-anchored variant, best-of
  +prompt     prompt ensemble (bare / 'a video of' / subject), best-of
  xclip       X-CLIP-large clip vector, own text tower
  fuse        RRF of the best SigLIP variant + xclip
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elidedb import Store                                    # noqa: E402
from elidedb.embeddings import _vec_table                    # noqa: E402
from elidedb.context import embed_texts                      # noqa: E402
from elidedb.fusion import ranks_from_scores                 # noqa: E402
from regress10 import QUERIES                                # noqa: E402


def main():
    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    eps = [(stream_of.get(int(i)), int(a), int(b), (k or "").lower())
           for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"],
                                 t["task"]) if k]
    rng = np.random.default_rng(0)
    chosen = {}
    for q, pred in QUERIES:
        rel = [e for e in eps if pred(e[3])]
        for e in (rel if len(rel) <= 25 else
                  [rel[i] for i in rng.choice(len(rel), 25,
                                              replace=False)]):
            chosen[(e[0], e[1])] = e
    dist = [e for e in eps if (e[0], e[1]) not in chosen]
    for e in [dist[i] for i in rng.choice(len(dist),
                                          min(150, len(dist)),
                                          replace=False)]:
        chosen[(e[0], e[1])] = e
    sample = list(chosen.values())

    ft, F = _vec_table(db, "frame_vectors")
    fs = np.array(ft.column("stream").to_pylist())
    fts = np.array([int(v) for v in ft.column("ts").to_pylist()])
    xt, X = _vec_table(db, "xclip_vectors")
    xkey = {(s, int(a)): i for i, (s, a) in enumerate(
        zip(xt.column("stream").to_pylist(),
            xt.column("ts").to_pylist()))}

    frames_of, labels, xrows = [], [], []
    for s, a, b, lab in sample:
        m = np.where((fs == s) & (fts >= a) & (fts <= b))[0]
        if len(m) < 8 or (s, a) not in xkey:
            continue
        frames_of.append(m)
        labels.append(lab)
        xrows.append(xkey[(s, a)])
    n = len(labels)
    print(f"{n} recordings")
    Xs = np.asarray(X[xrows])

    from elidedb.subjects import subject_prefixes
    subj = (subject_prefixes(db, build=False) or ["a robot arm"])[0]

    qs = [q for q, _ in QUERIES]
    variants = {
        "bare": qs,
        "subj": [f"{subj} {q}" for q in qs],
        "video": [f"a video of {q}." for q in qs],
    }
    T = {k: embed_texts(v) for k, v in variants.items()}

    from elidedb.vid import _text_vec
    TX = np.stack([_text_vec(q) for q in qs])

    def frame_scores(qv):
        """per-recording arrays of frame cosines for one query vec."""
        sc = F @ qv
        return [sc[m] for m in frames_of]

    def grade(score_fn):
        total = 0
        per = []
        for qi, (q, pred) in enumerate(QUERIES):
            s = score_fn(qi)
            top = np.argsort(-s)[:10]
            g = sum(pred(labels[i]) for i in top)
            per.append(g)
            total += g
        return total, per

    def pool(fsc, how):
        if how == "mean":
            return np.array([v.mean() for v in fsc])
        if how == "max":
            return np.array([v.max() for v in fsc])
        if how == "top5":
            return np.array([np.sort(v)[-5:].mean() for v in fsc])

    cache = {}

    def siglip(qi, texts, how):
        key = (id(texts), qi)
        if key not in cache:
            cache[key] = frame_scores(texts[qi])
        return pool(cache[key], how)

    results = {}
    for how in ("mean", "max", "top5"):
        results[f"siglip-{how}"], _ = grade(
            lambda qi: siglip(qi, T["bare"], how))
        results[f"siglip-{how}+subj"], _ = grade(
            lambda qi: np.maximum(siglip(qi, T["bare"], how),
                                  siglip(qi, T["subj"], how)))
    results["siglip-top5+prompt3"], _ = grade(
        lambda qi: np.maximum.reduce([siglip(qi, T[k], "top5")
                                      for k in T]))
    results["xclip"], _ = grade(lambda qi: Xs @ TX[qi])

    def fuse(qi):
        a = np.maximum(siglip(qi, T["bare"], "top5"),
                       siglip(qi, T["subj"], "top5"))
        b = Xs @ TX[qi]
        return (1.0 / (61.0 + ranks_from_scores(a)) +
                1.0 / (61.0 + ranks_from_scores(b)))
    results["fuse(sig-top5+subj, xclip)"], per = grade(fuse)

    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{v:4d}/140  {k}")
    print("\nfused per-query:", per)


if __name__ == "__main__":
    main()
