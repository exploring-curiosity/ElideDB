"""STEPS 7-8 tested where it counts: does DISCRETISING units help?

The label oracle reaches 0.942/0.890 on the same oracle spans where
continuous unit vectors reach 0.600/0.400. The labels differ from the
vectors in exactly two ways: they are DISCRETE, and they are noiseless.
This measures how much of the gap the discreteness alone is worth.

Mechanism worth stating: DTW over continuous vectors ACCUMULATES error
- every step adds a noisy cosine, so a 23-unit sequence sums 23 noisy
terms. Symbols either match or they do not, and a histogram over an
episode AVERAGES noise instead of accumulating it. That is the reason
to expect a difference, and it predicts histograms should beat DTW
specifically when per-unit accuracy is poor - which is our regime
(action AUC 0.691 against a chance of 0.5).

Matchers, all on the SAME cached unit vectors so only the matching
changes:

  dtw       current: cosine DTW over the unit sequence
  mean      one vector per episode (mean of units) - the crudest
            possible pooling, included because it is the thing to beat
  bag       k-means vocabulary -> symbol histogram -> cosine
  bigram    same vocabulary, ordered PAIRS of consecutive symbols; the
            template is defined by an action SEQUENCE, so pure bags
            should lose information a bigram keeps
  soft      soft assignment to the vocabulary (no hard argmax), which
            avoids throwing away a unit that sits between two symbols

The vocabulary is fitted on the corpus WITHOUT labels; labels only
score. k is swept and reported rather than chosen quietly.

    python native/match7.py --enc siglip2_rank --seg uni:3:1.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg                                       # noqa: E402
from u6retr import score, units                                # noqa: E402
from unitenc import FPS                                        # noqa: E402


def vocab_fit(V, k, seed=0):
    from sklearn.cluster import KMeans
    km = KMeans(k, n_init=5, random_state=seed).fit(V)
    C = km.cluster_centers_
    return C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-8)


def hist(V, C, mode="bag", temp=0.05):
    S = V @ C.T
    k = len(C)
    if mode == "soft":
        W = np.exp((S - S.max(1, keepdims=True)) / temp)
        W = W / W.sum(1, keepdims=True)
        h = W.sum(0)
    elif mode == "bag":
        a = np.argmax(S, 1)
        h = np.bincount(a, minlength=k).astype(np.float32)
    elif mode == "bigram":
        a = np.argmax(S, 1)
        h = np.zeros(k * k, np.float32)
        for x, y in zip(a[:-1], a[1:]):
            h[x * k + y] += 1.0
        h = np.concatenate([np.bincount(a, minlength=k), h])
    else:
        raise ValueError(mode)
    h = np.asarray(h, np.float32)
    # sqrt = Hellinger: stops one dominant symbol swamping the cosine
    h = np.sqrt(h)
    return h / max(np.linalg.norm(h), 1e-8)


def main():
    import encode as E
    import pyarrow.parquet as pq
    from tqdm import tqdm

    limit = arg("--limit", 150, int)
    enc = arg("--enc", "siglip2_rank")
    seg = arg("--seg", "uni:3:1.0")
    KS = [int(x) for x in arg("--ks", "5,8,12,20,32").split(",")]

    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl, ev = {}, {}
    for e, tm, a, b in zip(t["episode"], t["template"], t["t0"],
                           t["t1"]):
        tmpl[int(e)] = tm
        ev.setdefault(int(e), []).append((float(a), float(b)))

    def uniform(dur, w, st):
        out, x = [], 0.0
        while x + w <= dur + 1e-6:
            out.append((x, x + w))
            x += st
        return out or [(0.0, dur)]

    dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                  if p.is_dir() and p.name.startswith("ep"))[:limit]
    seqs = {}
    for d in tqdm(dirs, desc="units", unit="ep"):
        ei = int(d.name[2:])
        if ei not in ev:
            continue
        cam = sorted(d.glob("cam*.mp4"))[0]
        dur = E.probe_duration(cam)
        F = E.decode(cam, fps=FPS, w=256)
        sp = (sorted(ev[ei]) if seg == "truth"
              else uniform(dur, *[float(x) for x in seg.split(":")[1:]]))
        seqs[ei] = units(enc, seg, ei, F, sp)

    allV = np.concatenate([v for v in seqs.values()])
    print(f"\n{enc} / {seg}: {len(seqs)} episodes, {len(allV)} units\n")
    print(f"{'matcher':<16}{'k':<6}{'yield':<9}{'prec':<9}"
          f"{'support':<9}{'returned'}")

    y, p, su, rt = score(seqs, tmpl)
    print(f"{'dtw':<16}{'-':<6}{y:<9.3f}{p:<9.3f}{su:<9.1f}{rt:.1f}",
          flush=True)

    ms = {e: (v.mean(0) / max(np.linalg.norm(v.mean(0)), 1e-8))[None, :]
          for e, v in seqs.items()}
    y, p, su, rt = score(ms, tmpl)
    print(f"{'mean':<16}{'-':<6}{y:<9.3f}{p:<9.3f}{su:<9.1f}{rt:.1f}",
          flush=True)

    for k in KS:
        C = vocab_fit(allV, k)
        for mode in ("bag", "bigram", "soft"):
            hs = {e: hist(v, C, mode)[None, :] for e, v in seqs.items()}
            y, p, su, rt = score(hs, tmpl)
            print(f"{mode:<16}{k:<6}{y:<9.3f}{p:<9.3f}{su:<9.1f}"
                  f"{rt:.1f}", flush=True)


if __name__ == "__main__":
    main()
