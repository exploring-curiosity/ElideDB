"""How good must boundaries be? Sensitivity of retrieval to segmentation error.

Before spending more on step 4 we need its TARGET. This takes truth
boundaries (perfect segmentation), degrades them to controlled quality
levels, and measures what retrieval does at each level. It answers two
questions at once:

  * the CEILING: what does the whole architecture score when
    segmentation is perfect (oracle units, everything else real)?
  * the SLOPE: how fast does that decay as boundary F1 drops, and
    therefore what F1 does step 4 have to reach?

Retrieval here is deliberately the simple version - units encoded by
the shared encoder, episodes compared by DTW over their unit-vector
sequences - so the number reflects SEGMENTATION quality, not a clever
matcher.

    python native/sensitivity.py --eps 60
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import f1_at                                     # noqa: E402


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def degrade(bs, dur, level, rng):
    """Produce boundaries at a controlled quality level.
    level 0 = exact truth; higher = more jitter, drops and spurious."""
    if level <= 0:
        return list(bs)
    out = []
    for b in bs:
        if rng.random() < level * 0.35:          # drop
            continue
        out.append(float(np.clip(b + rng.normal(0, level * 1.5),
                                 0, dur)))
    n_add = rng.poisson(level * 0.8 * max(len(bs), 1))
    out += list(rng.uniform(0, dur, n_add))
    return sorted(out)


def units_from(bs, dur, min_len=0.6):
    """Boundaries -> spans."""
    ts = [0.0] + [b for b in sorted(bs) if 0 < b < dur] + [dur]
    sp = []
    for a, b in zip(ts[:-1], ts[1:]):
        if b - a >= min_len:
            sp.append((a, b))
    return sp or [(0.0, dur)]


def dtw(A, B, band=0.4):
    n, m = len(A), len(B)
    C = 1.0 - A @ B.T
    r = max(int(band * max(n, m)), 2)
    acc = np.full((n + 1, m + 1), 1e9, np.float32)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, int(i * m / n) - r),
                       min(m, int(i * m / n) + r) + 1):
            acc[i, j] = C[i - 1, j - 1] + min(acc[i - 1, j - 1],
                                              acc[i - 1, j],
                                              acc[i, j - 1])
    return 1.0 - float(acc[n, m]) / (n + m)


def main():
    import encode as E
    import pyarrow.parquet as pq
    NEPS = arg("--eps", 60, int)
    LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5]

    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    bnd = {}
    for e, a, b in zip(t["episode"], t["t0"], t["t1"]):
        bnd.setdefault(int(e), set()).update(
            [round(float(a), 2), round(float(b), 2)])

    eps_dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                      if p.is_dir() and p.name.startswith("ep"))[:NEPS]
    print(f"decoding + encoding units for {len(eps_dirs)} episodes at "
          f"{len(LEVELS)} quality levels", flush=True)

    rng = np.random.default_rng(0)
    seqs = {lv: {} for lv in LEVELS}
    qual = {lv: [] for lv in LEVELS}
    for d in eps_dirs:
        ei = int(d.name[2:])
        cam = sorted(d.glob("cam*.mp4"))[0]
        dur = E.probe_duration(cam)
        F = E.decode(cam)
        truth = sorted(bnd.get(ei, []))
        for lv in LEVELS:
            bs = degrade(truth, dur, lv, rng)
            f, _, _ = f1_at(bs, truth, 0.05 * dur)
            qual[lv].append(f)
            sp = units_from(bs, dur)
            V = E.encode_spans(F, sp)
            seqs[lv][ei] = V
        print(f"  {d.name} dur {dur:.0f}s units " +
              " ".join(f"{lv}:{len(seqs[lv][ei])}" for lv in LEVELS),
              flush=True)

    groups = {}
    for e in seqs[0.0]:
        groups.setdefault(tmpl[e], []).append(e)
    print(f"\n{'boundary F1':<14}{'yield':<9}{'prec':<9}{'units/ep'}")
    for lv in LEVELS:
        rs = np.random.RandomState(0)
        ys, ps = [], []
        for tm, pool in sorted(groups.items()):
            if len(pool) < 6:
                continue
            sd = sorted(int(x) for x in rs.choice(pool, 5,
                                                  replace=False))
            support = len(pool) - len(sd)
            k = math.ceil(1.5 * support)
            cand = [e for e in seqs[lv] if e not in sd]
            sc = {}
            for e in cand:
                sc[e] = max(dtw(seqs[lv][s], seqs[lv][e]) for s in sd)
            got = sorted(sc, key=lambda x: -sc[x])[:k]
            tr = sum(1 for e in got if tmpl.get(e) == tm)
            ys.append(tr / support)
            ps.append(tr / len(got))
        nu = np.mean([len(v) for v in seqs[lv].values()])
        print(f"{np.mean(qual[lv]):<14.3f}{np.mean(ys):<9.3f}"
              f"{np.mean(ps):<9.3f}{nu:.1f}", flush=True)


if __name__ == "__main__":
    main()
