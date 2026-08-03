"""Chain tokens from the HOLD-STATE ALTERNATION. Geometry only.

Attempt 2 on the typing lever, following attempt 1's diagnosis
(retype_events: clustering whole-event kinematic profiles lost to the
original tokens because the ALTERNATION structure is the signal). Here
the token stream IS the alternation: the agent's hold signal - any
participant in contact - segments each episode into runs of

    E   empty hand (travel, approach)
    H   holding (carry, push)

and each segment carries corpus-quantile-quantised qualifiers:
    travel_q     horizontal path during the segment (terciles)
    height_q     how LOW the hand ends the segment, as a percentile of
                 the agent's episode-wide height (terciles) - a table
                 release ends lower than a tower release
    dur_q        segment duration (terciles)

Quantile edges are fitted from the corpus (no constants, no priors,
no text; the alphabet is positions in the corpus's own distributions).
Alignment and benchmark harness are chain_qbe's.

    python scripts/chain_holds.py [--store lake/sim_chains]
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
import chain_qbe                                               # noqa: E402

MIN_SEG_S = 0.4        # runs shorter than this are contact flicker


def hold_segments(db):
    """{episode -> [(state, travel, end_height_pct, dur_s), ...]}."""
    ep = db.table("episodes").scan().to_pydict()
    spans = sorted((int(t), int(e)) for e, t in
                   zip(ep["episode_index"], ep["ts"]))
    starts = [s for s, _ in spans]

    tr = db.table("trajectories").scan().select(
        ["stream", "ts", "px", "py", "contact", "is_agent"])
    d = tr.to_pydict()
    ag = defaultdict(list)
    touch = defaultdict(set)
    for i in range(len(d["ts"])):
        s, ts_ = str(d["stream"][i]), int(d["ts"][i])
        if d["is_agent"][i]:
            ag[s].append((ts_, float(d["px"][i]), float(d["py"][i])))
        elif d["contact"][i]:
            touch[s].add(ts_)

    import bisect
    out = {}
    for s, rows in ag.items():
        rows.sort()
        t = np.array([r[0] for r in rows], np.int64)
        x = np.array([r[1] for r in rows], np.float32)
        y = np.array([r[2] for r in rows], np.float32)
        held = np.array([tt in touch[s] for tt in t])
        # median-smooth the hold signal: single-frame flicker is not a
        # release
        k = 3
        sm = held.copy()
        for i in range(k, len(held) - k):
            sm[i] = np.median(held[i - k:i + k + 1]) > 0.5
        # split by episode, then into runs
        epi = np.array([spans[bisect.bisect_right(starts, int(tt)) - 1][1]
                        for tt in t])
        for e in np.unique(epi):
            m = epi == e
            te, xe, ye, he = t[m], x[m], y[m], sm[m]
            if len(te) < 6:
                continue
            ylo, yhi = float(ye.min()), float(ye.max())
            segs = []
            i0 = 0
            for i in range(1, len(he) + 1):
                if i == len(he) or he[i] != he[i0]:
                    dur = (te[i - 1] - te[i0]) / 1e9
                    if dur >= MIN_SEG_S:
                        travel = float(np.abs(np.diff(xe[i0:i])).sum())
                        endh = ((float(ye[i - 1]) - ylo)
                                / max(yhi - ylo, 1e-6))
                        segs.append((bool(he[i0]), travel, endh, dur))
                    i0 = i
            out[int(e)] = segs
    return out


def tokenise(segs_by_ep):
    """Quantise qualifiers on CORPUS terciles -> discrete tokens."""
    trav = np.array([s[1] for v in segs_by_ep.values() for s in v])
    dur = np.array([s[3] for v in segs_by_ep.values() for s in v])
    tq = np.percentile(trav, [33, 66])
    dq = np.percentile(dur, [33, 66])

    def q(v, edges):
        return int(np.searchsorted(edges, v))

    seqs = {}
    for e, segs in segs_by_ep.items():
        seqs[e] = [(("H" if h else "E", q(tr_, tq), q(eh * 3, [1, 2]),
                     ), 0, 0.0, None)
                   for h, tr_, eh, du in segs]
    return seqs


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    segs = hold_segments(db)
    print(f"{len(segs)} episodes, mean segments "
          f"{np.mean([len(v) for v in segs.values()]):.1f}")
    seqs = tokenise(segs)

    eps = sorted(seqs)
    pos = {e: i for i, e in enumerate(eps)}
    S = np.zeros((len(eps), len(eps)), np.float32)
    from tqdm import tqdm
    for i in tqdm(range(len(eps)), desc="align", unit="ep"):
        for j in range(i + 1, len(eps)):
            S[i, j] = S[j, i] = chain_qbe.align(seqs[eps[i]], seqs[eps[j]])

    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {}
    for e, tm in zip(t["episode"], t["template"]):
        tmpl[int(e)] = tm
    rs = np.random.RandomState(0)
    print(f"{'query':<20} {'yield':>6} {'prec':>6}   (event-kind chain y)")
    base = {"swap": 0.25, "precarious": 0.35,
            "push_then_build": 0.55, "build_unstack_move": 0.20}
    for target in ("swap", "precarious", "push_then_build",
                   "build_unstack_move"):
        pool = sorted(e for e, tm in tmpl.items() if tm == target
                      and e in pos)
        seeds = sorted(int(x) for x in rs.choice(pool, 5, replace=False))
        support = len(pool) - len(seeds)
        k = math.ceil(1.5 * support)
        si = [pos[e] for e in seeds]
        sc = S[si].max(0)
        for x_ in si:
            sc[x_] = -1e9
        got = [eps[i] for i in np.argsort(-sc)[:k]]
        true = sum(1 for e in got if tmpl.get(e) == target)
        print(f"{target:<20} {true/support:>6.2f} {true/len(got):>6.2f}   "
              f"({base[target]:.2f})")


if __name__ == "__main__":
    main()
