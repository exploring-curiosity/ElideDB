"""The debugged failure: I threw away the signal that separates classes.

Debugging the 0.492 ceiling rather than swapping another encoder found
this. The six sim templates are near-identical as action sequences -
every pair has similarity >= 0.57, and build_unstack_move is
relocate_build plus ONE unstack (edit distance 1). What actually
separates them is HOW MANY events occur:

    push_then_build   5.00 +- 0.00 events
    relocate_build    6.00 +- 0.00
    swap              6.00 +- 0.00
    build_unstack_move7.00 +- 0.00
    two_sites_merge   7.00 +- 0.00
    precarious        8.00 +- 0.00

Deterministic. Count alone narrows six templates to at most two.

And the pipeline destroyed that signal in THREE places, all mine:

  1. DTW's entire purpose is invariance to timing and repetition, so a
     6-event episode warps onto an 8-event one at little cost.
  2. Uniform windows impose a fixed grid - a 20 s and a 32 s episode
     both become "one window per second", carrying no event count.
  3. match7.py L2-normalised its symbol histograms, which removes
     magnitude - and magnitude WAS the count.

It also explains the cliff in sens2.py at last: only perfect boundaries
help because only perfect boundaries give the right COUNT. Boundary F1
0.867 already miscounts, and miscounting costs the whole signal.

So the fix is not a better encoder. It is to stop discarding count.

    python native/countmatch.py --limit 150
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

from flowgebd import arg                                       # noqa: E402
from sensitivity import dtw                                    # noqa: E402
from u6retr import units                                       # noqa: E402
from unitenc import FPS                                        # noqa: E402


def episode_score(A, B, dA, dB, w_dtw, w_len):
    """Content similarity AND scale agreement.

    DTW gives content while being deliberately blind to length; the
    length term restores exactly what DTW removes. Kept as a weighted
    sum of two z-free quantities in [0,1] so neither can dominate by
    unit choice.
    """
    s = dtw(A, B) if w_dtw > 0 else 0.0
    # symmetric relative length agreement: 1.0 when identical
    lo, hi = (dA, dB) if dA <= dB else (dB, dA)
    lsim = lo / hi if hi > 0 else 0.0
    return w_dtw * s + w_len * lsim


def score_cfg(seqs, durs, tmpl, w_dtw, w_len, seed_n=5, abstain=True):
    groups = {}
    for e in seqs:
        groups.setdefault(tmpl[e], []).append(e)
    rs = np.random.RandomState(0)
    ys, ps, sups, rets = [], [], [], []
    for tm, pool in sorted(groups.items()):
        if len(pool) < seed_n + 1:
            continue
        sd = sorted(int(x) for x in rs.choice(pool, seed_n,
                                              replace=False))
        support = len(pool) - len(sd)
        k = math.ceil(1.5 * support)
        cand = [e for e in seqs if e not in sd]
        sc = {e: max(episode_score(seqs[s], seqs[e], durs[s], durs[e],
                                   w_dtw, w_len) for s in sd)
              for e in cand}
        loo = [max(episode_score(seqs[a], seqs[b], durs[a], durs[b],
                                 w_dtw, w_len)
                   for b in sd if b != a) for a in sd]
        cut = min(loo) if loo else -np.inf
        ranked = sorted(sc, key=lambda x: -sc[x])[:k]
        got = [e for e in ranked if sc[e] >= cut] if abstain else ranked
        tr = sum(1 for e in got if tmpl.get(e) == tm)
        ys.append(tr / support)
        ps.append(tr / len(got) if got else 0.0)
        sups.append(support)
        rets.append(len(got))
    return (float(np.mean(ys)), float(np.mean(ps)),
            float(np.mean(sups)), float(np.mean(rets)))


def main():
    import encode as E
    import pyarrow.parquet as pq
    from tqdm import tqdm
    limit = arg("--limit", 150, int)
    enc = arg("--enc", "siglip2_rank")
    seg = arg("--seg", "uni:3:1.0")

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
    seqs, durs = {}, {}
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
        durs[ei] = dur

    print(f"\n{enc} / {seg}, {len(seqs)} episodes")
    print(f"{'w_dtw':<8}{'w_len':<8}{'yield':<9}{'prec':<9}"
          f"{'support':<9}{'returned'}")
    for wd, wl in ((1.0, 0.0), (0.0, 1.0), (1.0, 0.3), (1.0, 0.6),
                   (1.0, 1.0), (1.0, 1.5), (1.0, 2.5), (0.5, 1.0)):
        y, p, su, rt = score_cfg(seqs, durs, tmpl, wd, wl)
        tag = "  <- DTW only (today's 0.492)" if wl == 0 else (
            "  <- length only" if wd == 0 else "")
        print(f"{wd:<8.1f}{wl:<8.1f}{y:<9.3f}{p:<9.3f}{su:<9.1f}"
              f"{rt:<9.1f}{tag}", flush=True)


if __name__ == "__main__":
    main()
