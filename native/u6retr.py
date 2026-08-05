"""STEP 6, end to end: does better unit encoding move the PRODUCT metric?

unitenc.py measures unit quality as an AUC. That is a proxy, and this
project has been burned by proxies before (boundary F1 went up while
span F1 went to zero). So this closes the loop: same episodes, same
retrieval, only the unit encoder changes, scored in yield and precision
at k = ceil(1.5 x support) with k a MAX BOUND.

Why this is the right test for step 6 specifically: in this corpus the
ACTION sequence identifies the template uniquely (6 prim-sequences ->
6 templates, 0 ambiguous episodes out of 150). So an encoder that
represents the action should retrieve, and one that does not cannot.
Colour is irrelevant here by construction, which is why unitenc.py's
colour AUC sitting at chance is not a defect to chase.

Two segmentations are reported:
  truth  oracle spans - isolates the ENCODER
  cpd    native/cpd.py spans - what the system actually gets

Unit vectors are cached per (encoder, episode) so an interrupt costs
one encoder, not the run.

    python native/u6retr.py --limit 72        # ~20 min
"""
from __future__ import annotations

import hashlib
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
from unitenc import ENCODERS, FPS                              # noqa: E402

CACHE = Path("/private/tmp/claude-501/u6retr")


def units(enc, tag, ep, F, spans):
    key = hashlib.sha1(
        f"{enc}|{tag}|{ep}|{len(spans)}|{spans[:3]}"
        .encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{key}.npy"
    if fp.exists():
        return np.load(fp)
    V = ENCODERS[enc](F, spans, 0.0)
    np.save(fp, V.astype(np.float32))
    return V


def score(seqs, tmpl, seed_n=5):
    """yield = true/support, prec = true/returned, k = 1.5 x support.

    Returns (yield, prec, mean_support). SUPPORT IS PART OF THE RESULT:
    with few episodes per template the support falls to 1-2 and yield
    quantises to {0, 0.5, 1}, which produced a non-monotonic
    "sensitivity curve" that was pure noise. Never read a yield without
    its support.
    """
    groups = {}
    for e in seqs:
        groups.setdefault(tmpl[e], []).append(e)
    rs = np.random.RandomState(0)
    ys, ps, sups = [], [], []
    for tm, pool in sorted(groups.items()):
        if len(pool) < seed_n + 1:
            continue
        sd = sorted(int(x) for x in rs.choice(pool, seed_n,
                                              replace=False))
        support = len(pool) - len(sd)
        k = math.ceil(1.5 * support)
        cand = [e for e in seqs if e not in sd]
        sc = {e: max(dtw(seqs[s], seqs[e]) for s in sd) for e in cand}
        got = sorted(sc, key=lambda x: -sc[x])[:k]
        tr = sum(1 for e in got if tmpl.get(e) == tm)
        ys.append(tr / support)
        ps.append(tr / max(len(got), 1))
        sups.append(support)
    return (float(np.mean(ys)), float(np.mean(ps)),
            float(np.mean(sups)) if sups else 0.0)


def main():
    import encode as E
    import pyarrow.parquet as pq
    from cpd import fit
    from graphgebd import frame_features
    from segment import spans as to_spans
    from tqdm import tqdm

    limit = arg("--limit", 72, int)
    names = arg("--enc", "vjepa2,siglip2_rank,siglip2_delta,"
                         "r50_rank").split(",")
    segs = arg("--seg", "truth,cpd").split(",")

    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl, ev = {}, {}
    for e, tm, a, b in zip(t["episode"], t["template"], t["t0"],
                           t["t1"]):
        tmpl[int(e)] = tm
        ev.setdefault(int(e), []).append((float(a), float(b)))

    dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                  if p.is_dir() and p.name.startswith("ep"))[:limit]
    print(f"STEP 6 end-to-end — {len(dirs)} episodes, "
          f"{len(names)} encoders, segmentations {segs}", flush=True)

    def uniform(dur, w, st):
        """Dense OVERLAPPING windows - not a partition.

        The sensitivity ladder says yield only benefits from boundaries
        that are essentially PERFECT (F1 1.000 -> 0.492, everything
        below 0.867 flat at ~0.28). Real segmentation never reaches
        that, so chasing step 5 is chasing a cliff we cannot climb.
        Overlapping windows sidestep it: if a straddling span is what
        corrupts a direction vector, then at stride < event length SOME
        window always lies inside one event. Coverage is still total,
        which was the only reason a partition was required.
        """
        out, t = [], 0.0
        while t + w <= dur + 1e-6:
            out.append((t, t + w))
            t += st
        return out or [(0.0, dur)]

    media, sp_truth, sp_cpd = {}, {}, {}
    sp_uni = {}
    for d in tqdm(dirs, desc="decode+segment", unit="ep"):
        ei = int(d.name[2:])
        if ei not in ev:
            continue
        cam = sorted(d.glob("cam*.mp4"))[0]
        dur = E.probe_duration(cam)
        F = E.decode(cam, fps=FPS, w=256)
        media[ei] = F
        sp_truth[ei] = sorted(ev[ei])
        for sg in segs:
            if sg.startswith("uni:"):
                _, w, st = sg.split(":")
                sp_uni.setdefault(sg, {})[ei] = uniform(
                    dur, float(w), float(st))
        if "cpd" in segs:
            CACHE.mkdir(parents=True, exist_ok=True)
            fp = CACHE / f"cpd_{ei}_{len(F)}.npy"
            if fp.exists():
                sp_cpd[ei] = [tuple(r) for r in np.load(fp)]
            else:
                bk = fit(frame_features(F), kind="slope", model="rbf")
                sp_cpd[ei] = to_spans([b / FPS for b in bk], dur)
                np.save(fp, np.array(sp_cpd[ei], np.float32))

    print(f"\n{'encoder':<15}{'seg':<8}{'yield':<9}{'prec':<9}"
          f"{'units/ep':<10}{'support'}")
    for nm in names:
        for sg in segs:
            SP = (sp_truth if sg == "truth"
                  else sp_cpd if sg == "cpd" else sp_uni[sg])
            seqs = {}
            for ei, F in tqdm(media.items(), desc=f"{nm}/{sg}",
                              unit="ep", leave=False):
                seqs[ei] = units(nm, sg, ei, F, SP[ei])
            y, p, su = score(seqs, tmpl)
            nu = np.mean([len(v) for v in seqs.values()])
            print(f"{nm:<15}{sg:<8}{y:<9.3f}{p:<9.3f}{nu:<10.1f}"
                  f"{su:.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
