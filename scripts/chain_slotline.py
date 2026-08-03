"""BOUNDARY-FREE chain tokens: the slot-activity timeline.

The decidability panel closed the unit-pairing question: whether a
blind gap is one carry or two manipulations is not decidable from
trajectory elements (best signal 0.65 over seven candidates). So this
representation never decides it. An episode is a TIMELINE - per
half-second bin, WHICH COLOUR-SLOT is in motion, or silence:

    A A A _ A A _ _ B B B _ _ A A

A carry's blind hole stays a hole; alignment absorbs an inserted
silence gracefully where a wrong merge/split scrambled whole
sequences. Fragmentation, the 0.59-of-yield wall, is dissolved rather
than solved.

The enabling piece is AGENT-COLOUR SUPPRESSION: in-motion crops are
half gripper (measured: colour cut ballooned 58->90 Lab), but the
agent's own colour is fitted from its own crops, and a per-crop Otsu
over pixel distances to it splits gripper pixels from block pixels -
the block's colour survives mid-carry. Known hole, stated: a block
the same colour as the gripper vanishes to this method.

Everything fitted from the corpus: motion thresholds and participant
filters are chain_2v's; the slot cut is Otsu over within-episode bin-
colour distances; the agent colour is measured, not assumed. No
models, no text, no truth.

    python scripts/chain_slotline.py [--store lake/sim_chains]
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
from chain_moves import bench, otsu                            # noqa: E402
from chain_2v import Cropper, view_segments                    # noqa: E402

BIN_S = 0.5


def agent_colour(db, agents, cr, n=120):
    """The arm's own Lab colour, measured from its own crops."""
    rng = np.random.default_rng(0)
    keys = sorted(agents)
    vals = []
    while len(vals) < n and keys:
        e, sv = keys[rng.integers(len(keys))]
        rows = agents[(e, sv)]
        ts_, x0, y0, x1, y1 = rows[rng.integers(len(rows))]
        v = cr.lab(sv, ((x0 + x1) / 2, (y0 + y1) / 2), ts_)
        if v is not None:
            vals.append(v)
    lab = np.median(np.stack(vals), 0).astype(np.float32)
    print(f"agent colour: Lab {np.round(lab, 0)} ({len(vals)} crops)")
    return lab


def suppressed_lab(cr, sv, pos, ts_, agent_lab):
    """Median Lab of the crop EXCLUDING agent-coloured pixels: per-crop
    Otsu over pixel distances to the agent colour splits gripper from
    block. None when almost nothing survives (agent-coloured object)."""
    import cv2
    im = cr.frame(sv, ts_)
    if im is None:
        return None
    h = int(cr.med_diag * 0.35)
    x0 = max(int(pos[0]) - h, 0)
    y0 = max(int(pos[1]) - h, 0)
    c = im[y0:int(pos[1]) + h, x0:int(pos[0]) + h]
    if c.size < 48:
        return None
    lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB).reshape(-1, 3) \
        .astype(np.float32)
    d = np.linalg.norm(lab - agent_lab[None, :], axis=1)
    if d.max() - d.min() < 15:            # crop is all one thing
        keep = d > np.inf
    else:
        keep = d > otsu(d)
    if keep.mean() < 0.15:
        return None
    return np.median(lab[keep], 0).astype(np.float32)


def timelines(db):
    """{episode -> [(slot_lab|None per bin)]} + helpers."""
    segs, series, med_diag, thr, agents = view_segments(db)
    ep = db.table("episodes").scan().to_pydict()
    spans = {int(e): int(t) for e, t in
             zip(ep["episode_index"], ep["ts"])}
    dur = {int(e): int(b) - int(a) for e, a, b in
           zip(ep["episode_index"], ep["ts"], ep["t1"])}
    cr = Cropper(db, med_diag)
    a_lab = agent_colour(db, agents, cr)

    # strongest moving sample per (episode, bin, view)
    strongest = {}
    for (e, sv, oid), (t, x, y, nd) in series.items():
        t0 = spans[e]
        for i in range(len(t)):
            if nd[i] < thr:
                continue
            b = int((int(t[i]) - t0) / (BIN_S * 1e9))
            k = (e, b, sv)
            if k not in strongest or nd[i] > strongest[k][0]:
                strongest[k] = (float(nd[i]),
                                (float(x[i]), float(y[i])), int(t[i]))
    # bin colours, agent-suppressed, averaged over views
    from tqdm import tqdm
    bin_lab = {}
    for (e, b, sv), (_, pos, ts_) in tqdm(sorted(strongest.items()),
                                          desc="bins", unit="crop",
                                          mininterval=5):
        v = suppressed_lab(cr, sv, pos, ts_, a_lab)
        if v is None:
            continue
        k = (e, b)
        bin_lab.setdefault(k, []).append(v)
    lines = {}
    for e in spans:
        nb = int(dur[e] / (BIN_S * 1e9)) + 1
        lines[e] = [None] * nb
        for b in range(nb):
            vs = bin_lab.get((e, b))
            if vs:
                lines[e][b] = np.mean(np.stack(vs), 0)
    return lines


def tokenise(lines):
    """Slot clustering (fitted cut) + run-length tokens."""
    dists = []
    for e, bins in lines.items():
        vs = [v for v in bins if v is not None]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                dists.append(float(np.linalg.norm(vs[i] - vs[j])))
    cut = otsu(np.array(dists)) if dists else 30.0
    print(f"slot cut: {cut:.1f} Lab (otsu, {len(dists):,} bin pairs)")
    seqs = {}
    for e, bins in lines.items():
        reps, ids = [], []
        for v in bins:
            if v is None:
                ids.append(-1)
                continue
            hit = None
            for lab0, sid in reps:
                if float(np.linalg.norm(v - lab0)) <= cut:
                    hit = sid
                    break
            if hit is None:
                hit = len(reps)
                reps.append((v, hit))
            ids.append(hit)
        # run-length encode: (slot run) and (silence run) tokens with
        # duration terciles filled in later via a second pass
        toks = []
        i0 = 0
        for i in range(1, len(ids) + 1):
            if i == len(ids) or ids[i] != ids[i0]:
                run = (i - i0) * BIN_S
                if ids[i0] >= 0:
                    toks.append(["M", ids[i0], run])
                else:
                    toks.append(["G", -1, run])
                i0 = i
        seqs[e] = toks
    durs = np.array([t[2] for v in seqs.values() for t in v])
    dq = np.percentile(durs, [33, 66])
    out = {}
    for e, toks in seqs.items():
        out[e] = [((k, int(np.searchsorted(dq, d)), 0),
                   s, d, None) for k, s, d in toks]
    return out


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    lines = timelines(db)
    known = np.mean([np.mean([v is not None for v in bins])
                     for bins in lines.values()])
    print(f"episodes {len(lines)}, coloured-bin fraction {known:.2f}")
    seqs = tokenise(lines)
    print(f"mean tokens/episode "
          f"{np.mean([len(v) for v in seqs.values()]):.1f}")
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {}
    for e, tm in zip(t["episode"], t["template"]):
        tmpl[int(e)] = tm
    chain_qbe.W_KIND = 0.3
    dev = ("swap", "precarious", "push_then_build",
           "build_unstack_move")
    hold = ("relocate_build", "two_sites_merge")
    bench(seqs, tmpl, "DEV slotline", dev, w_slot=1.0)
    bench(seqs, tmpl, "HOLDOUT slotline", hold, w_slot=1.0)


if __name__ == "__main__":
    main()
