"""VERIFY the sim store against simulator truth - element by element.

The sim corpus's whole value: the truthset has FULL coverage, so what
the write derived from pixels can be audited against what actually
happened. EVAL-SIDE ONLY - reads the store and the sidecars, writes
nothing to either.

Layers, from the ground up (each depends on the one before):

  A episodes     store frame counts vs the generator's frames_per_cam
  B events       time-overlap of discovered events vs truth primitives
                 (recall: truth events covered; precision: store events
                 that correspond to any truth event)
  C types        discovered transition types vs truth primitive names -
                 contingency purity ("does t3 mean stack?")
  D binding      events.object_id vs the truth mover: does one true
                 block resolve to one identity? (fragmentation, and
                 same-block-same-id consistency within an episode)
  E objkind      kind-vector separation over TRUE shape x color labels
                 (labels arrive via D's binding; cross-episode pairs
                 only, so same-instance similarity cannot flatter it)
  F channels     per-channel episode-pair AUC for same-template and
                 same-block-count - what each encoder actually sees

    python scripts/verify_sim_store.py [--store lake/sim_chains]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402
from elidedb import qbe                                        # noqa: E402


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    v = np.r_[pos, neg]
    r = v.argsort().argsort()
    return float((r[:len(pos)].mean() - (len(pos) - 1) / 2) / len(neg))


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    src = ROOT / (argv[argv.index("--src") + 1]
                  if "--src" in argv else "data/sim_chains")
    db = Store.open(str(store))

    ep = db.table("episodes").scan().to_pydict()
    n_ep = len(ep["ts"])
    ep_t0 = {int(e): int(t) for e, t in zip(ep["episode_index"], ep["ts"])}
    ep_of_ts = {int(t): int(e) for e, t in ep_t0.items()}

    metas = {}
    for i in range(n_ep):
        p = src / f"ep{i:04d}" / "meta.json"
        metas[i] = json.loads(p.read_text())
    t = pq.read_table(src / "truth.parquet").to_pydict()
    truth = defaultdict(list)
    for j in range(len(t["episode"])):
        truth[int(t["episode"][j])].append(
            {k: t[k][j] for k in ("prim", "block", "shape", "color",
                                  "ok", "t0", "t1")})

    # A ---------------------------------------------------------------
    fr = db.table("frames").scan().to_pydict()
    per_ep = Counter(int(e) for e in fr["episode_index"])
    match = sum(per_ep[i] == metas[i]["frames_per_cam"] for i in range(n_ep))
    print(f"A episodes   frames match {match}/{n_ep} "
          f"(store total {sum(per_ep.values()):,})")

    # B + C + D --------------------------------------------------------
    ev = db.table("events").scan().to_pydict()
    n_ev = len(ev["ts"])
    matched, cont = 0, Counter()
    bind_of = defaultdict(set)          # (episode, true block) -> ids
    per_event_bind = []                 # (episode, block, object_id)
    covered = defaultdict(set)
    for j in range(n_ev):
        e = ep_of_ts.get(int(ev["ts"][j]))
        if e is None:
            continue
        a = (int(ev["ev_t0"][j]) - ep_t0[e]) / 1e9
        b = (int(ev["ev_t1"][j]) - ep_t0[e]) / 1e9
        best, ov = None, 0.0
        for k, tr in enumerate(truth[e]):
            o = min(b, tr["t1"]) - max(a, tr["t0"])
            if o > ov:
                ov, best = o, k
        if best is None:
            continue
        matched += 1
        tr = truth[e][best]
        covered[e].add(best)
        if ev["kind"][j]:
            cont[(ev["kind"][j], tr["prim"])] += 1
        oid = int(ev.get("object_id", [-1] * n_ev)[j])
        if oid >= 0:
            bind_of[(e, tr["block"])].add(oid)
            per_event_bind.append((e, tr["block"], oid,
                                   tr["shape"], tr["color"]))
    n_truth = sum(len(v) for v in truth.values())
    n_cov = sum(len(v) for v in covered.values())
    print(f"B events     store events matched to a truth primitive: "
          f"{matched}/{n_ev} ({matched/n_ev:.2f}); truth primitives "
          f"covered by >=1 store event: {n_cov}/{n_truth} "
          f"({n_cov/n_truth:.2f})")

    by_type = defaultdict(Counter)
    for (k, prim), c in cont.items():
        by_type[k][prim] += c
    typed = sum(sum(c.values()) for c in by_type.values())
    pure = sum(max(c.values()) for c in by_type.values())
    print(f"C types      {len(by_type)} discovered types over {typed} "
          f"typed matches; majority-primitive purity {pure/max(typed,1):.2f}")
    for k in sorted(by_type, key=lambda x: -sum(by_type[x].values()))[:6]:
        top = by_type[k].most_common(3)
        tot = sum(by_type[k].values())
        print(f"    {k:<4} n={tot:<4} " + "  ".join(
            f"{p}:{c/tot:.2f}" for p, c in top))

    # D ----------------------------------------------------------------
    frag = [len(v) for v in bind_of.values()]
    same = ok_pairs = 0
    by_key = defaultdict(list)
    for e, blk, oid, *_ in per_event_bind:
        by_key[(e, blk)].append(oid)
    for ids in by_key.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ok_pairs += 1
                same += ids[i] == ids[j]
    print(f"D binding    bound events {len(per_event_bind)}; identities "
          f"per true block mean {np.mean(frag):.2f} (1.0 = perfect); "
          f"same-block event pairs sharing one id: "
          f"{same}/{ok_pairs} ({same/max(ok_pairs,1):.2f})")

    # E ----------------------------------------------------------------
    ok_tab = db.table("objkind_vectors").scan().to_pydict()
    lab_of = {}
    for e, blk, oid, sh, co in per_event_bind:
        lab_of.setdefault(oid, f"{sh}:{co}")
    rows = []
    spans = sorted((v, int(e)) for e, v in ep_t0.items())
    starts = [s for s, _ in spans]
    for j in range(len(ok_tab["ts"])):
        oid = int(ok_tab["object_id"][j])
        if oid not in lab_of:
            continue
        i = np.searchsorted(starts, int(ok_tab["ts"][j]), "right") - 1
        rows.append((spans[i][1], lab_of[oid],
                     np.asarray(ok_tab["vector"][j], np.float32)))
    if rows:
        rng = np.random.default_rng(0)
        V = np.stack([r[2] for r in rows])
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
        eps_ = np.array([r[0] for r in rows])
        labs = np.array([r[1] for r in rows])
        pos, neg = [], []
        for _ in range(200000):
            i, j = rng.integers(len(rows), size=2)
            if eps_[i] == eps_[j]:
                continue
            s = float(V[i] @ V[j])
            (pos if labs[i] == labs[j] else neg).append(s)
        sh = np.array([l.split(":")[0] for l in labs])
        pos_s, neg_s = [], []
        for _ in range(200000):
            i, j = rng.integers(len(rows), size=2)
            if eps_[i] == eps_[j]:
                continue
            s = float(V[i] @ V[j])
            (pos_s if sh[i] == sh[j] else neg_s).append(s)
        print(f"E objkind    {len(rows)} labelled tracks, "
              f"{len(set(labs))} true kinds; cross-episode AUC "
              f"shape+color {auc(pos, neg):.3f}, shape alone "
              f"{auc(pos_s, neg_s):.3f}")

    # F ----------------------------------------------------------------
    tmpl = {i: metas[i]["template"] for i in range(n_ep)}
    nblk = {i: len(metas[i]["blocks"]) for i in range(n_ep)}
    keys, M = qbe.spaces(db)
    kidx = [ep_of_ts.get(int(a)) for _, a in keys]
    print(f"F channels   episode-pair AUC (n={n_ep}):")
    print(f"    {'channel':<8} {'same-template':>13} {'same-n-blocks':>13}")
    for c, (A, ok, _op) in sorted(M.items()):
        idx = [i for i in range(len(keys)) if ok[i] and kidx[i] is not None]
        X = A[idx]
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
        S = X @ X.T
        pt, nt, pb, nb = [], [], [], []
        for a_ in range(len(idx)):
            for b_ in range(a_ + 1, len(idx)):
                s = float(S[a_, b_])
                ea, eb = kidx[idx[a_]], kidx[idx[b_]]
                (pt if tmpl[ea] == tmpl[eb] else nt).append(s)
                (pb if nblk[ea] == nblk[eb] else nb).append(s)
        print(f"    {c:<8} {auc(pt, nt):>13.3f} {auc(pb, nb):>13.3f}")


if __name__ == "__main__":
    main()
