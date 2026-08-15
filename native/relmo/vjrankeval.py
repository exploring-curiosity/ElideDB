"""Score a trained ranker on the whole ladder. The val number is not the result.

A model trained with LABEL supervision will look strong on the split it was
supervised on; that says almost nothing. The evidence is:

  test        held-out SCENES, same task families
  fam         families excluded from training labels entirely - unseen events
  ood_val     rcasa_eval (ArrangeTea, OpenFridge) - task families that appear
              nowhere in rcasa, so nothing about them was ever supervised

Both metrics are reported everywhere: NDCG against the graded relevance the
model was trained on, and precision@support under vjeval.group_key, which is
the metric every earlier number in this project used.

    python -m relmo.vjrankeval --tags rk_noscorer_s0,rk_noscorer_s1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrank, vjrel, vjz  # noqa: E402
from relmo.vjeval import group_key, parse  # noqa: E402
from relmo.vjood import recs_for  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402


load_ckpt = vjrank.load_ckpt


def score_matrix(model, recs, q_ids, p_ids, chunk=64, qrecs=None):
    """Stream-time traces are ragged, so encode padded + masked, and chunk the
    pool: the (Nq,Nc,T,S) similarity tensor is quadratic in trace length."""
    import torch
    def enc(ids, src):
        a, m = vjrank.pad([src[i]["a"] for i in ids], "cpu", torch)
        g, _ = vjrank.pad([src[i]["g"] for i in ids], "cpu", torch)
        s, _ = vjrank.pad([src[i]["sig"] for i in ids], "cpu", torch)
        with torch.no_grad():
            return model.encode(a, g, s, m), m
    Zq, mq = enc(q_ids, qrecs if qrecs is not None else recs)
    out = []
    for c in range(0, len(p_ids), chunk):
        Zp, mp = enc(p_ids[c:c + chunk], recs)
        with torch.no_grad():
            out.append(vjrank.tile_score(model, Zq, mq, Zp, mp, torch).numpy())
    return np.concatenate(out, 1)


def grade(S, q_ids, p_ids, meta, min_support=5):
    """k=support precision under group_key + NDCG under the graded relevance."""
    gq = np.array([meta[i]["event"] for i in q_ids])
    gp = np.array([meta[i]["event"] for i in p_ids])
    dq = np.array([meta[i].get("dur", 0.0) for i in q_ids])
    dp = np.array([meta[i].get("dur", 0.0) for i in p_ids])
    rq = np.array([meta[i]["rollout"] for i in q_ids])
    rp = np.array([meta[i]["rollout"] for i in p_ids])
    hit = sup = 0.0
    rnd = 0.0
    nd = []
    # one relevance call over the union, then index - the old per-pair loop was
    # O(N^2) calls and gave the identical matrix
    uni = list(dict.fromkeys(list(q_ids) + list(p_ids)))
    pos = {i: k for k, i in enumerate(uni)}
    full, _ = vjrel.relevance(uni, meta)
    relq = full[np.array([pos[i] for i in q_ids])[:, None],
                np.array([pos[j] for j in p_ids])[None, :]]
    for a in range(len(q_ids)):
        keep = rp != rq[a]
        ratio = (np.maximum(dp[keep], dq[a])
                 / np.maximum(np.minimum(dp[keep], dq[a]), 1e-9))
        sv = (gp[keep] == gq[a]) & vjrel.same_moment(ratio)
        k = int(sv.sum())
        if k < min_support:
            continue
        hit += int(sv[np.argsort(-S[a][keep])[:k]].sum())
        sup += k
        rnd += k * k / int(keep.sum())
        nd.append(vjrel.ndcg(S[a][keep][None, :], relq[a][keep][None, :],
                             np.ones((1, int(keep.sum())), bool)))
    return dict(prec=hit / sup if sup else float("nan"),
                chance=rnd / sup if sup else float("nan"),
                ndcg=float(np.nanmean(nd)) if nd else float("nan"),
                n=len(nd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--phase", default="rcasa_train")
    ap.add_argument("--arc", type=float, default=-1.0)
    ap.add_argument("--rec-root", default="",
                    help="override the records generation; default "
                         "is whatever each checkpoint was trained on")
    a = ap.parse_args()

    sp = load_split()
    meta = vjrel.meta_table("rcasa")
    meta.update(vjrel.meta_table("rcasa_eval"))

    # records are loaded PER ARM, under the phase setting that arm was trained
    # with. Scoring a phase-retaining model on phase-corrected records (or the
    # reverse) measures a mismatch, not the model.
    from relmo.vjmatch import ARC_DS
    arcv = ARC_DS if a.arc < 0 else a.arc
    cache = {}
    def records(name, root, arc=None):
        name = name if root != "vjrec4" else ""   # v4 records carry no `step`
        arc = arcv if arc is None else arc
        arc = 0.0 if root == "vjrec4" else arc
        if (name, root, arc) not in cache:
            ph = None
            if name:
                from relmo.vjphase import load as load_phase
                ph = load_phase(name)
            cache[(name, root, arc)] = (
                recs_for("rcasa", phase=ph, root=root, arc=arc),
                recs_for("rcasa_eval", phase=ph, root=root, arc=arc))
        return cache[(name, root, arc)]

    rows = []
    for t in [x.strip() for x in a.tags.split(",") if x.strip()]:
        model, ck = load_ckpt(t)
        held = ck.get("hold", [])
        root = a.rec_root or ck.get("rec_root", "vjrec4")
        Rin, Rood = records(ck.get("phase", a.phase), root,
                            ck.get("arc", 0.0))
        allr = dict(Rin)
        allr.update(Rood)
        val = [i for i in sorted(sp["val"]) if i in Rin]
        tst = [i for i in sorted(sp["test"]) if i in Rin]
        inpool = sorted(set(val) | set(tst))
        ood_q = sorted(Rood)
        ood_pool = sorted(set(inpool) | set(ood_q))
        r = {}
        r["val"] = grade(score_matrix(model, Rin, val, val), val, val, meta)
        r["test"] = grade(score_matrix(model, Rin, tst, inpool), tst, inpool,
                          meta)
        r["ood_val"] = grade(score_matrix(model, allr, ood_q, ood_pool),
                             ood_q, ood_pool, meta)
        if held:
            hq = [i for i in inpool if parse(i)["task"] in held]
            if len(hq) >= 5:
                r["fam"] = grade(score_matrix(model, Rin, hq, inpool), hq,
                                 inpool, meta)
        rows.append((t, held, r,
                     "" if root == "vjrec4" else ck.get("phase", a.phase)))

    cols = ["val", "test", "ood_val", "fam"]
    print(f"{'arm':22s} " + " ".join(f"{c:>22s}" for c in cols))
    print("-" * (22 + 23 * len(cols)))
    for t, held, r, phn in rows:
        cells = []
        for c in cols:
            if c not in r:
                cells.append("-")
            else:
                cells.append(f"{r[c]['prec']:.3f} / {r[c]['ndcg']:.3f}")
        print(f"{t:22s} " + " ".join(f"{c:>22s}" for c in cells))
        if held:
            print(f"{'':22s}   held out of training: {held}")
    print("\ncells are  precision@support / NDCG")
    print(f"chance (prec): " + "  ".join(
        f"{c} {rows[0][2][c]['chance']:.3f}" for c in cols if c in rows[0][2]))
    print("phase correction per arm: " + ", ".join(
        f"{t}={'ON' if phn else 'off'}" for t, _, _, phn in rows))
    R.log("vjrankeval", tags=a.tags,
          **{f"{t}_{c}": round(r[c]["prec"], 4)
             for t, _, r, _ in rows for c in r})


if __name__ == "__main__":
    main()
