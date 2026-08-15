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


def load_ckpt(tag):
    import torch
    import torch.nn as nn
    ck = torch.load(vjrank.CKPT / f"{tag}.pt", weights_only=False,
                    map_location="cpu")
    m = vjrank.build(torch, nn, ck["dim"], use_sig=ck["use_sig"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def score_matrix(model, recs, q_ids, p_ids):
    import torch
    def enc(ids):
        with torch.no_grad():
            return model.encode(
                torch.tensor(np.stack([recs[i]["a"] for i in ids])),
                torch.tensor(np.stack([recs[i]["g"] for i in ids])),
                torch.tensor(np.stack([recs[i]["sig"] for i in ids])))
    Zq, Zp = enc(q_ids), enc(p_ids)
    with torch.no_grad():
        return model.score(Zq, Zp).numpy()


def grade(S, q_ids, p_ids, meta, min_support=5):
    """k=support precision under group_key + NDCG under the graded relevance."""
    gq = np.array([group_key(parse(i)) for i in q_ids])
    gp = np.array([group_key(parse(i)) for i in p_ids])
    rq = np.array([meta[i]["rollout"] for i in q_ids])
    rp = np.array([meta[i]["rollout"] for i in p_ids])
    hit = sup = 0.0
    rnd = 0.0
    nd = []
    relq = np.zeros((len(q_ids), len(p_ids)), np.float32)
    for a, i in enumerate(q_ids):
        for b, j in enumerate(p_ids):
            relq[a, b] = vjrel.relevance([i, j], meta)[0][0, 1]
    for a in range(len(q_ids)):
        keep = rp != rq[a]
        sv = gp[keep] == gq[a]
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
    a = ap.parse_args()

    sp = load_split()
    Rin, Rood = recs_for("rcasa"), recs_for("rcasa_eval")
    meta = vjrel.meta_table("rcasa")
    meta.update(vjrel.meta_table("rcasa_eval"))
    allr = dict(Rin)
    allr.update(Rood)

    val = [i for i in sorted(sp["val"]) if i in Rin]
    tst = [i for i in sorted(sp["test"]) if i in Rin]
    inpool = sorted(set(val) | set(tst))
    ood_q = sorted(Rood)
    ood_pool = sorted(set(inpool) | set(ood_q))

    rows = []
    for t in [x.strip() for x in a.tags.split(",") if x.strip()]:
        model, ck = load_ckpt(t)
        held = ck.get("hold", [])
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
        rows.append((t, held, r))

    cols = ["val", "test", "ood_val", "fam"]
    print(f"{'arm':22s} " + " ".join(f"{c:>22s}" for c in cols))
    print("-" * (22 + 23 * len(cols)))
    for t, held, r in rows:
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
    R.log("vjrankeval", tags=a.tags,
          **{f"{t}_{c}": round(r[c]["prec"], 4)
             for t, _, r in rows for c in r})


if __name__ == "__main__":
    main()
