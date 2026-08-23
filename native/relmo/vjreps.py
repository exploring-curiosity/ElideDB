"""Score MANY representations of one corpus under one matcher.

The iteration harness. Stage E asks "is the new z better"; this asks the
sharper question "better than what, and does it still need the frozen
channels" - because the read-side wins in EXPERIMENTS.md §3 (whitening,
frozen fusion) compose with ANY z and must be part of every comparison, not
a separate idea someone remembers later.

Whitening is fitted on the TARGET corpus with no labels: it is a property of
the corpus being searched, not of any training set, which is why it transfers
to corpora that did not exist when the model was trained.

    python3 -m relmo.vjreps --ckpt ssl_v1_s0 --dataset rcasa_composite_full \
        --queries 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrel, vjrank2  # noqa: E402
from relmo.vjrel import parse  # noqa: E402
from relmo.vjtest import score  # noqa: E402


def zs(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)


def fit_whiten(X, k, eps=1e-3):
    mu = X.mean(0, keepdims=True)
    C = np.cov((X - mu).T)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(lam)[::-1][:k]
    return mu, (V[:, o] / np.sqrt(lam[o] + eps)).astype(np.float32)


def apply_w(X, w):
    # Apple's Accelerate BLAS raises spurious divide/overflow/invalid status
    # flags on this sgemm (inputs and outputs verified finite, |w| ~ 1), which
    # numpy then reports as RuntimeWarnings on every query. Silence the flag,
    # and make a REAL non-finite result loud instead of a warning.
    with np.errstate(all="ignore"):
        Y = (X - w[0]) @ w[1]
    if not np.isfinite(Y).all():
        raise FloatingPointError("whitening produced non-finite values")
    return Y


def participation(X, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > n:
        X = X[rng.choice(len(X), n, replace=False)]
    Xc = (X - X.mean(0, keepdims=True)).astype(np.float64)
    lam = np.linalg.eigvalsh((Xc.T @ Xc) / len(Xc))[::-1].clip(0)
    return float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-12))


def build_reps(Z, D, have, k_white=256):
    """-> {name: {id: (T,d)}}. Every entry is label-free."""
    allz = np.concatenate([Z[i] for i in have])
    allf = np.concatenate([D[i]["fix"] for i in have]).astype(np.float32)
    alls = np.concatenate([D[i]["sig"] for i in have]).astype(np.float32)
    kz = min(k_white, Z[have[0]].shape[1])
    wz, wf, ws = (fit_whiten(allz, kz), fit_whiten(allf, k_white),
                  fit_whiten(alls, k_white))
    reps = {
        "z": {i: Z[i] for i in have},
        "z whitened": {i: apply_w(Z[i], wz) for i in have},
        "frozen fix+sig whitened": {
            i: np.concatenate([apply_w(D[i]["fix"].astype(np.float32), wf),
                               apply_w(D[i]["sig"].astype(np.float32), ws)], -1)
            for i in have},
        # THE CONTROL. The frozen baseline is whitened+512d while the fusion
        # is z-scored+2048d, so a fusion win could be the normalisation and
        # the kept dimensions rather than z. This row holds everything fixed
        # except z's presence.
        "fix+sig zscored NOz": {
            i: np.concatenate([zs(D[i]["fix"].astype(np.float32)),
                               zs(D[i]["sig"].astype(np.float32))], -1)
            for i in have},
        # NEW CHANNEL: spatial motion signature, free, label-free, never used
        "fix+sig+wmap zscored": {
            i: np.concatenate([zs(D[i]["fix"].astype(np.float32)),
                               zs(D[i]["sig"].astype(np.float32)),
                               zs(D[i]["wmap"].astype(np.float32))], -1)
            for i in have},
        "fix+wmap zscored": {
            i: np.concatenate([zs(D[i]["fix"].astype(np.float32)),
                               zs(D[i]["wmap"].astype(np.float32))], -1)
            for i in have},
        "z + fix + sig (z-scored)": {
            i: np.concatenate([zs(Z[i]), zs(D[i]["fix"].astype(np.float32)),
                               zs(D[i]["sig"].astype(np.float32))], -1)
            for i in have},
        "z + fix + sig WHITENED": {
            i: np.concatenate([apply_w(Z[i], wz),
                               apply_w(D[i]["fix"].astype(np.float32), wf),
                               apply_w(D[i]["sig"].astype(np.float32), ws)], -1)
            for i in have},
    }
    return reps, dict(pr_z=participation(allz), pr_fix=participation(allf),
                      pr_sig=participation(alls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="head tag, or 'none' for "
                                                  "frozen-only comparison")
    ap.add_argument("--dataset", default="rcasa_composite_full")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--breakdown", default="rcasa")
    ap.add_argument("--event-key", default="group", choices=["group", "task"],
                    help="'group' = vjeval.group_key (verb x object-kind). On "
                         "rcasa_composite_full that parser cannot read "
                         "compositional names and dumps 900 of 1152 records "
                         "into one 'Other/Other' bucket, which drives chance "
                         "to 0.543 and makes prec@support unable to "
                         "discriminate. 'task' uses task identity - 32 groups "
                         "of 36, chance 0.031 - and is the valid key for that "
                         "corpus. Grading only; never reaches retrieval.")
    ap.add_argument("--only", default="", help="comma-separated rep names")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    AM = vjrel.all_meta(("rcasa", "rcasa_eval", "rcasa_atomic_full",
                         a.dataset))
    if a.event_key == "task":
        for i in list(AM):
            AM[i] = dict(AM[i], event=parse(i)["task"])
    seen = None
    if a.breakdown:
        seen = {parse(e["id"])["task"]
                for e in R.read_manifest(a.breakdown)["episodes"]}
    D = vjrank2.load_corpus(datasets=(a.dataset,))
    have = sorted(set(D) & set(AM))
    print(f"{a.dataset}: {len(have)} graded records, median "
          f"{int(np.median([len(D[i]['tok']) for i in have]))} steps",
          flush=True)

    m, ck = vjrank2.load_ckpt(a.ckpt)
    print(f"head {a.ckpt}: d={ck['dim']} kind={ck.get('kind','discriminative')}"
          f" params={ck.get('params',0)/1e6:.2f}M", flush=True)
    Z = vjrank2.encode_all(m, D, ids=have)

    reps, prs = build_reps(Z, D, have)
    print(f"PR on this corpus: z {prs['pr_z']:.1f}  fix {prs['pr_fix']:.1f}  "
          f"sig {prs['pr_sig']:.1f}\n", flush=True)
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        reps = {k: v for k, v in reps.items() if k in keep}

    print(f"{'representation':28s} {'d':>5s} {'ALL':>7s} {'seen':>7s} "
          f"{'UNSEEN':>7s} {'chance':>7s}")
    print("-" * 68)
    out = {}
    for name, Zr in reps.items():
        t0 = time.time()
        rng = np.random.default_rng(a.seed)          # same query sample
        r = score(Zr, have, have, AM, rng, a.queries)
        cells = []
        for want in (True, False):
            v = [p for i, (p, _) in r["per_query"].items()
                 if seen is None or (parse(i)["task"] in seen) is want]
            cells.append(float(np.mean(v)) if v else float("nan"))
        d = Zr[have[0]].shape[1]
        print(f"{name:28s} {d:5d} {r['prec']:7.3f} {cells[0]:7.3f} "
              f"{cells[1]:7.3f} {r['chance']:7.3f}   [{time.time()-t0:.0f}s]",
              flush=True)
        out[name] = dict(prec=r["prec"], seen=cells[0], unseen=cells[1],
                         chance=r["chance"], ci95=r["ci95"], dim=d,
                         queries=r["queries"], pool=r["pool"])
    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(ckpt=a.ckpt, dataset=a.dataset, pr=prs, reps=out), indent=1))
        print(f"\nVERIFIED: wrote {a.out}")
    best = max(out, key=lambda k: out[k]["prec"])
    R.log("vjreps", ckpt=a.ckpt, dataset=a.dataset, pr_z=round(prs["pr_z"], 1),
          best=best, best_prec=round(out[best]["prec"], 4))


if __name__ == "__main__":
    main()
