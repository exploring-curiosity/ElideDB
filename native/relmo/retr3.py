"""L5.2 — is the retrieval gap a BASIS problem or an INFORMATION problem?

L5.1 measured that probe-validated features retrieve at mAP 0.202 while
their probe AUCs (0.63-0.72) suggested more, and that predicting the
relational scalars and retrieving on the predictions is WORSE (0.155)
than retrieving on the raw features. The formal reason: a probe's AUC is
invariant to any invertible linear reparametrisation of the features -
it certifies only the SPAN - while cosine distance is invariant only to
orthogonal transforms plus a global scale, so it consumes the BASIS and
the PER-AXIS SCALE. Good AUC therefore never implied a good metric.

That leaves two candidate diagnoses, and they imply opposite next steps:

  BASIS        the information is present but mis-weighted. Fixable with
               a linear map - no new features, no learned objective.
  INFORMATION  the features genuinely do not contain task identity.
               Only new channels or a learned objective help.

Three tests, decisive first.

  1 FISHER TOP-K. If a 4-feature descriptor chosen by Fisher ratio beats
    the full 20-feature one, the information was there and the basis was
    burying it. This single comparison settles the question.
  2 SCENE-ONLY BASELINE. The denominator L5.1 lacked: a descriptor that
    CANNOT see the event, built from first-frame appearance and static
    points. If 0.202 does not clearly beat it, every retrieval number on
    this corpus - including the privileged bars - is partly the room.
  3 SUPERVISED WHITENING (Radenovic Lw, TPAMI 2019). L5.1 tested
    UNSUPERVISED PCA-whitening and got a null, which is the PREDICTED
    outcome: the paper reports unsupervised whitening "often hurts
    significantly" and the supervised variant wins 22 of 24 cases. The
    method with evidence behind it was untested.

WARNING ON THIS FILE'S OWN PROTOCOL, kept because it is instructive.
The within-fold CV below shrinks the gallery to a fifth, which raises the
task prior from 0.107 to 0.175 and inflates EVERY number - RANDOM scores
0.183. Worse, it reversed a headline: within-fold, a top-3 Fisher
descriptor (0.295) appeared to beat the full 20-d one (0.259), which is
the exact signature the Fisher test was built to detect. On the
full-gallery protocol the same comparison is 0.181 vs 0.202 - the
opposite conclusion. Small galleries make short descriptors look good.
THE FULL-GALLERY PROTOCOL (relmo/retr2.py: fit on TRAIN episodes, score
over all windows) IS AUTHORITATIVE; the numbers here are diagnostic only.

PROTOCOL. 5-fold grouped CV by EPISODE. Every transform - standardising
constants, Fisher ranking, Lw - is fitted on the four training folds and
applied to the held-out fold, and retrieval is scored WITHIN that fold so
query and gallery share one metric that neither has seen. Supervised arms
are labelled as such. Absolute mAP is NOT comparable to L5.1's numbers
because the gallery is a fifth of the size; all arms here are scored
identically, so the comparison between them is valid.

MAP@R is reported beside mAP and recall@5 on Musgrave et al. (ECCV 2020),
which finds R@k coarse and noisy for model selection.

    python -m relmo.retr3 --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.relg3 import F2, F2C  # noqa: E402
from relmo.retr2 import POSE, REk, TRk, collect, prep  # noqa: E402

VID = list(F2) + list(F2C) + list(POSE)          # 20 serve-legal features


def map_at_r(rel):
    """MAP@R: precision averaged over the top R, R = number of relevant.

    Musgrave, Belongie & Lim, "A Metric Learning Reality Check"
    (arXiv:2003.08505): R@1/R@5 are coarse and noisy for selection, and
    MAP@R is more informative because missing relevant items in the top
    R is penalised rather than ignored."""
    Rn = int(rel.sum())
    if Rn == 0:
        return None
    top = rel[:Rn]
    hits = np.cumsum(top)
    prec = hits / np.arange(1, Rn + 1)
    return float((prec * top).sum() / Rn)


def score_all(M, labels, eps, k=5):
    """mAP, recall@k and MAP@R, same-episode neighbours excluded."""
    S = M @ M.T
    n = len(M)
    labels = np.asarray(labels)
    ap, rec, mr = [], [], []
    for i in range(n):
        mask = np.array([eps[j] != eps[i] for j in range(n)])
        if mask.sum() < 2:
            continue
        idx = np.argsort(-S[i][mask])
        rel = (labels[mask][idx] == labels[i]).astype(float)
        if rel.sum() == 0:
            continue
        hits = np.cumsum(rel)
        prec = hits / np.arange(1, len(rel) + 1)
        ap.append(float((prec * rel).sum() / rel.sum()))
        rec.append(float(rel[:k].max()))
        v = map_at_r(rel)
        if v is not None:
            mr.append(v)
    return ap, rec, mr


def fisher(Z, y):
    """Per-feature Fisher ratio: between-class over within-class variance.

    Computed w.r.t. TASK TYPE, which is the retrieval label. L5.1's
    features were selected against RELATIONAL labels (articulated yes/no,
    contact changed yes/no) - a different question, and label mismatch is
    one of the two things that can make good probe features retrieve
    badly."""
    y = np.asarray(y)
    gm = Z.mean(0)
    bw = np.zeros(Z.shape[1])
    wi = np.zeros(Z.shape[1])
    for c in np.unique(y):
        m = y == c
        if m.sum() < 2:
            continue
        bw += m.sum() * (Z[m].mean(0) - gm) ** 2
        wi += m.sum() * Z[m].var(0)
    return bw / np.maximum(wi, 1e-12)


def lw_whiten(Z, y, eps_ids, shrink=0.25):
    """Radenovic supervised whitening (arXiv:1711.02512 / TPAMI 2019).

    C_S is the covariance of difference vectors between SAME-class pairs,
    C_D between different-class pairs. The map whitens by C_S^-1/2 and
    rotates onto the eigenvectors of C_S^-1/2 C_D C_S^-1/2, i.e. it
    shrinks the directions along which matching items disagree and
    stretches those along which non-matching items differ - exactly the
    per-axis rescaling a cosine consumes and a probe is blind to.

    Same-episode pairs are excluded from C_S: two windows of one episode
    are near-duplicates, and letting them define "same task looks like
    this" would fit the metric to self-similarity."""
    y = np.asarray(y)
    eps_ids = np.asarray(eps_ids)
    n = len(Z)
    i, j = np.triu_indices(n, 1)
    same = (y[i] == y[j]) & (eps_ids[i] != eps_ids[j])
    diff = (y[i] != y[j])
    if same.sum() < Z.shape[1] or diff.sum() < Z.shape[1]:
        return None
    D = Z[i] - Z[j]
    CS = np.cov(D[same].T)
    CD = np.cov(D[diff].T)
    # shrinkage before inversion: C_S is estimated from a few hundred
    # pairs in 20 dimensions and whitening amplifies its smallest
    # eigenvalues, which are the least reliable numbers in it
    CS = (1 - shrink) * CS + shrink * np.trace(CS) / len(CS) * np.eye(len(CS))
    w, V = np.linalg.eigh(CS)
    w = np.maximum(w, 1e-10)
    CSi = V @ np.diag(w ** -0.5) @ V.T
    _, U = np.linalg.eigh(CSi @ CD @ CSi)
    return CSi @ U[:, ::-1]


def scene_feats(z, t0, W=24):
    """A descriptor that CANNOT see the event.

    First-frame appearance statistics plus the geometry of the points
    that never move. It has the room, the camera and the furniture, and
    no information about what happens - so whatever it scores is the
    floor that any 'this retrieves events' claim must clear."""
    if "appear" not in z.files:
        return None
    A = z["appear"][t0].astype(np.float64)          # (P,8) at the FIRST frame
    vis = z["vis"][t0].astype(bool)
    if vis.sum() < 12:
        return None
    A = A[vis]
    xy = z["xy"][t0][vis].astype(np.float64)
    a, b = t0, min(t0 + W, len(z["xy"]))
    X = z["xy"][a:b].astype(np.float64)
    ok = z["vis"][a:b].astype(bool).all(0)
    stat = None
    if ok.sum() >= 12:
        Xo = X[:, ok]
        d = np.linalg.norm(Xo[-1] - Xo[0], axis=-1)
        st = d < max(np.quantile(d, 0.5), 1e-6)     # the NON-moving half
        if st.sum() >= 4:
            S = Xo[:, st]
            stat = [float(S[0][:, 0].mean() / 320.0),
                    float(S[0][:, 1].mean() / 240.0),
                    float(np.log10(np.linalg.norm(
                        S[0] - S[0].mean(0), axis=-1).mean() + 1e-6))]
    if stat is None:
        stat = [0.0, 0.0, 0.0]
    return (list(A.mean(0)) + list(A.std(0))
            + [float(xy[:, 0].mean() / 320.0), float(xy[:, 1].mean() / 240.0)]
            + stat)


def build_named(rows, names, key):
    return np.array([[r[key][k] for k in names] for r in rows], np.float64)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dataset", default="rcasa")
    ap_.add_argument("--per-ep", type=int, default=3)
    ap_.add_argument("--folds", type=int, default=5)
    ap_.add_argument("--topk", type=int, default=4)
    a = ap_.parse_args()
    rows = collect(a.dataset, a.per_ep)
    # scene features need the track file again. rows carry their own t0
    # (added to retr2.collect for this): deriving it by counting how many
    # rows an episode had contributed would silently mis-pair every row
    # after the first window that pixel_feats/comotion rejected.
    sc = []
    for r in rows:
        z = np.load(R.TRACKS / a.dataset / (r["id"] + ".npz"))
        sc.append(scene_feats(z, r["t0"]))
    ok = [i for i, x in enumerate(sc) if x is not None]
    rows = [rows[i] for i in ok]
    SC = np.array([sc[i] for i in ok], np.float64)
    tasks = np.array([r["task"] for r in rows])
    eps = np.array([r["ep"] for r in rows])
    kits = np.array([r["kitchen"] for r in rows])
    print(f"{a.dataset}: {len(rows)} windows, {len(set(eps))} episodes, "
          f"{len(set(tasks))} tasks, {len(set(kits))} kitchens")

    # ---- SCENE x TASK CONTINGENCY, before any per-scene de-biasing ----
    print("\nSCENE x TASK CONTINGENCY (checked BEFORE attempting per-scene "
          "centering)")
    per_scene = {}
    for k, t in zip(kits, tasks):
        per_scene.setdefault(k, set()).add(t)
    nper = np.array([np.sum(kits == k) for k in np.unique(kits)])
    ntask = np.array([len(v) for v in per_scene.values()])
    print(f"  windows per scene: median {np.median(nper):.0f} "
          f"max {nper.max()}   scenes with 1 task only: "
          f"{int((ntask == 1).sum())}/{len(ntask)}")
    print("  => per-scene centering is NOT attempted: with "
          f"{int((ntask == 1).sum())} of {len(ntask)} scenes containing a "
          "single task,\n     subtracting a scene mean would delete the "
          "label, not the nuisance.")

    banks = {
        "RANDOM (floor)": ("floor", np.random.default_rng(0).normal(
            size=(len(rows), 16)), False),
        "SCENE-ONLY (cannot see event)": ("floor", SC, False),
        "TRAJ (privileged)": ("priv", build_named(rows, TRk, "tr"), False),
        "REL oracle (privileged)": ("priv", build_named(rows, REk, "re"),
                                    False),
        "FULL video (20d)": ("video", np.concatenate([
            build_named(rows, F2, "px"), build_named(rows, F2C, "cm"),
            build_named(rows, POSE, "po")], 1), False),
    }
    FULL = banks["FULL video (20d)"][1]

    rng = np.random.default_rng(0)
    uid = np.unique(eps)
    fold = {e: i % a.folds for i, e in enumerate(rng.permutation(uid))}
    fi = np.array([fold[e] for e in eps])

    def run(M0, supervised=None, topk=None):
        """Grouped CV: fit on 4 folds, score retrieval inside the 5th."""
        AP, RC, MR, frac = [], [], [], []
        for k in range(a.folds):
            te, tr = fi == k, fi != k
            if te.sum() < 8:
                continue
            _, f = prep(M0[tr])
            Ztr, _ = prep(M0[tr], f)
            Zte, _ = prep(M0[te], f)
            if topk is not None:
                fr = fisher(Ztr, tasks[tr])
                sel = np.argsort(-fr)[:topk]
                v = Ztr.var(0)
                frac.append(float(v[sel].sum() / max(v.sum(), 1e-12)))
                Zte = Zte[:, sel]
                Zte = Zte / (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)
            if supervised == "lw":
                P = lw_whiten(Ztr, tasks[tr], eps[tr])
                if P is None:
                    continue
                Zte = Zte @ P
                Zte = Zte / (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)
            ap, rc, mr = score_all(Zte, tasks[te], list(eps[te]))
            AP += ap
            RC += rc
            MR += mr
        return (float(np.mean(AP)), float(np.mean(RC)), float(np.mean(MR)),
                len(AP), float(np.mean(frac)) if frac else None)

    arms = [(nm, M, sup, None) for nm, (kind, M, sup) in banks.items()]
    arms.append(("FULL video + Lw (SUPERVISED)", FULL, "lw", None))
    for kk in (3, a.topk, 6, 8):
        arms.append((f"FULL video, top-{kk} Fisher (SUPERVISED)", FULL,
                     None, kk))

    print(f"\n{'descriptor':38s} {'sup':4s} {'mAP':>6s} {'r@5':>6s} "
          f"{'MAP@R':>7s} {'n':>5s}")
    print("-" * 74)
    res = {}
    for nm, M0, sup, tk in arms:
        m, r, mr, n, fr = run(M0, sup, tk)
        s = "yes" if (sup or tk) else "no"
        res[nm] = dict(mAP=round(m, 4), recall_at_5=round(r, 4),
                       map_at_r=round(mr, 4), n=n, supervised=bool(sup or tk),
                       var_frac_topk=None if fr is None else round(fr, 4))
        print(f"{nm:38s} {s:4s} {m:6.3f} {r:6.3f} {mr:7.3f} {n:5d}")
        if fr is not None:
            print(f"{'':38s}      variance in those dims: {fr:.3f}")
    R.log("retrieval_v3", dataset=a.dataset, windows=len(rows),
          folds=a.folds, results=res)
    (R.BASE / "retrieval_v3.json").write_text(json.dumps(res, indent=1))
