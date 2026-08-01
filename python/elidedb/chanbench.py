"""How to judge a channel — teacher or student, same yardstick.

Params and ms/clip say nothing about whether a representation works.
Neither does agreement with the teacher: the V-JEPA2 run made that
concrete, where the teacher's own clips sat at 0.977 mean cosine to
their centroid, so a student agreeing with it perfectly would inherit a
representation that cannot rank anything. Fidelity to a broken teacher
is a broken student with extra steps.

Three layers, and a channel has to pass them IN ORDER. Each one is a
different question, and a failure at any level makes the levels below
it unreadable.

  1 SEPARABILITY   does the representation distinguish anything at all?
                   mean pairwise cosine and effective rank. Catches the
                   collapse that both FDNN-V (0.978) and mean-pooled
                   V-JEPA2 (0.977) were hiding behind respectable
                   cosine numbers. Costs nothing, run it FIRST, on the
                   teacher, before training any student against it.

  2 TASK AUC       is it the RIGHT signal? A linear probe over labels
                   the WRITE PATH already produced - event kinds from
                   geometry, object ids from the identity store - so
                   this measures usefulness without ever touching the
                   truthset. That matters: the truthset is eval-only,
                   and a per-iteration metric that reads it would make
                   every later evaluation meaningless.

  3 FIDELITY       does the student rank like the teacher? Only worth
                   asking once 1 and 2 say the teacher is worth
                   matching.

THE SHIPPING GATE is layer 2, not layer 3: a student ships when its task
AUC is within tolerance of its teacher's, at a fraction of the cost. A
student can legitimately disagree with its teacher clip-by-clip and
still be as useful, and that is a pass, not a failure.

The product metric - yield and precision on the truthset at
k = ceil(1.5 x support) - stays the integration test, run once channels
are assembled. It is too coarse and too slow to steer a per-channel
loop, and it is the thing these layers exist to protect.
"""
from __future__ import annotations

import numpy as np


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def separability(V, sample=2000):
    """Layer 1. Does the space have room to rank anything?

    effective_rank is the participation ratio of the covariance
    eigenvalues - how many dimensions the data actually uses. A 1024-d
    embedding with an effective rank of 3 is a 3-d embedding that costs
    1024 floats to store and cannot support a nearest-neighbour query.
    """
    V = _l2(np.asarray(V, np.float32))
    n = min(len(V), sample)
    S = V[:n] @ V[:n].T
    iu = np.triu_indices(n, 1)
    mu = _l2(V.mean(0))
    ev = np.linalg.eigvalsh(np.cov(V[:n].T))[::-1]
    ev = np.clip(ev, 0, None)
    p = ev / (ev.sum() + 1e-12)
    eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return {"mean_pairwise_cos": round(float(S[iu].mean()), 4),
            "p95_pairwise_cos": round(float(np.percentile(S[iu], 95)), 4),
            "mean_baseline": round(float(np.mean(V @ mu)), 4),
            "effective_rank": round(eff, 1),
            "dims": int(V.shape[1])}


def probe_auc(V, y, folds=5, seed=0):
    """Layer 2. Linear-probe AUC for one binary label.

    A LINEAR probe on purpose: it asks whether the information is
    present and readable, not whether a big enough head can dig it out.
    That is the property a retrieval channel needs, because the thing
    consuming it downstream is a dot product.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    V, y = _l2(np.asarray(V, np.float32)), np.asarray(y).astype(int)
    if y.sum() < folds or (1 - y).sum() < folds:
        return None                      # too few of one class to score
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(V))
    V, y = V[idx], y[idx]
    out = []
    for f in range(folds):
        te = np.zeros(len(V), bool)
        te[f::folds] = True
        if y[~te].sum() == 0 or y[~te].sum() == (~te).sum():
            continue
        if len(set(y[te])) < 2:
            continue
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(V[~te], y[~te])
        out.append(roc_auc_score(y[te], m.decision_function(V[te])))
    return round(float(np.mean(out)), 4) if out else None


def retrieval_map(V, groups):
    """Layer 2, the ranking form. Mean average precision when the query
    is a clip and the relevant set is everything sharing its group.

    AUC says the information is linearly readable; mAP says it survives
    being turned into a ranking, which is what the channel is for.
    """
    V = _l2(np.asarray(V, np.float32))
    g = np.asarray(groups)
    S = V @ V.T
    np.fill_diagonal(S, -9)
    aps = []
    for i in range(len(V)):
        rel = (g == g[i])
        rel[i] = False
        if not rel.any():
            continue
        order = np.argsort(-S[i])
        hit = rel[order]
        cum = np.cumsum(hit)
        prec = cum / (np.arange(len(hit)) + 1)
        aps.append(float((prec * hit).sum() / hit.sum()))
    return round(float(np.mean(aps)), 4) if aps else None


def fidelity(P, Y, k=10):
    """Layer 3. Does the student rank like the teacher?"""
    P, Y = _l2(np.asarray(P, np.float32)), _l2(np.asarray(Y, np.float32))
    n = min(len(P), 400)
    Sp, St = P[:n] @ P[:n].T, Y[:n] @ Y[:n].T
    np.fill_diagonal(Sp, -9); np.fill_diagonal(St, -9)
    kk = min(k, n - 1)
    rp, rt = np.argsort(-Sp, 1)[:, :kk], np.argsort(-St, 1)[:, :kk]
    return {"cosine_to_teacher": round(float(np.mean(np.sum(P * Y, 1))), 4),
            "nn_top1": round(float(np.mean(Sp.argmax(1) == St.argmax(1))), 4),
            f"nn_recall@{kk}": round(float(np.mean(
                [len(set(a) & set(b)) / kk for a, b in zip(rp, rt)])), 4)}


def code_agreement(student, teacher, C, min_frac=0.05):
    """Layer 3b: does the student land in the TEACHER'S codebook cell?

    This metric exists because of how this store prunes. `code` is a
    clustered column and the planner reads only the row groups a query's
    probe selects, so a student that produces a beautiful vector in the
    WRONG cell puts its row in a row group the planner never opens. The
    row is then unreachable at any k. Cosine cannot see that failure and
    neither can nn_recall.

    Two numbers, and the second is the one that decides:

      code@1        student cell == teacher cell. Informative, but a
                    miss here is survivable - probe() widens.
      code_recall   the teacher's cell is INSIDE the probe set the
                    student's own vector selects. A miss here is
                    permanent data loss: nothing the reader does at
                    query time recovers that row.

    A student at code@1 0.70 with code_recall 0.99 ships. One at code@1
    0.95 with code_recall 0.95 silently loses 5% of the corpus.
    """
    from .teacher import assign, probe as _probe
    C = np.asarray(C, np.float32)
    ts_code, _ = assign(np.asarray(teacher, np.float32), C)
    st_code, _ = assign(np.asarray(student, np.float32), C)
    top1 = float(np.mean(st_code == ts_code))
    hit, widths = 0, []
    for i, v in enumerate(_l2(np.asarray(student, np.float32))):
        cells = _probe(v, C, min_frac=min_frac)
        widths.append(len(cells))
        if int(ts_code[i]) in cells:
            hit += 1
    return {"code@1": round(top1, 4),
            "code_recall": round(hit / max(len(student), 1), 4),
            "mean_probe_cells": round(float(np.mean(widths)), 2),
            "cells": int(len(C))}


def report(V, labels=None, groups=None, teacher=None, codebook=None):
    """One channel, all three layers. `labels` is {name: bool array}."""
    out = {"separability": separability(V)}
    if labels:
        out["task_auc"] = {k: probe_auc(V, y) for k, y in labels.items()}
        vals = [v for v in out["task_auc"].values() if v is not None]
        out["task_auc_mean"] = round(float(np.mean(vals)), 4) if vals else None
    if groups is not None:
        out["retrieval_map"] = retrieval_map(V, groups)
    if teacher is not None:
        out["fidelity"] = fidelity(V, teacher)
        if codebook is not None:
            out["code"] = code_agreement(V, teacher, codebook)
    return out


def verdict(student, teacher, tol=0.03, code_recall_min=0.98):
    """The shipping gate: task AUC within `tol` of the teacher's.

    Deliberately NOT fidelity. A student that disagrees with its teacher
    clip by clip while carrying the same usable signal has done its job;
    holding it to agreement would reject it for the wrong reason.
    """
    s, t = student.get("task_auc_mean"), teacher.get("task_auc_mean")
    if s is None or t is None:
        return {"ship": False, "reason": "no task labels to score on"}
    if t < 0.55:
        return {"ship": False,
                "reason": f"teacher itself is uninformative (AUC {t}) - "
                          "fix the teacher target before distilling"}
    cr = (student.get("code") or {}).get("code_recall")
    if cr is not None and cr < code_recall_min:
        return {"ship": False, "student_auc": s, "teacher_auc": t,
                "reason": f"code_recall {cr} < {code_recall_min}: rows "
                          "whose teacher cell falls outside the student's "
                          "probe are unreachable at any k"}
    return {"ship": bool(s >= t - tol), "student_auc": s, "teacher_auc": t,
            "gap": round(s - t, 4), "tolerance": tol, "code_recall": cr}
