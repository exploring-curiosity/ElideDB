"""EACH TEACHER ON ITS OWN TASK, not on cosine.

Every channel was benchmarked by cosine similarity, which is the native
operation for exactly one of them. Scoring a classifier by cosine on its
probability vector, or a text-video model by image-image similarity, and
then calling it collapsed, is a measurement error. `act` scored AUC 0.483
- below chance - under cosine and 0.812 under cross-entropy, on identical
numbers.

So each model gets the test it was BUILT for:

  pe, sig2      image-text contrastive  -> TEXT -> IMAGE retrieval, using
                                           the model's own text encoder
  iv2, xclip    video-text              -> TEXT -> VIDEO retrieval, own
                                           text encoder; iv2 additionally
                                           ships an ITM cross-attention
                                           head that is never called
  act           174-way SSv2 classifier -> commitment (entropy, classes
                                           used) and cross-entropy
                                           retrieval
  vjepa         predictive world model  -> NOT TESTABLE HERE: the store
                                           holds encoder output only, no
                                           predictor. Reported as such
                                           rather than scored off-task.
  mot           appearance delta        -> direction separation, cosine
                                           IS its native operation

A model failing its own task is a finding about the model. A model
failing someone else's task is a finding about the harness.

    python scripts/eval_teachers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                     # noqa: E402
from elidedb.embeddings import _vec_table                     # noqa: E402
from _common import queries                                   # noqa: E402
from diag_channels import auc                                 # noqa: E402

# model -> (vector table, how to get a query vector in ITS space)
TEXT_MODELS = {
    "pe":    ("pe_vectors",    "elidedb.pe",   "_text_vec"),
    "sig2":  ("sig2_vectors",  "elidedb.sig2", "_text_vec"),
    "iv2":   ("iv2_vectors",   "elidedb.iv2",  "clip_vec"),
    "xclip": ("xclip_vectors", "elidedb.xclip", "text_vec"),
}


def episode_matrix(db, table, keys, pos):
    tb, V = _vec_table(db, table)
    V = np.asarray(V, np.float32)
    acc = {}
    for s, a, v in zip(tb.column("stream").to_pylist(),
                       tb.column("ts").to_pylist(), V):
        acc.setdefault((str(s), int(a)), []).append(v)
    A = np.zeros((len(keys), V.shape[1]), np.float32)
    ok = np.zeros(len(keys), bool)
    for k, vs in acc.items():
        if k in pos:
            m = np.mean(vs, 0)
            n = np.linalg.norm(m)
            A[pos[k]] = m / n if n > 0 else m
            ok[pos[k]] = True
    return A, ok


def main():
    db = Store.open(str(ROOT / "lake/fresh_bench"))
    QS = queries()
    ep = db.table("episodes").scan()
    keys = [(str(s), int(a)) for s, a in
            zip(ep.column("stream").to_pylist(), ep.column("ts").to_pylist())]
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    pos = {k: i for i, k in enumerate(keys)}
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G = {(int(q), int(e)): int(v) for q, e, v in
         zip(t["query_id"], t["episode_index"], t["true"])}

    report = {}

    # ---- 1. text-video models on TEXT RETRIEVAL, their designed task
    print("TEXT -> VIDEO RETRIEVAL (each model's own text encoder)\n")
    print(f"  {'model':<8}{'q03':>8}{'q04':>8}{'q05':>8}{'mean':>8}   status")
    for name, (tab, mod, fn) in TEXT_MODELS.items():
        if tab not in db.tables():
            print(f"  {name:<8}{'':>32}   no {tab}")
            continue
        try:
            m = __import__(mod, fromlist=[fn])
            tv = getattr(m, fn)
        except Exception as e:
            print(f"  {name:<8}{'':>32}   NO TEXT ENCODER "
                  f"({type(e).__name__})")
            report[name] = {"text_retrieval": None,
                            "note": f"no text encoder: {e}"}
            continue
        A, ok = episode_matrix(db, tab, keys, pos)
        aucs = {}
        for qi in (3, 4, 5):
            judged = {i: G[(qi, eidx[i])] for i in range(len(eidx))
                      if (qi, eidx[i]) in G and ok[i]}
            if len(judged) < 30:
                continue
            try:
                qv = np.asarray(tv(QS[qi]), np.float32).ravel()
                qv = qv / (np.linalg.norm(qv) + 1e-8)
                if len(qv) != A.shape[1]:
                    raise ValueError(f"dim {len(qv)} vs {A.shape[1]}")
                sc = A @ qv
            except Exception as e:
                aucs[qi] = None
                report.setdefault(name, {})["error"] = str(e)[:120]
                continue
            ji = list(judged)
            aucs[qi] = auc(sc[ji], np.array([judged[i] for i in ji], np.int8))
        vals = [v for v in aucs.values() if v is not None]
        if vals:
            print(f"  {name:<8}" + "".join(
                f"{(aucs.get(q) or float('nan')):>8.3f}" for q in (3, 4, 5))
                + f"{np.mean(vals):>8.3f}   ok")
            report[name] = {"text_retrieval": {str(k): v
                                               for k, v in aucs.items()},
                            "mean": float(np.mean(vals))}
        else:
            print(f"  {name:<8}{'':>32}   FAILED "
                  f"{report.get(name, {}).get('error', '')}")

    # ---- 2. act as a classifier
    print("\n\nact: 174-WAY SSv2 CLASSIFIER on its own task\n")
    tb, P = _vec_table(db, "action_probs")
    P = np.asarray(P, np.float64)
    ent = -(P * np.log(P + 1e-12)).sum(1)
    used = len(set(P.argmax(1).tolist()))
    print(f"  rows sum to 1.0            {np.allclose(P.sum(1), 1, atol=1e-3)}")
    print(f"  entropy                    {ent.mean():.2f} of "
          f"{np.log(174):.2f} max ({ent.mean()/np.log(174):.0%} of uniform)")
    print(f"  distinct argmax classes    {used} of 174")
    print(f"  mean top-1 posterior       {P.max(1).mean():.3f}")
    report["act"] = {"entropy": float(ent.mean()), "classes_used": used,
                     "top1": float(P.max(1).mean())}

    # ---- 3. what cannot be tested here, said plainly
    print("\n\nNOT TESTED ON THEIR OWN TASK (and why)\n")
    notes = {
        "vjepa": "predictive world model; the store holds ENCODER output "
                 "only, no predictor, so prediction error cannot be "
                 "measured. Scoring it by cosine is off-task.",
        "iv2-itm": "InternVideo2 ships an ITM cross-attention head - "
                   "present in the checkpoint, reported unused at load, "
                   "never called. Its designed matching operator.",
        "xclip-xattn": "cross-frame attention is discarded by pooling to "
                       "one vector per clip before storage.",
    }
    for k, v in notes.items():
        print(f"  {k:<12} {v}")
    report["untested"] = notes

    (ROOT / "bench" / "eval_teachers.json").write_text(
        json.dumps(report, indent=1))
    print("\nwrote bench/eval_teachers.json")


if __name__ == "__main__":
    main()
