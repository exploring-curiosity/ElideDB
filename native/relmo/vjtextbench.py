"""Query-by-TEXT and query-by-EXAMPLE on the same pool, P@support and P@10.

WHAT CAN AND CANNOT ANSWER A TEXT QUERY TODAY. The trained ranker
(relmo/vjrank.py) encodes a 24-step sequence of V-JEPA expectation + realised
change + SigLIP appearance. A text string has no V-JEPA latents and no temporal
extent, so the ranker cannot embed one. Text therefore runs through the RAW
SigLIP channel only - text tower against mean-pooled image embeddings - which
is a different and much weaker system than the one that scores 0.894 by
example. Reporting them side by side is the point: it measures the cost of the
modality, not two settings of one model.

TEXT QUERIES ARE THE CORPUS'S OWN INSTRUCTIONS, verbatim, one per task family
("Open the cabinet doors."). They are graded against vjeval.group_key, so a
cabinet query is correct when it returns any hinged-door opening.

P@10 CAVEAT, stated because it bites hard here: a class with 8 items in the
pool cannot exceed 0.8 at k=10. Support is printed beside every row so a
capped number is never mistaken for a bad one.

    python -m relmo.vjtextbench
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrel  # noqa: E402
from relmo.vjeval import group_key, parse  # noqa: E402
from relmo.vjood import recs_for  # noqa: E402
from relmo.vjrankeval import load_ckpt, score_matrix  # noqa: E402
from relmo.vjsig import MODEL as SIG_MODEL  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402


def instructions(dataset="rcasa"):
    """task family -> its modal natural-language instruction."""
    man = R.read_manifest(dataset)
    c = defaultdict(Counter)
    for e in man["episodes"]:
        if e.get("instruction"):
            c[e["task"]][e["instruction"]] += 1
    return {t: v.most_common(1)[0][0] for t, v in c.items()}


def prec_at(order, correct, k):
    k = min(k, len(order))
    return float(correct[order[:k]].sum()) / k if k else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="rk_noscorer_s0,rk_noscorer_s1,"
                                      "rk_noscorer_s2")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    sp = load_split()
    Rin = recs_for("rcasa")
    meta = vjrel.meta_table("rcasa")
    val = [i for i in sorted(sp["val"]) if i in Rin]
    tst = [i for i in sorted(sp["test"]) if i in Rin]
    pool = sorted(set(val) | set(tst))
    gp = np.array([group_key(parse(i)) for i in pool])
    rp = np.array([meta[i]["rollout"] for i in pool])
    print(f"pool {len(pool)} episodes (val+test), {len(set(gp))} classes")

    # ---------------- query by TEXT: raw SigLIP text tower ----------------
    instr = instructions("rcasa")
    tasks = sorted(instr)
    tok = AutoTokenizer.from_pretrained(SIG_MODEL)
    sig = AutoModel.from_pretrained(SIG_MODEL, dtype=torch.float32).eval()
    with torch.no_grad():
        t = tok([instr[k] for k in tasks], padding="max_length",
                max_length=64, truncation=True, return_tensors="pt")
        T = sig.get_text_features(**t)
    T = torch.nn.functional.normalize(T, dim=-1).numpy().astype(np.float64)
    IMG = np.stack([Rin[i]["sig"].mean(0) for i in pool]).astype(np.float64)
    IMG = IMG / (np.linalg.norm(IMG, axis=-1, keepdims=True) + 1e-9)
    S_txt = T @ IMG.T                                    # (12, |pool|)

    # WHY TEXT FAILS, measured rather than asserted: the text tower
    # collapses the direction of the action. If 'Open the cabinet
    # doors' and 'Close the cabinet doors' are the same point, no
    # image-side quality can separate the two classes they name.
    C = T @ T.T
    print("\ntext-tower collapse (cosine between instruction pairs):")
    for x, y in (("OpenCabinet", "CloseCabinet"),
                 ("OpenMicrowave", "CloseMicrowave"),
                 ("OpenDrawer", "CloseDrawer")):
        if x in tasks and y in tasks:
            print(f'   {x:16s} vs {y:16s} '
                  f'{C[tasks.index(x), tasks.index(y)]:.4f}')

    rows = []
    for qi, task in enumerate(tasks):
        cls = group_key(parse(task + "_episode_000000__x"))
        correct = (gp == cls)
        sup = int(correct.sum())
        if sup < 5:
            continue
        order = np.argsort(-S_txt[qi])
        rows.append((task, cls, sup, prec_at(order, correct, sup),
                     prec_at(order, correct, a.k), sup / len(pool)))
    print(f"\n=== QUERY BY TEXT (raw SigLIP text tower - the ranker cannot "
          f"embed text) ===")
    print(f"{'text query (task)':26s} {'class':18s} {'sup':>4s} "
          f"{'P@sup':>7s} {'P@'+str(a.k):>6s} {'chance':>7s}")
    for t_, c_, s_, ps, pk, ch in rows:
        print(f"{t_:26s} {c_:18s} {s_:4d} {ps:7.3f} {pk:6.3f} {ch:7.3f}")
    w = np.array([r[2] for r in rows], float)
    print(f"{'MEAN (support-weighted)':26s} {'':18s} {int(w.sum()):4d} "
          f"{np.average([r[3] for r in rows], weights=w):7.3f} "
          f"{np.average([r[4] for r in rows], weights=w):6.3f} "
          f"{np.average([r[5] for r in rows], weights=w):7.3f}")

    # ---------------- query by EXAMPLE: the trained ranker ----------------
    print(f"\n=== QUERY BY EXAMPLE (trained ranker, {a.tags.count(',')+1} "
          f"seeds) ===")
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for tag in [x.strip() for x in a.tags.split(",") if x.strip()]:
        model, _ = load_ckpt(tag)
        S = score_matrix(model, Rin, tst, pool)
        for qi, q in enumerate(tst):
            keep = rp != meta[q]["rollout"]
            correct = gp[keep] == group_key(parse(q))
            sup = int(correct.sum())
            if sup < 5:
                continue
            order = np.argsort(-S[qi][keep])
            r = agg[group_key(parse(q))]
            r[0] += prec_at(order, correct, sup)
            r[1] += prec_at(order, correct, a.k)
            r[2] += sup / int(keep.sum())
            r[3] += sup
            r[4] += 1
    print(f"{'class':18s} {'n_q':>4s} {'sup':>5s} {'P@sup':>7s} "
          f"{'P@'+str(a.k):>6s} {'chance':>7s}")
    tot = np.zeros(3)
    n = 0
    for c_, v in sorted(agg.items(), key=lambda x: -x[1][1] / x[1][4]):
        print(f"{c_:18s} {v[4]:4d} {v[3]/v[4]:5.0f} {v[0]/v[4]:7.3f} "
              f"{v[1]/v[4]:6.3f} {v[2]/v[4]:7.3f}")
        tot += np.array([v[0], v[1], v[2]])
        n += v[4]
    print(f"{'MEAN over queries':18s} {n:4d} {'':5s} {tot[0]/n:7.3f} "
          f"{tot[1]/n:6.3f} {tot[2]/n:7.3f}")
    R.log("vjtextbench", k=a.k, pool=len(pool))


if __name__ == "__main__":
    main()
