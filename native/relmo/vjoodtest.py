"""ood_test: real robot video. Read ONCE.

bridge is the strongest domain shift available without a download - real
cameras, real kitchens and tabletops, a WidowX arm instead of a simulated
Panda, 5 fps, natural lighting, and NO sim state at all, which is exactly what
serve time looks like. The recurrence trained only on RoboCasa renders and only
on physics targets read from RoboCasa's simulator.

WHAT THIS CAN AND CANNOT SHOW. Of 50,415 bridge episodes only 1,499 reach 64
frames, and those hold two event types: "sweep into pile" and a family of "put
X in pot/pan, put pot/pan on stove". So it is a balanced TWO-CLASS problem. It
is real evidence that the representation survives the domain shift; it is NOT
evidence about fine-grained discrimination, since a weak appearance feature
would also separate sweeping from placing. Quoted with that limit attached.

Class comes from the free-text task string, GRADING ONLY, and never reaches the
model. Same protocol as everywhere else: k = support, chance = support/pool.

    python -m relmo.vjoodtest --tags sd_r1.0_s0,...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjz  # noqa: E402
from relmo.vjeval import l2  # noqa: E402
from relmo.vjeval5 import dtw_from_cost  # noqa: E402

REC = R.BASE / "vjrec4" / "bridge_L6"
SIG = R.BASE / "vjsig" / "bridge"


def evaluate(desc, boot=2000, seed=0):
    ids = sorted(desc)
    cls = np.array(["sweep" if i.startswith("sweep") else "put" for i in ids])
    P = np.stack([l2(desc[i]) for i in ids])
    hit, sup, rnd, qcls = [], [], [], []
    for k, q in enumerate(ids):
        m = np.arange(len(ids)) != k
        sv = cls[m] == cls[k]
        s_ = int(sv.sum())
        if s_ < 5:
            continue
        C = 1.0 - np.einsum("sd,nkd->nsk", l2(desc[q]), P[m])
        sc = -dtw_from_cost(C)
        hit.append(int(sv[np.argsort(-sc)[:s_]].sum()))
        sup.append(s_)
        rnd.append(s_ * s_ / int(m.sum()))
        qcls.append(cls[k])
    hit, sup, rnd = (np.array(v, float) for v in (hit, sup, rnd))
    qcls = np.array(qcls)
    out = dict(overall=hit.sum() / sup.sum(), chance=rnd.sum() / sup.sum(),
               n=len(hit))
    rng = np.random.default_rng(seed)
    b = [hit[k].sum() / sup[k].sum()
         for k in rng.integers(0, len(hit), (boot, len(hit)))]
    out["ci95"] = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    for c in ("sweep", "put"):
        sel = qcls == c
        out[c] = float(hit[sel].sum() / sup[sel].sum()) if sel.any() else \
            float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    a = ap.parse_args()

    recs = vjz.gather([p.stem for p in sorted(REC.glob("*.npz"))],
                      "bridge", want_y=False, rec_dir=REC, sig_dir=SIG)
    print(f"bridge records: {len(recs)}")
    rows = [("frozen pred_change", {i: recs[i]["a"] for i in recs})]
    for t in [x.strip() for x in a.tags.split(",") if x.strip()]:
        m, ck = vjz.load_ckpt(t)
        rows.append((t, vjz.encode(m, ck, recs)))

    print(f"\n{'arm':22s} {'ood_test':>9s} {'95% CI':>16s} {'chance':>8s} "
          f"{'lift':>6s} {'sweep':>7s} {'put':>7s}")
    agg = {}
    for name, d in rows:
        r = evaluate(d)
        agg[name] = r["overall"]
        ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
        print(f"{name:22s} {r['overall']:9.3f} {ci:>16s} {r['chance']:8.3f} "
              f"{r['overall'] / r['chance']:5.2f}x {r['sweep']:7.3f} "
              f"{r['put']:7.3f}")
    R.log("vjoodtest", n=len(recs), **{k.replace(".", "_"): round(v, 4)
                                       for k, v in agg.items()})


if __name__ == "__main__":
    main()
