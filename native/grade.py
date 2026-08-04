"""G1 GATE: position-verified event grading. PLAN §5.6. EVAL-ONLY.

Reuses scripts/chain_grade.py's palette machinery (truth-side by
definition). An event only counts if the pixels at its own position
show the truth block appearing (arrival) or leaving (departure).
Time-window graders reported 0.91-0.98 where this instrument measured
0.48 - that gap is the reason this file exists and why no detection
claim may bypass it.

GATE: set-down recall >=0.95, pick recall >=0.95, precision >=0.80,
slot consistency >=0.90.

    python native/grade.py [--store lake/sim_chains] [--limit N]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

VER = "v1"
WIN_S = 3.0          # truth-anchor tolerance


def scratch():
    from track import SCRATCH
    return SCRATCH


def claims_from_events(evs):
    """One claim per event side: (row_for_label_all, kind, key)."""
    rows, meta = [], []
    for i, e in enumerate(evs):
        rows.append([e["ep"], e["sv"], e["t_dep"],
                     e["p_dep"][0], e["p_dep"][1]])
        meta.append((i, "dep"))
        rows.append([e["ep"], e["sv"], e["t_arr"],
                     e["p_arr"][0], e["p_arr"][1]])
        meta.append((i, "arr"))
    return rows, meta


def main():
    import pyarrow.parquet as pq
    from elidedb import Store
    import chain_grade as cg
    import events as ev_mod

    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv \
        else 0
    db = Store.open(str(store))
    epd = db.table("episodes").scan().to_pydict()
    ep0 = {int(e): int(t) for e, t in zip(epd["episode_index"],
                                          epd["ts"])}
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    cols_of = {}
    for e, b, c in zip(t["episode"], t["block"], t["color"]):
        cols_of.setdefault(int(e), {})[str(b)] = str(c)

    z = np.load(scratch() / f"native_events_{VER}_{store.name}.npz",
                allow_pickle=True)
    evs = list(z["evs"])
    if limit:
        evs = [e for e in evs if e["ep"] < limit]
    rows, meta = claims_from_events(evs)
    print(f"{len(evs)} events -> {len(rows)} claims", flush=True)

    cache = scratch() / f"native_labels_{VER}_{store.name}.npz"
    if cache.exists():
        labs = np.load(cache, allow_pickle=True)["labs"]
        if len(labs) != len(rows):
            labs = None
    else:
        labs = None
    if labs is None:
        labs = cg.label_all(db, rows, cols_of)
        np.savez(cache, labs=labs)

    THR = cg.THR
    ver = {}          # claim index -> block verified
    for k, (i, kind) in enumerate(meta):
        fb, fa = labs[k, 0], labs[k, 1]
        for b in fa:
            if kind == "arr" and fa[b] > THR and fb.get(b, 0) < THR / 2:
                ver[k] = b
            elif kind == "dep" and fb.get(b, 0) > THR \
                    and fa[b] < THR / 2:
                ver[k] = b

    # ---- recall against truth prims
    sd, pk = defaultdict(list), defaultdict(list)
    for e, p, b, a1 in zip(t["episode"], t["prim"], t["block"],
                           t["t1"]):
        (pk if str(p) == "pick" else sd)[int(e)].append(
            (float(a1), str(b)))
    got = defaultdict(list)          # (ep) -> (t_s, kind, block)
    for k, (i, kind) in enumerate(meta):
        if k not in ver:
            continue
        e = evs[i]
        tt = e["t_dep"] if kind == "dep" else e["t_arr"]
        got[e["ep"]].append(((tt - ep0[e["ep"]]) / 1e9, kind, ver[k]))

    def recall(anchors, kind):
        hv = tt_ = 0
        miss = []
        for e, lst in anchors.items():
            if limit and e >= limit:
                continue
            for a1, b in lst:
                tt_ += 1
                if any(k2 == kind and b2 == b and abs(te - a1) < WIN_S
                       for te, k2, b2 in got.get(e, [])):
                    hv += 1
                else:
                    miss.append((e, b, a1))
        return hv, tt_, miss

    h1, t1_, miss_sd = recall(sd, "arr")
    h2, t2_, miss_pk = recall(pk, "dep")
    prec = len(ver) / max(len(rows), 1)

    # ---- slot consistency: same bundle -> same truth block
    bl = defaultdict(list)
    for k, (i, kind) in enumerate(meta):
        if k in ver:
            e = evs[i]
            bl[(e["ep"], e["sv"], e["bundle"])].append(ver[k])
    ok = tot = 0
    for key, bs in bl.items():
        for a in range(len(bs)):
            for b_ in range(a + 1, len(bs)):
                tot += 1
                ok += bs[a] == bs[b_]
    slot = ok / max(tot, 1)

    print(f"\nGATE G1 ({store.name})")
    print(f"  set-down recall  {h1}/{t1_} = {h1/max(t1_,1):.3f}"
          f"   (gate 0.95)")
    print(f"  pick recall      {h2}/{t2_} = {h2/max(t2_,1):.3f}"
          f"   (gate 0.95)")
    print(f"  precision        {len(ver)}/{len(rows)} = {prec:.3f}"
          f"   (gate 0.80)")
    print(f"  slot consistency {ok}/{tot} = {slot:.3f}"
          f"   (gate 0.90)")
    passed = (h1 / max(t1_, 1) >= 0.95 and h2 / max(t2_, 1) >= 0.95
              and prec >= 0.80 and slot >= 0.90)
    print(f"  => {'PASS' if passed else 'FAIL'}")
    if miss_sd[:10]:
        print("  missed set-downs (ep, block, t):",
              miss_sd[:10])
    return passed


if __name__ == "__main__":
    main()
