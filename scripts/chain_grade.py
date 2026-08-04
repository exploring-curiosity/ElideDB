"""POSITION-GROUNDED event grading. EVAL-SIDE ONLY - never a path.

Four position-blind graders in a row (nearest-in-time, existence-in-
window, boundary-window, stage counters) reported detection recall
0.91-0.98 while the true number was 0.48: co-timed arm/shadow junk
near every anchor satisfied any time-only criterion. This grader is
the fix and the program's standing instrument: an event is a REAL
arrival/departure of block b iff its clean single-frame crop CONTAINS
b's truth colour on the correct side (palette anchors are truth-side
by definition - the grader may know what the write path never can).

Prints: position-verified pool composition, TRUE set-down recall,
corpus-feature AUCs against verified labels, and the CEILING bench
(verified events + truth slots) - the number any write-path detection
rebuild must push toward 0.95+ before retrieval work resumes.

    python scripts/chain_grade.py
"""
from __future__ import annotations

import bisect
import io
import contextlib
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import chain_serial as cs                                      # noqa: E402
import chain_delta as cd                                       # noqa: E402
import chain_qbe                                               # noqa: E402
from chain_moves import bench                                  # noqa: E402
from elidedb import Store                                      # noqa: E402

PAL = dict(red=(200, 40, 40), green=(40, 170, 40),
           blue=(40, 60, 220), cyan=(40, 200, 200),
           orange=(235, 140, 35), yellow=(230, 220, 40),
           purple=(150, 60, 200), white=(235, 235, 235),
           pink=(240, 120, 180), brown=(140, 90, 50),
           magenta=(220, 50, 220))
THR = 0.08
LABS = cs.SCRATCH / "palette_labels.npz"


def frac_color(crop, rgb, ang_thr=60.0, mag_min=60):
    v = crop.reshape(-1, 3).astype(np.float64)
    m = np.linalg.norm(v, axis=1)
    ok = m > mag_min
    if not ok.any():
        return 0.0
    u = np.asarray(rgb, np.float64)
    cos = (v[ok] @ u) / np.maximum(m[ok] * np.linalg.norm(u), 1e-6)
    return float(((1 - cos) * 1000 < ang_thr).mean())


def label_all(db, events, cols_of):
    from tqdm import tqdm
    by_ev = {}
    for i, ev in enumerate(events):
        by_ev.setdefault((int(ev[0]), str(ev[1])), []).append(i)
    views = {(v[0], v[1]): v[2] for v in cd.episode_views(db)}
    labs = np.zeros((len(events), 2), object)
    for (e, sv), idx in tqdm(sorted(by_ev.items()), desc="label",
                             mininterval=10):
        ts, F = cd.decode_view(db, views[(e, sv)])
        T, fps, w, g, ga, step = cd.grid_params(ts, F)
        tsl = list(ts)
        blocks = cols_of.get(e, {})
        for i in idx:
            ev = events[i]
            c_ev = min(bisect.bisect_left(tsl, int(ev[2])), len(F) - 1)
            out = []
            for off in (-(g + w // 2), (ga + w // 2)):
                fi = min(max(c_ev + off, 0), len(F) - 1)
                cx, cy = ev[3], ev[4]
                y0 = max(int(cy) - 30, 0)
                x0 = max(int(cx) - 30, 0)
                c_ = F[fi, y0:int(cy) + 30, x0:int(cx) + 30]
                out.append({b: frac_color(c_, PAL.get(col,
                                                      (128, 128, 128)))
                            for b, col in blocks.items()})
            labs[i, 0], labs[i, 1] = out
    return labs


def main():
    import pyarrow.parquet as pq
    events, crops, agents = cs.load_events()
    dirs, junk = cd.classify_events(events)
    db = Store.open(str(ROOT / "lake/sim_chains"))
    epd = db.table("episodes").scan().to_pydict()
    ep0 = {int(e): int(t) for e, t in zip(epd["episode_index"],
                                          epd["ts"])}
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    cols_of = {}
    for e, b, c in zip(t["episode"], t["block"], t["color"]):
        cols_of.setdefault(int(e), {})[str(b)] = str(c)
    if LABS.exists():
        labs = np.load(LABS, allow_pickle=True)["labs"]
    else:
        labs = label_all(db, events, cols_of)
        np.savez(LABS, labs=labs)

    ev_role = []
    for i, ev in enumerate(events):
        fb, fa = labs[i, 0], labs[i, 1]
        for b in fa:
            if fa[b] > THR and fb.get(b, 0) < THR / 2:
                ev_role.append((i, "arr", b))
            elif fb.get(b, 0) > THR and fa[b] < THR / 2:
                ev_role.append((i, "dep", b))
    print(f"position-verified: {len(ev_role)} of {len(events)}")
    by_e = {}
    for i, r, b in ev_role:
        by_e.setdefault(int(events[i][0]), []).append((i, r, b))
    sd = {}
    for e, p, b, a1 in zip(t["episode"], t["prim"], t["block"],
                           t["t1"]):
        if str(p) != "pick":
            sd.setdefault(int(e), []).append((float(a1), str(b)))
    have = tot = 0
    for e, lst in sd.items():
        for a1, b in lst:
            tot += 1
            if any(r == "arr" and b2 == b
                   and abs((int(events[i][2]) - ep0[e]) / 1e9 - a1)
                   < 3.0 for i, r, b2 in by_e.get(e, [])):
                have += 1
    print(f"TRUE set-down recall: {have}/{tot} = {have/tot:.2f}")

    mans = {}
    for e in tmpl:
        lst = sorted(by_e.get(e, []),
                     key=lambda x: int(events[x[0]][2]))
        ded = []
        for i, r, b in lst:
            tt = int(events[i][2])
            if ded and ded[-1][1] == r and ded[-1][2] == b \
                    and tt - ded[-1][0] < int(1.5e9):
                continue
            ded.append((tt, r, b))
        ms, open_dep = [], {}
        for tt, r, b in ded:
            if r == "dep":
                open_dep[b] = tt
            else:
                t0 = open_dep.pop(b, tt)
                ms.append((t0, tt, b, 100.0, False))
        mans[e] = ms
    seqs = cs.tokenise(mans)
    chain_qbe.W_KIND = 0.5
    bench(seqs, tmpl, "CEILING DEV (verified + truth slots)",
          ("swap", "precarious", "push_then_build",
           "build_unstack_move"), w_slot=1.0)
    bench(seqs, tmpl, "CEILING HOLDOUT",
          ("relocate_build", "two_sites_merge"), w_slot=1.0)


if __name__ == "__main__":
    main()
