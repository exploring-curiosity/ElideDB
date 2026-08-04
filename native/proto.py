"""END-TO-END prototype: scout -> masked dense track -> bundles ->
events -> position-verified gate. One file, small N, fast iteration.

Two-stage tracking (measured this morning): a cheap half-res scout
finds WHERE things move; a dense full-res grid restricted by segm_mask
tracks only those patches. segm_mask keeps CoTracker's support grid
(explicit `queries` drops it - measured quality collapse), so density
comes cheap without losing context.

    python native/proto.py --eps 2        (~35 s/episode + model load)
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

STRIDE = 2            # 5 fps
SC_SCALE = 0.5        # scout resolution
SC_GRID = 20          # scout grid (400 pts)
BOX = 26              # half-size of dense boxes around moved scout pts
PT_BUDGET = 600       # dense points target
LAG = 5               # 1 s displacement lag (at 5 fps)


def arg(name, default, cast):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def track_view(model, dev, F, torch, cv2):
    """Two-stage tracking of one decoded view -> (tracks T,N,2 fullres)."""
    import torch.nn.functional  # noqa: F401
    H, W = F.shape[1:3]
    t0 = time.time()
    Fs = np.stack([cv2.resize(f, None, fx=SC_SCALE, fy=SC_SCALE,
                              interpolation=cv2.INTER_AREA) for f in F])
    vid = torch.tensor(Fs, dtype=torch.float32, device=dev) \
        .permute(0, 3, 1, 2)[None]
    with torch.no_grad():
        tr, vi = model(vid, grid_size=SC_GRID)
    scout = tr[0].float().cpu().numpy() / SC_SCALE
    del vid, tr, vi
    t_scout = time.time() - t0

    # moved & not-always-busy scout tracks mark object patches
    d = np.linalg.norm(scout[LAG:] - scout[:-LAG], axis=2).max(0)
    spd = np.linalg.norm(scout[1:] - scout[:-1], axis=2)
    busy = (spd > 3).mean(0)
    sel = np.where((d > 12) & (busy < 0.5))[0]
    mask = np.zeros((H, W), np.float32)
    for x, y in scout[0][sel]:
        mask[int(max(y - BOX, 0)):int(y + BOX),
             int(max(x - BOX, 0)):int(x + BOX)] = 1.0
    frac = float(mask.mean())
    t1 = time.time()
    if frac > 0:
        grid = int(min(110, max(20, np.sqrt(PT_BUDGET / frac))))
        vid = torch.tensor(F, dtype=torch.float32, device=dev) \
            .permute(0, 3, 1, 2)[None]
        mt = torch.tensor(mask, device=dev)[None, None]
        with torch.no_grad():
            tr, vi = model(vid, grid_size=grid, segm_mask=mt)
        dense = tr[0].float().cpu().numpy()
        del vid, tr, vi
    else:
        dense = np.zeros((len(F), 0, 2), np.float32)
    t_dense = time.time() - t1
    tracks = np.concatenate([scout, dense], axis=1)
    return tracks, dict(scout_s=t_scout, dense_s=t_dense,
                        n_scout=scout.shape[1], n_dense=dense.shape[1],
                        mask_pct=100 * frac)


def bundle(tracks, k_means_1d):
    """Movers -> velocity-signature average-linkage clusters.
    Returns labels (N,), speed (N, T-1), fitted cuts."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    T, N = tracks.shape[:2]
    spd = np.linalg.norm(np.diff(tracks, axis=0), axis=2).T   # N,T-1
    d1 = np.linalg.norm(tracks[LAG:] - tracks[:-LAG], axis=2).max(0)
    c3 = k_means_1d(np.log10(d1 + 0.1), 3)
    mv_cut = 10 ** ((c3[1] + c3[2]) / 2) - 0.1
    mv = np.where(d1 > mv_cut)[0]
    labels = np.full(N, -1, np.int32)
    if len(mv) < 3:
        return labels, spd, mv_cut, 0.0
    # RIGIDITY, not velocity signatures (both midpoint choices failed,
    # measured: mega-merge / arm-fragment spray). Two points on one
    # rigid body keep CONSTANT mutual distance all episode (std ~2px);
    # any cross-object pair changes by the carry distance (~50-150px)
    # the moment either object moves. Gripper-vs-held-block also
    # separates: constant during the carry, different before/after.
    P = tracks[:, mv, :]                       # T,M,2
    M = len(mv)
    s1 = np.zeros((M, M)); s2 = np.zeros((M, M))
    for t_ in range(T):
        dx = P[t_, :, 0][:, None] - P[t_, :, 0][None, :]
        dy = P[t_, :, 1][:, None] - P[t_, :, 1][None, :]
        dd = np.sqrt(dx * dx + dy * dy)
        s1 += dd
        s2 += dd * dd
    D = np.sqrt(np.maximum(s2 / T - (s1 / T) ** 2, 0))
    iu = np.triu_indices(M, 1)
    # 3 natural scales: same rigid body (~1-2px), coupled (gripper vs
    # held block, adjacent arm links: rigid only part of the episode,
    # ~10-30px), unrelated (~80px+). Cut between the first two.
    lc = k_means_1d(np.log10(D[iu] + 0.1), 3)
    coh = 10 ** ((lc[0] + lc[1]) / 2) - 0.1
    # COMPLETE linkage: in a rigid body EVERY pair is rigid; average
    # linkage let gripper tracks bridge into block bundles and the
    # centroid teleported between subgroups (measured: phantom arrival
    # at the arm's position, real arrival 3s late)
    Z = linkage(squareform(D, checks=False), method="complete")
    frag = fcluster(Z, t=coh, criterion="distance")
    labels[mv] = frag
    return labels, spd, mv_cut, coh


def view_events(tracks, labels, spd, k_means_1d):
    """Pure fragments (recall 1.00 measured) -> events; junk killed at
    the EVENT level by REST-BACKING (a manipulated object is still
    before and after its move; arm fragments move again within ~1s -
    the rule that carried chain_moves), duplicates merged across
    fragments by position with dominant-fragment identity."""
    ids = [b for b in np.unique(labels) if b > 0
           and (labels == b).sum() >= 3]
    if not ids:
        return []
    ls = np.log10(spd[spd > 0.05] + 1e-3)
    cs = k_means_1d(ls, 3)
    scut = 10 ** ((cs[1] + cs[2]) / 2)
    prof = {b: np.median(spd[labels == b], axis=0) for b in ids}
    # AGENT fragments: huge total path AND many bursts; a block moves
    # once or twice over tens of px. (Dropping agent exclusion with
    # the fragments rewrite was the precision collapse, measured
    # 0.38 -> 0.12.) 2-means on the combined log score.
    def n_bursts(on):
        return int(((~np.r_[False, on[:-1]]) & on).sum())
    score = {}
    for b in ids:
        m = labels == b
        path = float(np.median(spd[m].sum(1)))
        nb = n_bursts(prof[b] > scut)
        score[b] = np.log10(path + 1) + np.log10(nb + 1)
    sv_ = np.array(sorted(score.values()))
    if len(sv_) >= 4:
        c2 = k_means_1d(sv_, 2)
        acut = (c2[0] + c2[1]) / 2
    else:
        acut = np.inf
    evs = []
    for b in ids:
        if score[b] >= acut and np.isfinite(acut):
            continue
        m = labels == b
        on = prof[b] > scut
        # bursts with <=3-frame gaps merged
        spans, i0 = [], None
        for i, o in enumerate(list(on) + [False]):
            if o and i0 is None:
                i0 = i
            elif not o and i0 is not None:
                if spans and i0 - spans[-1][1] <= 3:
                    spans[-1][1] = i
                else:
                    spans.append([i0, i])
                i0 = None
        for s, e in spans:
            if e - s < 2:
                continue
            p0 = np.median(tracks[max(s - 5, 0):max(s, 1)][:, m, :]
                           .reshape(-1, 2), axis=0)
            p1 = np.median(tracks[min(e + 1, len(tracks) - 1):
                                  min(e + 6, len(tracks))][:, m, :]
                           .reshape(-1, 2), axis=0)
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                continue
            # PHYSICAL event times, not burst edges: landing wobble
            # and nearby-arm bumps extend bursts past the set-down
            # (measured: claim 3s late, outside the prim window).
            # dep = last frame still at the start rest position;
            # arr = first frame at the final rest position.
            cen = np.median(tracks[:, m, :], axis=1)      # T,2
            fs_ref = s
            for i in range(s, min(e + 1, len(cen))):
                if np.linalg.norm(cen[i] - p0) < 4.0:
                    fs_ref = i
                else:
                    break
            fe_ref = e
            for i in range(min(e, len(cen) - 1), s, -1):
                if np.linalg.norm(cen[i] - p1) < 4.0:
                    fe_ref = i
                else:
                    break
            # rest gaps: frames of stillness before/after this burst
            # in the bundle's own profile
            gb = 0
            for i in range(s - 1, -1, -1):
                if on[i]:
                    break
                gb += 1
            ga_ = 0
            for i in range(e + 1, len(on)):
                if on[i]:
                    break
                ga_ += 1
            evs.append(dict(bundle=int(b), n=int(m.sum()),
                            fs=int(fs_ref), fe=int(fe_ref),
                            rest=min(gb + (s == 0) * 99,
                                     ga_ + (e >= len(on) - 1) * 99),
                            p_dep=(float(p0[0]), float(p0[1])),
                            p_arr=(float(p1[0]), float(p1[1])),
                            disp=float(np.linalg.norm(p1 - p0))))
    return evs


def clean_events(all_evs, k_means_1d):
    """CORPUS-pooled filters (per-view fitting was wildly unstable,
    measured 14 vs 58 events across views): jitter cut, rest-backing
    cut, then cross-fragment dedupe with dominant-fragment identity."""
    if not all_evs:
        return all_evs
    dv = np.array([e["disp"] + 0.5 for e in all_evs])
    c2 = k_means_1d(np.log10(dv), 2)
    dcut = 10 ** ((c2[0] + c2[1]) / 2) - 0.5
    evs = [e for e in all_evs if e["disp"] >= dcut]
    # (rest-backing filter REMOVED: chained re-manipulations rest only
    # 0.6s and its modes overlap the arm's - measured kills of real
    # set-downs at rest=3f. Agent exclusion addresses the junk source.)
    print(f"  clean: disp>={dcut:.1f}px  "
          f"{len(all_evs)} -> {len(evs)} events")
    # dedupe radius FITTED from co-timed arrival distances: fragments
    # of one manipulation (block + its shadow + sub-parts) land 10-40
    # px apart; different manipulations land 100+ apart
    by_view = defaultdict(list)
    for e in evs:
        by_view[(e["ep"], e["sv"])].append(e)
    pair_d = []
    for lst in by_view.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a_, b_ = lst[i], lst[j]
                ov = min(a_["fe"], b_["fe"]) - max(a_["fs"], b_["fs"])
                if ov > 0.3 * min(a_["fe"] - a_["fs"] + 1,
                                  b_["fe"] - b_["fs"] + 1):
                    pair_d.append(np.hypot(
                        a_["p_arr"][0] - b_["p_arr"][0],
                        a_["p_arr"][1] - b_["p_arr"][1]))
    if len(pair_d) > 20:
        c2 = k_means_1d(np.log10(np.array(pair_d) + 1.0), 2)
        rad = 10 ** ((c2[0] + c2[1]) / 2) - 1.0
    else:
        rad = 30.0
    # bound by the jitter scale: an unbounded fit landed at 62px and
    # merged distinct stack events (measured, recall 0.73 -> 0.67)
    rad = float(min(rad, 2.0 * dcut))
    print(f"  dedupe radius {rad:.0f}px (fitted, jitter-bounded)")
    out = []
    for lst in by_view.values():
        lst.sort(key=lambda e_: -e_["n"])
        kept = []
        for e_ in lst:
            dup = False
            for k_ in kept:
                ov = min(e_["fe"], k_["fe"]) - max(e_["fs"], k_["fs"])
                if ov > 0.3 * (e_["fe"] - e_["fs"] + 1) and \
                        np.hypot(e_["p_arr"][0] - k_["p_arr"][0],
                                 e_["p_arr"][1] - k_["p_arr"][1]) \
                        < rad:
                    dup = True
                    break
            if not dup:
                kept.append(e_)
        out += kept
    return out


def main():
    import cv2
    import torch
    import pyarrow.parquet as pq
    from elidedb import Store
    import chain_delta as cd
    import chain_grade as cg
    from chain_delta import k_means_1d

    n_eps = arg("--eps", 2, int)
    ep_from = arg("--from", 0, int)
    db = Store.open(str(ROOT / "lake/sim_chains"))
    views = [v for v in cd.episode_views(db)
             if ep_from <= v[0] < ep_from + n_eps]
    from track import SCRATCH as _SC
    _todo = [v for v in views if not
             (_SC / "proto_tracks_v2" / f"{v[0]:05d}_{v[1]}.npz")
             .exists()]
    print(f"{len(views)} views, {len(_todo)} to track "
          f"(ETA ~{len(_todo)*40/60:.0f} min)", flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = None
    if _todo:
        model = torch.hub.load("facebookresearch/co-tracker",
                               "cotracker3_offline").to(dev).eval()

    from track import SCRATCH
    tdir = SCRATCH / "proto_tracks_v2"
    tdir.mkdir(parents=True, exist_ok=True)
    all_evs = []
    for ep, sv, sl in views:
        tc = tdir / f"{ep:05d}_{sv}.npz"
        if tc.exists():
            z = np.load(tc)
            tracks, ts = z["tracks"].astype(np.float32), z["ts"]
            info = dict(scout_s=0, dense_s=0, n_dense=-1, mask_pct=-1)
        else:
            ts, F = cd.decode_view(db, sl)
            F = F[::STRIDE]
            ts = ts[::STRIDE]
            tracks, info = track_view(model, dev, F, torch, cv2)
            np.savez_compressed(tc, tracks=tracks.astype(np.float16),
                                ts=np.asarray(ts, np.int64))
        labels, spd, mv_cut, coh = bundle(tracks, k_means_1d)
        evs = view_events(tracks, labels, spd, k_means_1d)
        print(f"  ep{ep} {sv}: scout {info['scout_s']:.0f}s dense "
              f"{info['n_dense']}pts {info['dense_s']:.0f}s "
              f"(mask {info['mask_pct']:.1f}%)  bundles "
              f"{len([b for b in np.unique(labels) if b>0])}  "
              f"raw events {len(evs)}", flush=True)
        for e_ in evs:
            e_.update(ep=ep, sv=sv,
                      t_dep=int(ts[min(e_["fs"], len(ts) - 1)]),
                      t_arr=int(ts[min(e_["fe"], len(ts) - 1)]))
            all_evs.append(e_)
    all_evs = clean_events(all_evs, k_means_1d)
    claims, meta = [], []
    for i, e_ in enumerate(all_evs):
        claims.append([e_["ep"], e_["sv"], e_["t_dep"], *e_["p_dep"]])
        meta.append((i, "dep"))
        claims.append([e_["ep"], e_["sv"], e_["t_arr"], *e_["p_arr"]])
        meta.append((i, "arr"))

    # ---- gate metrics (EVAL side) ----
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    cols_of = {}
    for e, b, c in zip(t["episode"], t["block"], t["color"]):
        cols_of.setdefault(int(e), {})[str(b)] = str(c)
    # tighter palette angle: at 60 milli the tan table itself counts
    # as "orange" (angle 42, measured) and orange blocks can never
    # verify. 25 milli keeps block-vs-palette (<10) and excludes the
    # table. Grader-side only.
    _frac0 = cg.frac_color
    cg.frac_color = lambda crop, rgb: _frac0(crop, rgb, ang_thr=25.0)
    labs = cg.label_all(db, claims, cols_of)
    cg.frac_color = _frac0
    # margin rule instead of a hard floor: at 25 milli a small stacked
    # block reads fa~0.06 (measured blue: 0.06 vs before 0.00); what
    # verifies an event is the CONTRAST between sides, not raw area
    ver = {}
    for k, (i, kind) in enumerate(meta):
        fb, fa = labs[k, 0], labs[k, 1]
        for b in fa:
            a_, b_ = fa[b], fb.get(b, 0)
            if kind == "arr" and a_ > 0.05 and a_ > 2.5 * b_ + 0.02:
                ver[k] = b
            elif kind == "dep" and b_ > 0.05 \
                    and b_ > 2.5 * a_ + 0.02:
                ver[k] = b
    epd = db.table("episodes").scan().to_pydict()
    ep0 = {int(e): int(x) for e, x in zip(epd["episode_index"],
                                          epd["ts"])}
    got = defaultdict(list)
    for k, (i, kind) in enumerate(meta):
        if k in ver:
            e_ = all_evs[i]
            tt = e_["t_dep"] if kind == "dep" else e_["t_arr"]
            got[e_["ep"]].append(((tt - ep0[e_["ep"]]) / 1e9, kind,
                                  ver[k]))
    # match inside the PRIM'S OWN INTERVAL (t0-1.5, t1+1.5): the last
    # prim's t1 includes settle+retreat, so |te - t1| < 3 missed
    # arrivals that landed mid-prim (measured: blue stack at 20.0s vs
    # t1 23.9)
    sd, pk = defaultdict(list), defaultdict(list)
    for e, p, b, a0, a1 in zip(t["episode"], t["prim"], t["block"],
                               t["t0"], t["t1"]):
        if not (ep_from <= int(e) < ep_from + n_eps):
            continue
        (pk if str(p) == "pick" else sd)[int(e)].append(
            (float(a0), float(a1), str(b)))

    def recall(anch, kind):
        h = tot = 0
        miss = []
        for e, lst in anch.items():
            for a0, a1, b in lst:
                tot += 1
                if any(k2 == kind and b2 == b
                       and a0 - 1.5 <= te <= a1 + 1.5
                       for te, k2, b2 in got.get(e, [])):
                    h += 1
                else:
                    miss.append((e, b, round(a1, 1)))
        return h, tot, miss

    h1, t1, m1 = recall(sd, "arr")
    h2, t2, m2 = recall(pk, "dep")
    bl = defaultdict(list)
    for k, (i, kind) in enumerate(meta):
        if k in ver:
            e_ = all_evs[i]
            bl[(e_["ep"], e_["sv"], e_["bundle"])].append(ver[k])
    ok = tot = 0
    for bs in bl.values():
        for a in range(len(bs)):
            for b_ in range(a + 1, len(bs)):
                tot += 1
                ok += bs[a] == bs[b_]
    print(f"\nGATE (subset {ep_from}..{ep_from+n_eps-1}):")
    print(f"  set-down recall {h1}/{t1} = {h1/max(t1,1):.2f}   "
          f"pick recall {h2}/{t2} = {h2/max(t2,1):.2f}")
    print(f"  precision {len(ver)}/{len(claims)} = "
          f"{len(ver)/max(len(claims),1):.2f}   slot-consistency "
          f"{ok}/{tot} = {ok/max(tot,1):.2f}")
    if m1:
        print(f"  missed set-downs: {m1[:8]}")
    if m2:
        print(f"  missed picks: {m2[:8]}")
    if "--debug" in sys.argv:
        for (e_missed, b_missed, a1_missed) in m1[:6]:
            print(f"  DEBUG miss ep{e_missed} {b_missed} @{a1_missed}:")
            for k, (i, kind) in enumerate(meta):
                ev_ = all_evs[i]
                if ev_["ep"] != e_missed:
                    continue
                tt = (ev_["t_arr"] - ep0[e_missed]) / 1e9
                if abs(tt - a1_missed) > 6:
                    continue
                fb, fa = labs[k, 0], labs[k, 1]
                tops = sorted(fa, key=lambda b: -fa[b])[:2]
                s_ = " ".join(f"{b}:a{fa[b]:.2f}/b{fb.get(b,0):.2f}"
                              for b in tops)
                print(f"    claim {kind} b{ev_['bundle']} {ev_['sv']} "
                      f"t{tt:5.1f} pos({claims[k][3]:.0f},"
                      f"{claims[k][4]:.0f}) ver={ver.get(k)} {s_}")


if __name__ == "__main__":
    main()
