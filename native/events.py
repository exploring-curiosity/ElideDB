"""[E] Events and chain tokens from bundles. PLAN §5.5. PURE CODE.

One motion burst of one non-agent bundle IS one manipulation:
departure at burst start, arrival at burst end, positions read from
the rest before and after (never mid-flight - the in-motion read is
gripper-contaminated, a banked lesson). The blind carry gap that
defeated seven pairing signals last program does not exist here: the
bundle is the same object on both sides of its own burst, so pairing
is identity, not inference.

Cross-view: simA and simB share the clock by construction of the
write, so manipulations whose spans overlap are the same physical
event; the bundles that meet in a merged event are the same physical
object (union-find), which is what makes slots consistent across
views without any appearance matching.

Push vs carry: a carry arcs (the block is lifted and set down), a
push slides straight. Arc height normalised by path length, cut
fitted from the corpus.

    python native/events.py [--store lake/sim_chains] [--limit N]
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

VER = "v1"
REST_FR = 5          # frames of rest to average a position (1 s @5fps)


def scratch():
    from track import SCRATCH
    return SCRATCH


def load_bundles(store_name, ver=VER):
    p = scratch() / f"native_bundles_{ver}_{store_name}.npz"
    z = np.load(p, allow_pickle=True)
    data = {tuple(k): v for k, v in z["data"]}
    return data, z["cuts"]


def track_positions(store_name, ep, sv):
    from track import cache_dir
    z = np.load(cache_dir(store_name) / f"{ep:05d}_{sv}.npz")
    return z["tracks"].astype(np.float32), z["vis"], z["ts"]


def rest_pos(tracks, members, i0, i1):
    """Median position of the bundle's points over [i0,i1)."""
    i0 = max(int(i0), 0)
    i1 = min(int(i1), len(tracks))
    if i1 <= i0:
        return None
    P = tracks[i0:i1][:, members, :]
    return np.median(P.reshape(-1, 2), axis=0)


def raw_events(store_name, data, limit=0):
    """Per (episode, view, bundle, burst) -> one manipulation."""
    from tqdm import tqdm
    out = []
    keys = sorted(data)
    for (ep, sv) in tqdm(keys, desc="events", unit="view"):
        d = data[(ep, sv)]
        tracks, vis, ts = track_positions(store_name, ep, sv)
        labels = d["labels"]
        # burst indices are on the velocity grid (offset by WIN/2)
        from objects import WIN
        for b, spans in d["bursts"].items():
            members = np.where(labels == b)[0]
            if len(members) < 2:
                continue
            for (s, e) in spans:
                fs = int(s) + WIN // 2
                fe = int(e) + WIN // 2 + WIN
                p0 = rest_pos(tracks, members, fs - REST_FR, fs)
                p1 = rest_pos(tracks, members, fe, fe + REST_FR)
                if p0 is None or p1 is None:
                    continue
                mid = tracks[max(fs, 0):min(fe, len(tracks))][:, members, :]
                mid = np.median(mid, axis=1) if len(mid) else None
                arc = 0.0
                if mid is not None and len(mid) > 2:
                    d01 = p1 - p0
                    L = float(np.linalg.norm(d01)) + 1e-6
                    nrm = np.array([-d01[1], d01[0]]) / L
                    arc = float(np.abs((mid - p0) @ nrm).max())
                out.append(dict(
                    ep=int(ep), sv=str(sv), bundle=int(b),
                    t_dep=int(ts[min(max(fs, 0), len(ts) - 1)]),
                    t_arr=int(ts[min(max(fe, 0), len(ts) - 1)]),
                    p_dep=(float(p0[0]), float(p0[1])),
                    p_arr=(float(p1[0]), float(p1[1])),
                    disp=float(np.linalg.norm(p1 - p0)),
                    arc=arc, n=len(members)))
    return out


def fuse_views(evs):
    """Same physical event = overlapping spans in the two views.
    Bundles meeting in a merged event are the same object."""
    by_ep = defaultdict(list)
    for e in evs:
        by_ep[e["ep"]].append(e)
    uf = {}

    def find(a):
        while uf.get(a, a) != a:
            a = uf[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[ra] = rb

    fused = {}
    for ep, lst in by_ep.items():
        lst.sort(key=lambda e: e["t_dep"])
        groups = []
        for e in lst:
            hit = None
            for g in groups:
                if e["sv"] in {x["sv"] for x in g}:
                    continue
                a0, a1 = e["t_dep"], e["t_arr"]
                b0 = min(x["t_dep"] for x in g)
                b1 = max(x["t_arr"] for x in g)
                ov = min(a1, b1) - max(a0, b0)
                if ov > 0.5 * min(a1 - a0, b1 - b0):
                    hit = g
                    break
            if hit is None:
                groups.append([e])
            else:
                hit.append(e)
        for g in groups:
            for x in g[1:]:
                union((ep, g[0]["sv"], g[0]["bundle"]),
                      (ep, x["sv"], x["bundle"]))
        fused[ep] = groups
    return fused, find


def tokens(fused, find, arc_cut, disp_q):
    """Chain tokens in the exact format chain_qbe.align expects."""
    out = {}
    for ep, groups in fused.items():
        rows = []
        for g in groups:
            t0 = int(np.median([x["t_dep"] for x in g]))
            t1 = int(np.median([x["t_arr"] for x in g]))
            disp = float(np.max([x["disp"] for x in g]))
            arc = float(np.max([x["arc"] / max(x["disp"], 1.0)
                                for x in g]))
            obj = find((ep, g[0]["sv"], g[0]["bundle"]))
            rows.append((t0, t1, obj, disp, arc))
        rows.sort()
        slot_of, toks, prev = {}, [], None
        for t0, t1, obj, disp, arc in rows:
            if obj not in slot_of:
                slot_of[obj] = len(slot_of)
            if prev is not None:
                gap = max((t0 - prev) / 1e9, 0.0)
                toks.append((("G", int(min(gap / 4.0, 2)), 0), -1,
                             0.0, None))
            push = arc < arc_cut
            kind = ("P", 0, 0) if push else \
                ("M", int(np.searchsorted(disp_q, disp)), 0)
            toks.append((kind, slot_of[obj], (t1 - t0) / 1e9, None))
            prev = t1
        out[ep] = toks
    return out


def build(store_name, limit=0):
    data, cuts = load_bundles(store_name)
    if limit:
        data = {k: v for k, v in data.items() if k[0] < limit}
    evs = raw_events(store_name, data, limit)
    print(f"raw manipulations: {len(evs)} "
          f"({len(evs)/max(len(data),1):.1f}/view)")
    fused, find = fuse_views(evs)
    n = [len(g) for g in fused.values()]
    print(f"fused manipulations: mean {np.mean(n):.2f}/episode "
          f"(true 3.5 on sim_chains)")
    from chain_delta import k_means_1d
    ratios = np.array([e["arc"] / max(e["disp"], 1.0) for e in evs])
    r = ratios[np.isfinite(ratios) & (ratios > 0)]
    lc = k_means_1d(np.log10(r + 1e-3), 2)
    arc_cut = 10 ** ((lc[0] + lc[1]) / 2) - 1e-3
    disps = np.array([e["disp"] for e in evs])
    disp_q = np.percentile(disps[disps > 0], [33, 66])
    print(f"  fitted arc_cut {arc_cut:.3f} (push<cut)  "
          f"disp terciles {np.round(disp_q,1)}")
    return evs, fused, find, tokens(fused, find, arc_cut, disp_q)


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv \
        else 0
    evs, fused, find, toks = build(store.name, limit)
    np.savez_compressed(
        scratch() / f"native_events_{VER}_{store.name}.npz",
        evs=np.array(evs, object),
        toks=np.array(sorted(toks.items()), object))
    for e in sorted(toks)[:6]:
        print(f"  ep{e}: " + " ".join(f"{k[0]}{s}" for k, s, _, _
                                      in toks[e]))
    print(f"saved -> native_events_{VER}_{store.name}.npz")


if __name__ == "__main__":
    main()
