"""REST-LEDGER chains: object permanence as bookkeeping. Route 2.

Every lab-side route died on the same primitive - identifying the
mover WHILE IT MOVES (14-40% reliable everywhere measured). The
measured asymmetry points the other way: AT REST, blocks are reliable
(positions solid, colours 0.78 from one crop, better from medians).
So this never looks at motion. It tracks the WORLD'S REST STATE:

    entry      a block resting at a spot: (position, colour, span).
               Entries persist across track breaks - same place +
               same colour = the same entry continuing - so tracker
               churn becomes a no-op instead of a fragment.
    itinerary  per colour, the time-ordered sequence of its entries.
               A manipulation IS a transition: entry k ends (pickup),
               entry k+1 begins elsewhere (set-down). The carry
               between them is never observed and never needs to be -
               the blind gap that defeated seven pairing signals is
               dissolved by object permanence.

Chain tokens = all transitions ordered by departure time:
M(colour-slot, travel_q) with G gaps. Everything fitted from the
corpus (motion/rest thresholds from chain_2v, colour cut by Otsu over
within-episode entry pairs); colours are pixels; no models, no text,
no truth. Cross-view: entries merged by colour + time overlap.

    python scripts/chain_ledger.py [--store lake/sim_chains]
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
import chain_qbe                                               # noqa: E402
from chain_moves import bench, otsu                            # noqa: E402
from chain_2v import Cropper, view_segments                    # noqa: E402

MIN_REST_S = 0.8


def rest_intervals(series, thr):
    """Maximal resting stretches per (e, sv, oid) series."""
    out = []
    for (e, sv, oid), (t, x, y, nd) in series.items():
        i0 = None
        for i in range(len(t) + 1):
            resting = i < len(t) and nd[i] < thr
            if resting and i0 is None:
                i0 = i
            elif not resting and i0 is not None:
                if (t[i - 1] - t[i0]) / 1e9 >= MIN_REST_S:
                    out.append((e, sv,
                                float(np.median(x[i0:i])),
                                float(np.median(y[i0:i])),
                                int(t[i0]), int(t[i - 1])))
                i0 = None
    return out


def spot_chain(iv, med_diag):
    """Chain rest intervals into SPOT runs per (episode, view): same
    place = the same sitting block continuing across track breaks.
    Colour is checked at the ENTRY level afterwards; position alone
    chains here, and a different-coloured newcomer at the same spot is
    split later by its colour."""
    by_ev = defaultdict(list)
    for e, sv, x, y, a, b in iv:
        by_ev[(e, sv)].append((a, b, x, y))
    entries = []
    for (e, sv), rows in by_ev.items():
        rows.sort()
        open_ = []                     # [x, y, a, b, members]
        for a, b, x, y in rows:
            hit = None
            for en in open_:
                if np.hypot(x - en[0], y - en[1]) < 0.8 * med_diag \
                        and a - en[3] < int(3.0e9):
                    hit = en
                    break
            if hit is None:
                open_.append([x, y, a, b, [(a, b, x, y)]])
            else:
                n = len(hit[4])
                hit[0] = (hit[0] * n + x) / (n + 1)
                hit[1] = (hit[1] * n + y) / (n + 1)
                hit[3] = max(hit[3], b)
                hit[4].append((a, b, x, y))
        for x, y, a, b, mem in open_:
            entries.append((e, sv, x, y, a, b, mem))
    return entries


def crop_at(cr, sv, pos, ts_):
    im = cr.frame(sv, int(ts_))
    if im is None:
        return None
    h = int(cr.med_diag * 0.5)
    x0 = max(int(pos[0]) - h, 0)
    y0 = max(int(pos[1]) - h, 0)
    c = im[y0:int(pos[1]) + h, x0:int(pos[0]) + h]
    return c if c.size >= 3 * 12 * 12 else None


def embed_crops(crops, mid):
    from elidedb import dinov3
    E = dinov3.embed(crops, mid=mid)
    E = np.asarray(E, np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True),
                          1e-8)


def entry_embeddings(db, entries, agents, med_diag):
    """VISION-NATIVE entry signatures + junk galleries, all in ONE
    encoder space (the store's kind encoder). No hand features: the
    encoder itself distinguishes a green block from a red block from a
    blue cylinder - that is the rule's point - and junk is whatever
    matches the AGENT or BACKGROUND galleries by the store's own
    fitted standard.
    """
    mid = (db.table("objkind_vectors").state().meta
           or {}).get("encoder",
                      "facebook/dinov3-vits16-pretrain-lvd1689m")
    from elidedb import dinov3
    print(f"loading kind encoder {mid.split('/')[-1]} (fp32)...",
          flush=True)
    dinov3._load(mid)
    cr = Cropper(db, med_diag)
    rng = np.random.default_rng(0)

    # entry crops (up to 3 rest samples each)
    from tqdm import tqdm
    crops, owner = [], []
    for i, (e, sv, x, y, a, b, mem) in enumerate(
            tqdm(entries, desc="entry-crops", mininterval=5)):
        for tp in np.linspace(a, b, min(3, max(1, len(mem))))                 .astype(np.int64):
            best = min(mem, key=lambda m: min(abs(m[0] - tp),
                                              abs(m[1] - tp)))
            ts_ = best[0] if abs(best[0] - tp) < abs(best[1] - tp) \
                else best[1]
            c = crop_at(cr, sv, (x, y), int(ts_))
            if c is not None:
                crops.append(c)
                owner.append(i)

    # galleries: agent (its own box centres) and background (random
    # positions far from every entry) - crops of the same size, same
    # encoder, cuts fitted from the match distributions
    ag_crops = []
    keys = sorted(agents)
    while len(ag_crops) < 200 and keys:
        e, sv = keys[rng.integers(len(keys))]
        rows = agents[(e, sv)]
        ts_, x0, y0, x1, y1 = rows[rng.integers(len(rows))]
        c = crop_at(cr, sv, ((x0 + x1) / 2, (y0 + y1) / 2), ts_)
        if c is not None:
            ag_crops.append(c)
    ent_pos = defaultdict(list)
    for e, sv, x, y, a, b, mem in entries:
        ent_pos[(e, sv)].append((x, y))
    bg_crops = []
    ent_keys = sorted(ent_pos)
    while len(bg_crops) < 200 and ent_keys:
        e, sv = ent_keys[rng.integers(len(ent_keys))]
        rows = agents.get((e, sv))
        if not rows:
            continue
        ts_ = rows[rng.integers(len(rows))][0]
        im = cr.frame(sv, int(ts_))
        if im is None:
            continue
        H, W = im.shape[:2]
        x, y = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
        if any(np.hypot(x - px, y - py) < 2.0 * med_diag
               for px, py in ent_pos[(e, sv)]):
            continue
        c = crop_at(cr, sv, (x, y), int(ts_))
        if c is not None:
            bg_crops.append(c)

    print(f"embedding {len(crops):,} entry + {len(ag_crops)} agent + "
          f"{len(bg_crops)} background crops...", flush=True)
    E = embed_crops(crops, mid)
    A = embed_crops(ag_crops, mid)
    B = embed_crops(bg_crops, mid)
    V = np.zeros((len(entries), E.shape[1]), np.float32)
    n = np.zeros(len(entries), np.int32)
    for e_, o in zip(E, owner):
        V[o] += e_
        n[o] += 1
    ok = n > 0
    V[ok] /= n[ok, None]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)

    a_match = (V @ A.T).max(1)
    b_match = (V @ B.T).max(1)
    bar_a = otsu(a_match[ok])
    bar_b = otsu(b_match[ok])
    keep = ok & (a_match < bar_a) & (b_match < bar_b)
    print(f"vision junk filter: {int(keep.sum()):,} of {int(ok.sum()):,}"
          f" kept (agent bar {bar_a:.2f}, background bar {bar_b:.2f})")
    return V, keep


def ledgers(db):
    segs, series, med_diag, thr, agents = view_segments(db)
    iv = rest_intervals(series, thr)
    print(f"rest intervals: {len(iv):,}")
    entries = spot_chain(iv, med_diag)
    print(f"spot entries: {len(entries):,} "
          f"({len(entries)/150:.1f}/episode-view)")
    import os
    cache = Path(os.environ.get("ELIDEDB_LEDGER_CACHE",
                                "/tmp/ledger_emb.npz"))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        V, keep = z["V"], z["keep"]
        print(f"entry embeddings from cache ({int(keep.sum())} kept)")
    else:
        V, keep = entry_embeddings(db, entries, agents, med_diag)
        np.savez(cache, V=V, keep=keep)
    # slots: corpus-DISCOVERED kind codebook in the encoder's own
    # space, nearest-centroid completion (the fitted, vision-native
    # replacement for every colour mechanism this script ever had)
    from elidedb.transitions import discover
    kept_idx = np.where(keep)[0]
    lab_, cent, ti = discover(V[kept_idx])
    print(f"kind codebook: {ti.get('types', 0)} types, unclustered "
          f"{ti.get('unclustered_frac', 1):.2f}")
    if not len(cent):
        raise SystemExit("no kind clusters discovered")
    mu = np.asarray(ti["mu"], np.float32)
    sd = np.asarray(ti["sd"], np.float32)
    Z = (V[kept_idx] - mu) / sd
    near = np.linalg.norm(Z[:, None, :] - np.asarray(cent)[None, :, :],
                          axis=2).argmin(1)
    cid = np.where(lab_ >= 0, lab_, near)
    by_e = defaultdict(list)
    for i, c in zip(kept_idx, cid):
        e, sv, x, y, a, b, mem = entries[int(i)]
        by_e[int(e)].append(((e, sv, x, y, int(a), int(b), None),
                             int(c)))
    return by_e, med_diag


def transitions(by_e, med_diag):
    """Spells per (episode, kind) merged across views by time overlap;
    per-kind itineraries -> transitions."""
    seqs = {}
    for e, rows in by_e.items():
        spells = defaultdict(list)
        for r, c in rows:
            _, sv, x, y, a, b, _ = r
            lst = spells[c]
            hit = None
            for sp in lst:
                if a <= sp[1] + int(1.0e9) and b >= sp[0] - int(1.0e9):
                    hit = sp
                    break
            if hit is None:
                lst.append([int(a), int(b), float(x), float(y)])
            else:
                hit[0] = min(hit[0], int(a))
                hit[1] = max(hit[1], int(b))
        trans = []
        for c, lst in spells.items():
            lst.sort()
            for i in range(1, len(lst)):
                d = float(np.hypot(lst[i][2] - lst[i - 1][2],
                                   lst[i][3] - lst[i - 1][3]))
                trans.append((lst[i - 1][1], lst[i][0], c, d))
        trans.sort()
        seqs[e] = trans
    return seqs


def tokenise(seqs):
    trav = np.array([d for v in seqs.values() for *_ , d in v])
    tq = np.percentile(trav, [33, 66]) if len(trav) else [1, 2]
    out = {}
    for e, trans in seqs.items():
        toks = []
        prev = None
        for dep, arr, s_, d in trans:
            if prev is not None:
                gap = max((dep - prev) / 1e9, 0.0)
                toks.append((("G", int(min(gap / 4.0, 2)), 0),
                             -1, 0.0, None))
            toks.append((("M", int(np.searchsorted(tq, d)), 0),
                         s_, (arr - dep) / 1e9, None))
            prev = arr
        out[e] = toks
    return out


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))
    by_e, med_diag = ledgers(db)
    seqs_t = transitions(by_e, med_diag)
    print(f"episodes {len(seqs_t)}, mean transitions "
          f"{np.mean([len(v) for v in seqs_t.values()]):.1f}")
    seqs = tokenise(seqs_t)
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    tmpl = {}
    for e, tm in zip(t["episode"], t["template"]):
        tmpl[int(e)] = tm
    chain_qbe.W_KIND = 0.5
    dev = ("swap", "precarious", "push_then_build",
           "build_unstack_move")
    hold = ("relocate_build", "two_sites_merge")
    bench(seqs, tmpl, "DEV ledger", dev, w_slot=1.0)
    bench(seqs, tmpl, "HOLDOUT ledger", hold, w_slot=1.0)


if __name__ == "__main__":
    main()
