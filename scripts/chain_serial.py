"""SERIALITY PARSE: one arm = the event stream is a grammar.

Local pairing rules (LIFO + serial context) were brittle both ways -
recovering missed events DROPPED slot agreement (0.88 -> 0.84,
measured), because every spurious event corrupts the stacks. But the
corpus's physics is a global constraint: with ONE arm, true events
alternate dep -> arr (a carry), with standalone pushes between, and
NOTHING real happens mid-carry. So the decoder is a max-weight parse:
choose the alternating subsequence that keeps the most attested
events, paying to skip attested ones and to stretch carries beyond
the fitted gap; ambiguous both-occupied events (stack arrivals,
unstack departures, pushes all read "occupied->occupied") take
whatever ROLE the parse demands. Junk needs no separate veto - it is
whatever the parse skips - but two vetoes cheapen it: the encoder
AGENT gallery (arm states that survived persistence) and CROSS-VIEW
fusion (both cameras share the clock; a real event is attested twice).

Slots ride on spot continuity over the CLEAN sequence: an object next
departs from where it last arrived; per-spot LIFO handles stacks.

    python scripts/chain_serial.py            # parse + bench + grades
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import chain_qbe                                               # noqa: E402
from chain_moves import bench, otsu                            # noqa: E402
import chain_delta as cd                                       # noqa: E402

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-sudharshanramesh-Studies-"
    "MyProjects-StreetDex/98dca676-44e6-4cc3-8fda-c9744a9c5fb3/"
    "scratchpad")
CACHE = Path(os.environ.get("ELIDEDB_DELTA_CACHE",
                            str(SCRATCH / "delta_events.npz")))


def load_events():
    z = np.load(CACHE, allow_pickle=True)
    events = [list(e) for e in z["events"]]
    crops = list(z["crops"])
    agents = list(z["agents"])
    return events, crops, agents


def agent_veto(events, crops, agents):
    """Arm states that survived persistence (episode-end parks): veto
    events whose either side matches the AGENT gallery in the store's
    kind-encoder space at an otsu-fitted bar."""
    ec = SCRATCH / "serial_emb.npz"
    if ec.exists():
        z = np.load(ec)
        E, A = z["E"], z["A"]
    else:
        from elidedb import Store, dinov3
        db = Store.open(str(ROOT / "lake/sim_chains"))
        mid = (db.table("objkind_vectors").state().meta
               or {}).get("encoder",
                          "facebook/dinov3-vits16-pretrain-lvd1689m")
        print(f"embedding {len(crops):,} crops + {len(agents)} agent "
              f"gallery ({mid.split('/')[-1]})...", flush=True)
        dinov3._load(mid)
        from chain_ledger import embed_crops
        E = embed_crops(crops, mid)
        A = embed_crops(agents, mid)
        np.savez(ec, E=E, A=A)
    m = (E @ A.T).max(1)
    bar = otsu(m)
    veto = np.zeros(len(events), bool)
    for i, ev in enumerate(events):
        veto[i] = m[ev[6]] >= bar or m[ev[7]] >= bar
    print(f"agent veto: bar {bar:.2f}, {int(veto.sum())}/{len(events)}"
          f" events", flush=True)
    return veto


def fuse(events, dirs, junk, veto):
    """Cross-view fusion on the shared clock: same episode, same
    direction, |dt| <= 1s = the same physical event seen twice.
    conf = per-view area/median, summed over attestations."""
    med_area = np.median([e[5] for e in events])
    by_ep = defaultdict(list)
    for i, (ev, dr, jk, vt) in enumerate(zip(events, dirs, junk,
                                             veto)):
        if jk or vt:
            continue
        pers = int(ev[12]) if len(ev) > 12 else 1
        by_ep[int(ev[0])].append(
            dict(t=int(ev[2]), dir=int(dr),
                 conf=float(math.sqrt(ev[5] / med_area))
                 * (1.6 if pers else 1.0),
                 pos={str(ev[1]): (float(ev[3]), float(ev[4]))},
                 views=1))
    fused = {}
    for ep, evs in by_ep.items():
        evs.sort(key=lambda d: d["t"])
        out = []
        for e in evs:
            hit = None
            for o in out:
                if abs(e["t"] - o["t"]) <= int(1.0e9) \
                        and o["dir"] == e["dir"] \
                        and not (set(o["pos"]) & set(e["pos"])):
                    hit = o
                    break
            if hit is None:
                out.append(e)
            else:
                n = hit["views"]
                hit["t"] = (hit["t"] * n + e["t"]) // (n + 1)
                hit["conf"] += e["conf"]
                hit["pos"].update(e["pos"])
                hit["views"] += 1
        fused[ep] = out
    return fused


def parse(evs, g_med, alpha=1.0, beta=0.45, gamma=1.2, wmax_s=9.0):
    """Max-weight alternating parse. Roles: dir=-1 dep only, dir=1
    arr only, dir=0 either or push. Transitions from each IDLE
    position i (0-based over time-sorted events):
      skip e_i                    -alpha*conf_i
      push at e_i (dir 0)         +conf_i
      manip (e_i dep, e_j arr)    +conf_i+conf_j - gamma*|log(gap/
                                  g_med)| - alpha*sum(conf skipped
                                  between)
      lone arr / lone dep         +beta*conf_i
    Returns the chosen manipulations."""
    n = len(evs)
    best = [(-1e18, None)] * (n + 1)
    best[0] = (0.0, None)
    for i in range(n):
        sc, _ = best[i]
        if sc < -1e17:
            continue
        e = evs[i]
        # skip
        cand = sc - alpha * e["conf"]
        if cand > best[i + 1][0]:
            best[i + 1] = (cand, (i, "skip", None))
        # push
        if e["dir"] == 0:
            cand = sc + e["conf"]
            if cand > best[i + 1][0]:
                best[i + 1] = (cand, (i, "push", None))
        # lone arr / lone dep
        if e["dir"] in (0, 1):
            cand = sc + beta * e["conf"]
            if cand > best[i + 1][0]:
                best[i + 1] = (cand, (i, "lone_arr", None))
        if e["dir"] in (0, -1):
            cand = sc + beta * e["conf"]
            if cand > best[i + 1][0]:
                best[i + 1] = (cand, (i, "lone_dep", None))
        # manip: e_i departs, e_j arrives
        if e["dir"] in (0, -1):
            mid_cost = 0.0
            for j in range(i + 1, n):
                a = evs[j]
                gap = (a["t"] - e["t"]) / 1e9
                if gap > wmax_s:
                    break
                if a["dir"] in (0, 1):
                    pen = gamma * abs(math.log(max(gap, 0.2)
                                               / g_med))
                    cand = sc + e["conf"] + a["conf"] - pen \
                        - alpha * mid_cost
                    if cand > best[j + 1][0]:
                        best[j + 1] = (cand, (i, "manip", j))
                mid_cost += a["conf"]
    # backtrack
    out = []
    k = n
    while k > 0:
        _, tr = best[k]
        if tr is None:
            k -= 1
            continue
        i, kind, j = tr
        if kind == "manip":
            out.append((evs[i]["t"], evs[j]["t"], "M",
                        evs[i], evs[j]))
        elif kind == "push":
            out.append((evs[i]["t"], evs[i]["t"], "P",
                        evs[i], evs[i]))
        elif kind == "lone_arr":
            out.append((evs[i]["t"], evs[i]["t"], "A",
                        None, evs[i]))
        elif kind == "lone_dep":
            out.append((evs[i]["t"], evs[i]["t"], "D",
                        evs[i], None))
        k = i
    out.sort(key=lambda r: (r[0], r[1], r[2]))
    return out


def fit_gap(fused):
    """Median dep->next-arr gap over unambiguous adjacent pairs."""
    gaps = []
    for evs in fused.values():
        for i in range(len(evs) - 1):
            if evs[i]["dir"] == -1 and evs[i + 1]["dir"] == 1:
                g = (evs[i + 1]["t"] - evs[i]["t"]) / 1e9
                if 0.2 < g < 12:
                    gaps.append(g)
    return float(np.median(gaps)) if gaps else 2.5


def slots_and_tokens(parsed, med_r):
    """Spot-continuity object chains over the parsed sequence, then
    tokens: M(slot, travel-q) / P(slot) with G gaps."""
    mans_by_ep = {}
    for ep, ms in parsed.items():
        spots = defaultdict(list)          # (view, spot_i) -> stack
        centers = defaultdict(list)        # view -> [xy]
        nxt = [0]
        uf = {}

        def find(a):
            while uf.get(a, a) != a:
                a = uf[a]
            return a

        def union(a, b):
            uf[find(a)] = find(b)

        def new_tok():
            nxt[0] += 1
            return nxt[0] - 1

        def spot_of(sv, x, y):
            for si, (px, py) in enumerate(centers[sv]):
                if np.hypot(x - px, y - py) < med_r:
                    centers[sv][si] = ((px + x) / 2, (py + y) / 2)
                    return si
            centers[sv].append((x, y))
            return len(centers[sv]) - 1

        def pop_at(e):
            toks = []
            for sv, (x, y) in e["pos"].items():
                st = spots[(sv, spot_of(sv, x, y))]
                toks.append(st.pop() if st else new_tok())
            tok = toks[0]
            for t2 in toks[1:]:
                union(tok, t2)
            return tok

        def push_at(e, tok):
            for sv, (x, y) in e["pos"].items():
                spots[(sv, spot_of(sv, x, y))].append(tok)

        rows = []
        for t0, t1, kind, d, a in ms:
            if kind == "M":
                tok = pop_at(d)
                push_at(a, tok)
                trav = -1.0
                for sv in d["pos"]:
                    if sv in a["pos"]:
                        trav = max(trav, float(np.hypot(
                            a["pos"][sv][0] - d["pos"][sv][0],
                            a["pos"][sv][1] - d["pos"][sv][1])))
                rows.append((t0, t1, tok, trav, False))
            elif kind == "P":
                tok = pop_at(d)
                push_at(d, tok)
                rows.append((t0, t1, tok, 0.0, True))
            elif kind == "A":
                tok = new_tok()
                push_at(a, tok)
                rows.append((t0, t1, tok, -1.0, False))
            elif kind == "D":
                pop_at(d)          # departs the scene of record
        mans_by_ep[ep] = [(t0, t1, find(tok), tv, p)
                          for t0, t1, tok, tv, p in rows]
    return mans_by_ep


def grade(mans, tag):
    import pyarrow.parquet as pq
    from elidedb import Store
    db = Store.open(str(ROOT / "lake/sim_chains"))
    epd = db.table("episodes").scan().to_pydict()
    ep0 = {int(e): int(t) for e, t in zip(epd["episode_index"],
                                          epd["ts"])}
    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    tmpl = {int(e): tm for e, tm in zip(t["episode"], t["template"])}
    tru = defaultdict(list)
    for e, p, b, a1 in zip(t["episode"], t["prim"], t["block"],
                           t["t1"]):
        if str(p) != "pick":
            tru[int(e)].append((float(a1), str(b)))
    ns = [len(v) for v in mans.values()]
    hist = np.bincount(np.clip(ns, 0, 8), minlength=9)
    agree = tot = 0
    matched = 0
    for e, ms in mans.items():
        pairs, used = [], set()
        for t0, t1, tok, tv, p in sorted(ms):
            te = (t1 - ep0[e]) / 1e9
            cs = [(abs(te - a1), j) for j, (a1, b) in
                  enumerate(tru[e]) if j not in used
                  and abs(te - a1) < 3.5]
            if cs:
                _, j = min(cs)
                used.add(j)
                pairs.append((tok, tru[e][j][1]))
        matched += len(pairs)
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                tot += 1
                agree += (pairs[i][0] == pairs[j][0]) == \
                    (pairs[i][1] == pairs[j][1])
    print(f"[{tag}] manips {np.mean(ns):.2f}/ep (true 3.5)  "
          f"hist {list(hist)}  matched {matched}  "
          f"slot-agree {agree}/{tot} = {agree/max(tot,1):.2f}",
          flush=True)
    return tmpl


def tokenise(mans):
    trav = np.array([m[3] for v in mans.values() for m in v
                     if m[3] > 0])
    tq = np.percentile(trav, [33, 66]) if len(trav) else [1, 2]
    out = {}
    for e, ms in mans.items():
        slot_of, toks, prev = {}, [], None
        for t0, t1, tok, d, push in sorted(ms):
            if tok not in slot_of:
                slot_of[tok] = len(slot_of)
            if prev is not None:
                gap = max((t0 - prev) / 1e9, 0.0)
                toks.append((("G", int(min(gap / 4.0, 2)), 0),
                             -1, 0.0, None))
            kind = ("P", 0, 0) if push else \
                ("M", int(np.searchsorted(tq, d)) if d >= 0 else 1, 0)
            toks.append((kind, slot_of[tok], (t1 - t0) / 1e9, None))
            prev = t1
        out[e] = toks
    return out


def main():
    events, crops, agents = load_events()
    dirs, junk = cd.classify_events(events)
    veto = agent_veto(events, crops, agents)
    fused = fuse(events, dirs, junk, veto)
    nf = [len(v) for v in fused.values()]
    print(f"fused events: {np.mean(nf):.1f}/ep  "
          f"({np.mean([sum(x['views'] > 1 for x in v) for v in fused.values()]):.1f} double-attested)",
          flush=True)
    g_med = fit_gap(fused)
    print(f"fitted carry gap median: {g_med:.2f}s", flush=True)
    parsed = {ep: parse(evs, g_med) for ep, evs in fused.items()}
    med_r = 0.9 * cd.DS * math.sqrt(np.median([e[5]
                                               for e in events]))
    mans = slots_and_tokens(parsed, med_r)
    tmpl = grade(mans, "serial")
    seqs = tokenise(mans)
    chain_qbe.W_KIND = 0.5
    dev = ("swap", "precarious", "push_then_build",
           "build_unstack_move")
    hold = ("relocate_build", "two_sites_merge")
    bench(seqs, tmpl, "DEV serial", dev, w_slot=1.0)
    bench(seqs, tmpl, "HOLDOUT serial", hold, w_slot=1.0)


if __name__ == "__main__":
    main()
