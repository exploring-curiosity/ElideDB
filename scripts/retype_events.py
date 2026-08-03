"""Re-TYPE events from the AGENT'S OWN motion geometry. No text, ever.

MEASURED OUTCOME (2026-08-03): a NEGATIVE result, kept as the record.
Agent-kinematic types scored 0.24-0.28 mean yield at chain retrieval
vs 0.34 for the original screen-space mover types, and the store was
restored to those (events v12). Two reasons, both instructive: the
original tokens include the contact/release BOUNDARY kinds whose
alternation carries alignment structure this descriptor collapses;
and HDBSCAN types only the dense 30% of the smooth kinematic manifold,
so most tokens are nearest-centroid quantisations of weak cores. The
typing lever remains open - the next attempt should token-ise the
hold-state ALTERNATION itself (acquire/carry/release segments from the
d_hold signature) rather than cluster whole-event profiles.

Original rationale: the chain benchmark's oracle (yield 1.00 from true
event tokens) attributes its whole gap to token quality, and the
shipped types - screen-space motion profiles of the weakly-bound mover
- sit at majority-baseline purity. This retypes every event from the
one element that is reliably bound (agent_object_id: 100%): the
agent's kinematic profile over the event span.

Everything stays inside the no-text rule: features are geometry from
the trajectories table, per-episode normalised (cameras differ per
episode); types are DISCOVERED ids (tN) via the same corpus-fitted
discover(); profiles are physical numbers. The truth sidecar is never
read - grading happens in chain_qbe/verify_sim_store, not here.

Descriptor per event (agent samples in the padded span):
    dur_s        span length
    net_dx/dy    net motion, episode-normalised
    path_h       horizontal path length, episode-normalised
    depth_in     descent from entry to the lowest point
    rise_out     ascent from the lowest point to exit
    v_pos        when the lowest point happens (0..1 in span)
    bottom_pct   how LOW the lowest point is vs the agent's whole
                 episode (a tower-top stop is higher than a table stop)
    hold_frac    fraction of span with a participant in agent contact
    onset/offset first/last contact position in span (1.0/0.0 if none)
    d_hold       holding at exit minus holding at entry - the signed
                 signature of acquiring vs releasing

    python scripts/retype_events.py [--store lake/sim_chains]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                      # noqa: E402
from elidedb.transitions import discover                       # noqa: E402

PAD_NS = int(0.2e9)
FIELDS = ("dur_s", "net_dx", "net_dy", "path_h", "depth_in", "rise_out",
          "v_pos", "bottom_pct", "hold_frac", "onset", "offset", "d_hold")


def main():
    argv = sys.argv
    store = ROOT / (argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/sim_chains")
    db = Store.open(str(store))

    tr = db.table("trajectories").scan().select(
        ["stream", "ts", "px", "py", "contact", "is_agent"])
    d = tr.to_pydict()
    ag = defaultdict(list)              # stream -> agent (ts, x, y)
    touch = defaultdict(set)            # stream -> ts with any contact
    for i in range(len(d["ts"])):
        s, ts_ = str(d["stream"][i]), int(d["ts"][i])
        if d["is_agent"][i]:
            ag[s].append((ts_, float(d["px"][i]), float(d["py"][i])))
        elif d["contact"][i]:
            touch[s].add(ts_)
    stats = {}
    for s, rows in ag.items():
        rows.sort()
        y = np.array([r[2] for r in rows], np.float32)
        x = np.array([r[1] for r in rows], np.float32)
        stats[s] = (np.array([r[0] for r in rows], np.int64), x, y,
                    float(x.std() + 1e-6), float(y.std() + 1e-6),
                    np.sort(y))

    ev = db.table("events").scan()
    e = ev.to_pydict()
    n = len(e["ts"])
    D, di = [], []
    for j in range(n):
        s = str(e["stream"][j])
        if s not in stats:
            continue
        t, x, y, sx, sy, ysorted = stats[s]
        a = int(e["ev_t0"][j]) - PAD_NS
        b = int(e["ev_t1"][j]) + PAD_NS
        lo, hi = np.searchsorted(t, a), np.searchsorted(t, b)
        if hi - lo < 4:
            continue
        ts_, xs, ys = t[lo:hi], x[lo:hi], y[lo:hi]
        held = np.array([tt in touch[s] for tt in ts_], np.float32)
        k = int(ys.argmax())            # image +y is down: max = lowest
        m = len(ys) - 1
        onset = (float(np.argmax(held) / m) if held.any() else 1.0)
        offset = (float((m - np.argmax(held[::-1])) / m)
                  if held.any() else 0.0)
        D.append([
            (b - a - 2 * PAD_NS) / 1e9,
            (xs[-1] - xs[0]) / sx,
            (ys[-1] - ys[0]) / sy,
            float(np.abs(np.diff(xs)).sum()) / sx,
            (ys[k] - ys[0]) / sy,
            (ys[k] - ys[-1]) / sy,
            k / m,
            float(np.searchsorted(ysorted, ys[k]) / len(ysorted)),
            float(held.mean()),
            onset,
            offset,
            float(held[-1] - held[0]),
        ])
        di.append(j)
    print(f"{len(di)}/{n} events with agent coverage", flush=True)

    # cluster on the COMPACT hold+shape subset: the full 12-d descriptor
    # is a smooth manifold HDBSCAN calls 100% noise (measured), while
    # hold-signature + vertical shape (+ how LOW the stop is) carries
    # the density. The rest of the descriptor still ships in the
    # profile for the record.
    import os
    sub = os.environ.get("ELIDEDB_RETYPE_FEATS",
                         "d_hold,hold_frac,depth_in,rise_out,bottom_pct")
    idx = [FIELDS.index(c) for c in sub.split(",")]
    Dfull = np.asarray(D, np.float32)
    lab, cent, tinfo = discover(Dfull[:, idx])
    # COMPLETE the assignment: HDBSCAN types only dense cores (70%
    # unclustered here), and an untyped chain token carries nothing.
    # Types stay DISCOVERED; the remainder is quantised to the nearest
    # type centroid in the standardised space.
    if len(cent):
        mu = np.asarray(tinfo["mu"], np.float32)
        sd = np.asarray(tinfo["sd"], np.float32)
        Z = (Dfull[:, idx] - mu) / sd
        near = np.linalg.norm(Z[:, None, :] - cent[None, :, :],
                              axis=2).argmin(1)
        lab = np.where(lab < 0, near, lab).astype(np.int32)
        tinfo["sizes"] = [int((lab == i).sum()) for i in range(len(cent))]
    kinds = list(e["kind"])
    for j, k_ in zip(di, lab):
        kinds[j] = f"t{int(k_)}" if k_ >= 0 else ""
    for j in set(range(n)) - set(di):
        kinds[j] = ""
    e["kind"] = kinds
    out = pa.table({c: pa.array(e[c], ev.schema.field(c).type)
                    for c in ev.column_names})
    db.table("events").replace(
        out, kind="events",
        meta={"builder": "retype_events",
              "descriptor": "agent-kinematic " + ",".join(FIELDS),
              "types": int(len(cent))})

    if len(cent):
        import json
        mu, sd = np.asarray(tinfo["mu"]), np.asarray(tinfo["sd"])
        prof = [dict(zip(FIELDS, np.round(c * sd + mu, 3).tolist()))
                for c in np.asarray(cent)]
        db.table("transition_types").replace(pa.table({
            "ts": pa.array([min(e["ts"])] * len(cent), pa.int64()),
            "t1": pa.array([max(e["t1"])] * len(cent), pa.int64()),
            "type_id": pa.array(list(range(len(cent))), pa.int32()),
            "n_members": pa.array(tinfo["sizes"], pa.int32()),
            "centroid": pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(cent).reshape(-1),
                         pa.float32()), cent.shape[1]),
        }), kind="index", meta={"discovered": True,
                                "descriptor": "agent-kinematic",
                                "profile": json.dumps(prof)})
    typed = sum(1 for k_ in kinds if k_.startswith("t"))
    print({"events": n, "typed": typed, "types": int(len(cent)),
           "sizes": tinfo.get("sizes")})


if __name__ == "__main__":
    main()
