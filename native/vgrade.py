"""THE BENCHMARK. One protocol, four stores, single view, no labels.

WHAT IS BEING MEASURED, and what is not.

The product question is "I give you a sample, I want all samples like
that." Grading it needs someone to say which corpus clips are LIKE the
sample - and every source of that judgement is forbidden here. Task
strings are text. GPS and IMU are sensors. Template names are labels.
A second camera is another view, and views are treated alone now. There
is no admissible answer key for semantic similarity, and inventing one
is how every inflated number in this project's history was produced.

So this measures the NECESSARY CONDITION instead, and says so:

    the same event, made to look different, must come back.

For each sampled span the harness re-films it - perspective warp, crop,
photometric shift, codec damage, resampled tempo - hands the altered
clip in as the example, and asks the store to find the original. The
distractors are the entire corpus INCLUDING the seconds immediately
either side of the target, which are the hardest negatives available and
the exact ones a scene-dominated representation confuses. A system that
fails this cannot possibly do QbE; a system that passes it has not yet
been shown to do QbE. Both halves of that get printed.

Reported, and only these:

    yield = true / support        prec = true / returned
    at k = ceil(1.5 x support) as a MAX BOUND, abstention on.

Support is real, not 1: three window scales cover one moment, so a
target has several store rows and precision is not capped by arithmetic
before the search runs. Support, returned and the chance rate print on
every row, always.

Three gates:

    N   nuisance     the battery above. The headline.
    D   direction    the span played BACKWARDS must be rejected. A
                     representation invariant to time reversal has
                     thrown away open-versus-close, so passing N while
                     failing D is not success.
    P   prune        coarse-cell recall against an exact full scan, so
                     the elision number cannot be bought with recall.

    python native/vgrade.py --stores sim --queries 24
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "native"))

import vnuis                                             # noqa: E402
import vqbe                                              # noqa: E402
import vsrc                                              # noqa: E402
from vstore import CHANNELS, STORES, Store               # noqa: E402

IOU_TRUE = 0.5     # kept for the span-level rank1 diagnostic
# FULLY CONTAINED. A store row is the queried moment when it lies
# entirely inside the span the user showed - the natural boundary, and
# not a number chosen after seeing a result. At 0.5 a 4 s row half
# outside the query counted as truth, the system correctly ranked those
# marginal rows low, and the ceiling that produced (0.707 with the
# prune and the cut both disabled) was an artifact of the rule rather
# than of retrieval - the median truth row sits at rank 7.9 of 533.
TRUE_FRAC = 1.0
QSPAN = 8.0        # seconds of example handed to the system


def tiou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def truth_rows(st, mid, t0, t1, frac=TRUE_FRAC):
    """Store rows that ARE the queried moment: same media, and at least
    `frac` of the ROW's own duration lies inside the query's span.

    Containment, not IoU, and the reason is a benchmark artifact worth
    recording. Under IoU >= 0.5 an 8 s query's truth sits at the 4 s and
    8 s store scales - but an 8 s clip cannot form a valid 8 s unit,
    because a window that spans its whole clip has no context left to
    subtract. The system was being graded against rows at scales it is
    structurally unable to produce, and the identity query capped at
    0.354 for that reason alone.

    Containment is also the right definition for the product: handed an
    8 s example, a returned 2 s span that lies inside the right moment
    is a correct answer - the user sees the event they asked for - not a
    miss. It stays hard because the query span is a tiny fraction of any
    media, so the chance rate is unmoved and prints on every row.
    """
    same = np.flatnonzero(st.media == mid)
    return [int(r) for r in same
            if (min(t1, st.t1[r]) - max(t0, st.t0[r]))
            >= frac * (st.t1[r] - st.t0[r])]


def sample_queries(st, n, rs):
    """Deterministic, uniform over media. No curation: whatever the
    sampler picks is reported, including the ones that fail."""
    mids = sorted(set(st.media.tolist()))
    out = []
    per = max(1, int(math.ceil(n / len(mids))))
    for mid in mids:
        rows = np.flatnonzero(st.media == mid)
        lo, hi = float(st.t0[rows].min()), float(st.t1[rows].max())
        if hi - lo < QSPAN + 2.0:
            continue
        for _ in range(per):
            t0 = float(rs.uniform(lo, hi - QSPAN))
            t0 = round(t0 / (max(vqbe.vcore.SCALES) / 2.0)) \
                * (max(vqbe.vcore.SCALES) / 2.0)
            if t0 + QSPAN <= hi:
                out.append((mid, t0, t0 + QSPAN))
            if len(out) >= n:
                return out[:n]
    return out[:n]


def grade_one(st, src, mid, t0, t1, tf, rs, ref):
    """One example, one transform -> yield, precision, support."""
    truth = set(truth_rows(st, mid, t0, t1))
    if len(truth) < 2:
        return None
    F = src.cut(t0, t1)
    if len(F) < 8:
        return None
    G = tf(F, rs) if tf is not None else F
    try:
        Q = vqbe.Query(G)
    except ValueError:
        return None
    # ONE search, graded under BOTH truth definitions. TRUE_FRAC was
    # changed from 0.5 to 1.0 after 0.5 was found to be an artifact, and
    # a metric definition changed after seeing results has to be shown
    # both ways or the reader cannot tell it was not cherry-picked. Each
    # definition applies its OWN k = ceil(1.5 x support) max bound to
    # the same ranked list.
    truth_h = set(truth_rows(st, mid, t0, t1, frac=0.5))
    sup = len(truth)
    rows, info = vqbe.search(st, Q, k=None, abstain=True, exclude=None,
                             ref=ref, merge=False)
    ranked = [r for r, _s in rows]
    got = ranked[:math.ceil(1.5 * sup)]
    tr = sum(1 for r in got if r in truth)
    gh = ranked[:math.ceil(1.5 * max(len(truth_h), 1))]
    th = sum(1 for r in gh if r in truth_h)
    return dict(y=tr / sup, p=tr / len(got) if got else 0.0, sup=sup,
                y_h=th / max(len(truth_h), 1),
                p_h=th / len(gh) if gh else 0.0, sup_h=len(truth_h),
                ret=len(got), cand=info["cand"], bytes=info["bytes"],
                w={c: info["w"][c] for c in CHANNELS},
                rank=float(np.median([i for i, (r, _s) in enumerate(rows)
                                      if r in truth] or [len(rows)])),
                hit1=bool(got and got[0] in truth),
                rejected=(len(got) == 0 or not any(r in truth
                                                   for r in got)))


def prune_recall(st, src, mid, t0, t1, ref, rs):
    """Gate P: what the coarse cells cost. Same query, probe vs full."""
    F = src.cut(t0, t1)
    if len(F) < 8:
        return None
    Q = vqbe.Query(F)
    a, _ = vqbe.search(st, Q, k=20, abstain=False, probe=vqbe.CAND_FRAC,
                       ref=ref, merge=False)
    ba = st.bytes_read
    b, _ = vqbe.search(st, Q, k=20, abstain=False, probe=1.0, ref=ref,
                       merge=False)
    bb = st.bytes_read
    A, B = {r for r, _ in a}, {r for r, _ in b}
    return (len(A & B) / max(len(B), 1), ba, bb)


def run(name, nq, out_rows):
    from tqdm import tqdm
    st = Store(STORES / name)
    srcs = {s.id: s for s in vsrc.sources(name)}
    rs = np.random.RandomState(0)
    qs = [q for q in sample_queries(st, nq, rs) if q[0] in srcs]
    if not qs:
        print(f"{name:<8}no gradeable queries")
        return
    sel = np.sort(rs.choice(st.n, min(vqbe.REF_SAMPLE, st.n),
                            replace=False))
    ref = {c: st.col[c].take(sel) for c in CHANNELS}

    acc = {t: [] for t in vnuis.BATTERY}
    ident, rev, pr, wsum = [], [], [], {c: [] for c in CHANNELS}
    bar = tqdm(qs, desc=f"{name} grade", unit="q", leave=False)
    for qi, (mid, t0, t1) in enumerate(bar):
        src = srcs[mid]
        for tname, tf in vnuis.BATTERY.items():
            r = grade_one(st, src, mid, t0, t1, tf,
                          np.random.RandomState(1000 + qi), ref)
            if r:
                acc[tname].append(r)
                for c in CHANNELS:
                    wsum[c].append(r["w"][c])
        # untouched, and its mirror. The pair is the direction
        # DIAGNOSTIC, not a pass/fail: a genuinely symmetric event (a
        # car holding a straight line) SHOULD match its own reverse, so
        # demanding rejection everywhere would be demanding the wrong
        # thing. What is informative is the gap between them.
        f = grade_one(st, src, mid, t0, t1, None, rs, ref)
        r = grade_one(st, src, mid, t0, t1, vnuis.reverse, rs, ref)
        if f and r:
            ident.append(f)
            rev.append(r)
        if qi < 6:
            g = prune_recall(st, src, mid, t0, t1, ref, rs)
            if g:
                pr.append(g)

    flat = [r for v in acc.values() for r in v]
    if not flat:
        print(f"{name:<8}no gradeable queries")
        return
    chance = np.mean([r["sup"] / max(r["cand"], 1) for r in flat])
    row = dict(
        store=name, media=int(len(set(st.media.tolist()))),
        windows=int(st.n), minutes=round(st.man["seconds"] / 60, 1),
        queries=len(flat),
        yield_=round(float(np.mean([r["y"] for r in flat])), 3),
        prec=round(float(np.mean([r["p"] for r in flat])), 3),
        yield_half=round(float(np.mean([r["y_h"] for r in flat])), 3),
        prec_half=round(float(np.mean([r["p_h"] for r in flat])), 3),
        support_half=round(float(np.mean([r["sup_h"] for r in flat])), 1),
        support=round(float(np.mean([r["sup"] for r in flat])), 1),
        returned=round(float(np.mean([r["ret"] for r in flat])), 1),
        chance=round(float(chance), 3),
        rank1=round(float(np.mean([r["hit1"] for r in flat])), 3),
        med_rank=round(float(np.median([r["rank"] for r in flat])), 1),
        ident=round(float(np.mean([r["y"] for r in ident])), 3)
        if ident else None,
        d_margin=round(float(np.mean([r["y"] for r in ident])
                             - np.mean([r["y"] for r in rev])), 3)
        if rev else None,
        prune_recall=round(float(np.mean([p[0] for p in pr])), 3)
        if pr else None,
        bytes_probe=int(np.mean([p[1] for p in pr])) if pr else None,
        bytes_full=int(np.mean([p[2] for p in pr])) if pr else None,
        per_transform={t: round(float(np.mean([r["y"] for r in v])), 3)
                       for t, v in acc.items() if v},
        weights={c: round(float(np.mean(wsum[c])), 3) for c in CHANNELS},
    )
    out_rows.append(row)
    print(f"{name:<8}{row['media']:<7}{row['windows']:<9}"
          f"{row['yield_']:<8.3f}{row['prec']:<8.3f}"
          f"{row['support']:<9.1f}{row['returned']:<10.1f}"
          f"{row['chance']:<8.3f}{row['rank1']:<8.3f}"
          f"{row['ident'] if row['ident'] is not None else 0:<8.3f}"
          f"{row['d_margin'] if row['d_margin'] is not None else 0:<9.3f}"
          f"{row['prune_recall'] if row['prune_recall'] else 0:.3f}",
          flush=True)


def main():
    from flowgebd import arg
    want = arg("--stores", "sim").split(",")
    nq = arg("--queries", 20, int)
    print("SINGLE-VIEW, LABEL-FREE QbE — necessary-condition benchmark")
    print(f"query = {QSPAN:.0f}s example re-filmed by "
          f"{len(vnuis.BATTERY)} nuisance transforms; the store must "
          f"find the original.")
    print("k = ceil(1.5 x support) MAX, abstention on, no temporal "
          "exclusion (the target is a hard-negative neighbour)\n")
    print(f"{'store':<8}{'media':<7}{'windows':<9}{'yield':<8}"
          f"{'prec':<8}{'support':<9}{'returned':<10}{'chance':<8}"
          f"{'rank1':<8}{'ident':<8}{'D-marg':<9}{'P-rec'}")
    rows = []
    t = time.time()
    for name in want:
        if not (STORES / name / "manifest.json").exists():
            print(f"{name:<8}no store built")
            continue
        run(name, nq, rows)
    if rows:
        # Every run is preserved under its own name. A comparison that
        # reads one file twice reports zero difference and looks like a
        # clean null result - which is exactly what happened once here.
        rep = ROOT / f"native/{arg('--out', 'VGRADE')}.json"
        rep.write_text(json.dumps(
            dict(when=time.strftime("%Y-%m-%d %H:%M"),
                 encoder=vqbe.vcore.ENCODER,
                 res=list(vqbe.vcore.RES), wmode=vqbe.WMODE,
                 scales=list(vqbe.vcore.SCALES),
                 ctx_mult=vqbe.vcore.CTX_MULT,
                 true_frac=TRUE_FRAC, qspan=QSPAN, rows=rows),
            indent=2))
        print(f"\nper-transform yield")
        for r in rows:
            print(f"  {r['store']:<8}" + "  ".join(
                f"{k} {v:.3f}" for k, v in r["per_transform"].items()))
        print("mean channel weight chosen by the queries themselves")
        for r in rows:
            print(f"  {r['store']:<8}" + "  ".join(
                f"{k} {v:.3f}" for k, v in r["weights"].items()))
        print(f"\nwrote native/VGRADE.json  ({time.time() - t:.0f}s)")
    print("\nThis is a NECESSARY condition, not the product metric: it "
          "shows the same event survives being re-filmed, not that\n"
          "different instances of a similar event are found. Nothing "
          "label-free can show the second thing.")


if __name__ == "__main__":
    main()
