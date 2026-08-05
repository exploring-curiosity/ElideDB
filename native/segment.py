"""STEP 5 - SEGMENT. Ranked cuts -> the span set that becomes units.

Step 4 emits BOUNDARIES ranked by their own Ncut value. Step 5 turns
those into the actual retrieval units: a set of (t0, t1) spans.

Three design positions, each with a reason:

1. THE SPANS PARTITION THE TIMELINE EXACTLY. Coverage is 1.0, no gaps,
   no overlap. A unit set with holes silently drops footage - the
   content is in the store but unreachable by any query, which is the
   worst failure a database can have because nothing reports it.

2. SHORT SPANS ARE MERGED, NEVER DROPPED. Dropping a span punches a
   hole (see 1). Merging into the shorter neighbour preserves the
   partition. In practice this never fires - see 3.

3. THE MINIMUM SPAN LENGTH IS NOT A CONSTANT I PICK; it falls out of
   step 4. recursive_ncut only splits an interval at least MIN_SEG
   frames from either end and recurses on disjoint children, so EVERY
   boundary is >= MIN_SEG frames from every other. At MIN_SEG=4 and
   4 fps that is 1.0 s, so degenerate spans are impossible by
   construction rather than by filtering. The grade verifies this
   rather than trusting it.

   MIN_SAL below IS a constant I picked - the one in this module. It is
   fitted on both corpora jointly and sits on a flat plateau, but it is
   honest to call it what it is rather than claim the step is
   threshold-free.

The emitter decides HOW MANY of the ranked cuts to keep. Step 4 used a
z<0 cut, i.e. everything below the mean - roughly half the candidates
whatever the media contains. Otsu replaces it (default) because the
count then adapts to the media instead of to the candidate list; both
are still fitted per media, neither adds a constant. Measured:

    emitter | sim spanF1 | sim bMAE | oxford bMAE | oxford spans
    z<0     |   0.607    |  1.22 s  |   3.36 s    |  7 (truth 4)
    otsu    |   0.652    |  1.18 s  |   2.49 s    |  6 (truth 4)

Quality bar (set by native/sensitivity.py, deliberately loose): units
built from F1-0.5 boundaries retrieve as well as units from perfect
ones (yield 0.267 vs 0.267). So step 5 must produce REASONABLE,
non-degenerate spans - not precise ones.

FIXED (was: oxford span F1 0.000). The recursion used to split by a
DEPTH COUNTER, so a 9 s stretch of uniform driving got chopped into
four and no predicted span could reach IoU 0.5 against it. It now
stops on HOMOGENEITY: a segment is split only if its best cut is
prominent within its own Ncut(t) curve (salience >= MIN_SAL). A real
boundary makes a sharp deep minimum; a slow drift makes a shallow one.

    config                  | sim spanF1 | oxford spanF1 | oxford bMAE
    depth-only (old)        |   0.652    |     0.000     |   2.49 s
    homogeneity stop (new)  |   0.566    |     0.667     |   0.50 s

Deliberate trade: -0.086 on sim to turn a total failure on the second
domain into a pass. MIN_SAL sits on a flat plateau (1.0-1.2, breaks at
1.3) fitted on BOTH corpora at once, so it is one value for every
domain rather than per-corpus configuration.

Step 4's boundary detection is unaffected (count-matched F1 still 0.711
sim / 0.500 oxford); only the emitted SET shrinks. Note the two metrics
disagree - the change lowers boundary-level emitter F1 (0.667 -> 0.538
on sim) while raising span-level F1 on oxford from nothing to 0.667.
Boundary F1 was never the metric the next step consumes.

    python native/segment.py --corpus sim      # ~1 min
    python native/segment.py --corpus oxford   # ~15 s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import f1_at, media_jobs, arg                     # noqa: E402
from graphgebd import (FPS, MIN_SEG, affinity, frame_features,   # noqa: E402
                       recursive_ncut)

DEPTH = 4
# Fitted on BOTH corpora, flat plateau 1.0-1.2, breaks at 1.3.
# One value for every domain - no per-corpus configuration.
MIN_SAL = float(os.environ.get('SDX_MIN_SAL', '1.1'))
SPAN_F1_FLOOR = 0.40   # a real gate; the old build scored 0.000 here
RANK = os.environ.get('SDX_RANK', 'ncut')


def otsu(v):
    """Threshold minimising intra-class variance of a 1-D set.

    The z<0 emitter keeps whatever falls below the MEAN, i.e. roughly
    half the candidates no matter how many real events the media has -
    a count that cannot adapt. Otsu instead splits the Ncut values
    where they actually separate, so a media with two real events can
    emit two cuts and one with nine can emit nine. Still fitted from
    the media's own distribution; still no constant of mine.
    """
    x = np.sort(np.asarray(v, float))
    if len(x) < 3:
        return float(x[-1]) if len(x) else 0.0
    best, thr = -1.0, float(x[0])
    for i in range(1, len(x)):
        a, b = x[:i], x[i:]
        w0, w1 = len(a) / len(x), len(b) / len(x)
        s = w0 * w1 * (a.mean() - b.mean()) ** 2
        if s > best:
            best, thr = s, float((a[-1] + b[0]) / 2)
    return thr


def boundaries(F, depth=DEPTH, emitter="otsu"):
    """Step 4's output: emitted boundary times, ascending.

    Returns (kept, all_ranked_times) so the grade can separate "the
    detector ranked it well" from "the emitter chose to keep it".
    """
    W = affinity(frame_features(F))
    cuts = recursive_ncut(W, depth=depth, min_sal=MIN_SAL,
                          rank=RANK)
    if not cuts:
        return [], []
    times = [c[0] / FPS for c in cuts]
    # the quantity the emitter thresholds must match the ranking:
    # low-is-good for Ncut, high-is-good for salience.
    vals = np.array([c[1] if RANK != "sal" else -c[2] for c in cuts])
    if len(vals) <= 2:
        return sorted(times), times
    if emitter == "otsu":
        t = otsu(vals)
        keep = [tm for tm, v in zip(times, vals) if v <= t]
    else:
        z = (vals - vals.mean()) / (vals.std() + 1e-8)
        keep = [tm for tm, zz in zip(times, z) if zz < 0.0]
    return sorted(keep), times


def spans(bs, dur):
    """Boundaries -> an exact partition of [0, dur].

    Merge-not-drop: a span shorter than the structural minimum is
    absorbed into its shorter neighbour, so coverage stays 1.0.
    """
    mn = MIN_SEG / FPS
    ts = [0.0] + sorted(b for b in bs if 0.0 < b < dur) + [float(dur)]
    out = [[a, b] for a, b in zip(ts[:-1], ts[1:])]
    i = 0
    while len(out) > 1 and i < len(out):
        if out[i][1] - out[i][0] < mn:
            if i == 0:
                j = 1
            elif i == len(out) - 1:
                j = i - 1
            else:
                lo = out[i - 1][1] - out[i - 1][0]
                hi = out[i + 1][1] - out[i + 1][0]
                j = i - 1 if lo <= hi else i + 1
            k = min(i, j)
            out[k] = [min(out[i][0], out[j][0]),
                      max(out[i][1], out[j][1])]
            del out[max(i, j)]
            i = 0
            continue
        i += 1
    return [(float(a), float(b)) for a, b in out]


def segment(F, dur, depth=DEPTH, emitter="otsu"):
    """The step-5 entry point: frames -> retrieval spans."""
    bs, _ = boundaries(F, depth, emitter)
    return spans(bs, dur)


# ---------------------------------------------------------------- grade

def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def span_f1(pred, truth, thr=0.5):
    """Greedy one-to-one IoU matching, best pairs first."""
    pairs = sorted(((iou(p, t), i, j)
                    for i, p in enumerate(pred)
                    for j, t in enumerate(truth)),
                   key=lambda r: -r[0])
    up, ut, m = set(), set(), 0
    for v, i, j in pairs:
        if v < thr or i in up or j in ut:
            continue
        up.add(i)
        ut.add(j)
        m += 1
    pr = m / len(pred) if pred else 0.0
    rc = m / len(truth) if truth else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc


def truth_spans(gt, dur):
    return spans(list(gt), dur)


def main():
    import encode as E
    from tqdm import tqdm
    corpus = arg("--corpus", "sim")
    limit = arg("--limit", 12, int)
    em = arg("--emitter", "otsu")
    jobs = media_jobs(corpus, limit)
    print(f"STEP 5 segment — {corpus}, {len(jobs)} media, depth {DEPTH},"
          f" emitter {em}", flush=True)

    rows = []
    for name, path, gt in tqdm(jobs, desc="segment", unit="media"):
        dur = E.probe_duration(path)
        F = E.decode(path, fps=FPS, w=256)
        sp = segment(F, dur, emitter=em)
        sp2 = segment(F, dur, emitter=em)           # determinism
        ts = truth_spans(gt, dur)
        f, pr, rc = span_f1(sp, ts)
        bpred = [a for a, _ in sp[1:]]
        bmae = []
        for b in bpred:
            if gt:
                bmae.append(min(abs(b - g) for g in gt))
        cov = sum(b - a for a, b in sp) / dur
        gap = max((sp[i + 1][0] - sp[i][1])
                  for i in range(len(sp) - 1)) if len(sp) > 1 else 0.0
        d = [b - a for a, b in sp]
        rows.append(dict(name=name, f1=f, pr=pr, rc=rc,
                         n=len(sp), nt=len(ts), cov=cov, gap=abs(gap),
                         mn=min(d), md=float(np.median(d)), mx=max(d),
                         mae=float(np.mean(bmae)) if bmae else float("nan"),
                         det=(sp == sp2)))

    print(f"\n{'media':<9}{'spans':<7}{'truth':<7}{'spanF1':<9}"
          f"{'cov':<7}{'min_s':<8}{'med_s':<8}{'bMAE_s'}")
    for r in rows:
        print(f"{r['name']:<9}{r['n']:<7}{r['nt']:<7}{r['f1']:<9.2f}"
              f"{r['cov']:<7.3f}{r['mn']:<8.2f}{r['md']:<8.2f}"
              f"{r['mae']:.2f}")

    mn = min(r["mn"] for r in rows)
    print(f"\n--- STEP 5 GRADE ({corpus}, n={len(rows)}) ---")
    print(f"span F1 @IoU0.5   {np.mean([r['f1'] for r in rows]):.3f}"
          f"   (prec {np.mean([r['pr'] for r in rows]):.3f} / "
          f"rec {np.mean([r['rc'] for r in rows]):.3f})")
    print(f"boundary MAE      "
          f"{np.nanmean([r['mae'] for r in rows]):.2f} s")
    print(f"spans / media     {np.mean([r['n'] for r in rows]):.1f}  "
          f"(truth {np.mean([r['nt'] for r in rows]):.1f})")
    print(f"coverage          {np.mean([r['cov'] for r in rows]):.4f} "
          f"(want 1.0)   max gap {max(r['gap'] for r in rows):.2e} s")
    print(f"shortest span     {mn:.2f} s  (structural floor "
          f"{MIN_SEG / FPS:.2f} s)")
    print(f"longest span      {max(r['mx'] for r in rows):.1f} s")
    print("PASS gates:")
    ok = []
    # THE gate. The four structural checks below are all satisfied by
    # construction - coverage by the partition code, the floor by
    # MIN_SEG geometry, determinism trivially - so on their own they
    # certify nothing but the absence of a bug. An earlier revision
    # gated on those alone and stamped PASS on a domain scoring 0.000.
    f1m = float(np.mean([r["f1"] for r in rows]))
    ok.append((f"span F1 >= {SPAN_F1_FLOOR:.2f} (got {f1m:.3f})",
               f1m >= SPAN_F1_FLOOR))
    ok.append(("coverage == 1.0 (no unreachable footage)",
               all(abs(r["cov"] - 1.0) < 1e-6 for r in rows)))
    ok.append((f"no span < structural floor {MIN_SEG / FPS:.2f}s",
               mn >= MIN_SEG / FPS - 1e-6))
    ok.append(("deterministic (same frames -> same spans)",
               all(r["det"] for r in rows)))
    ok.append(("non-degenerate count (2 <= spans <= 3x truth)",
               all(2 <= r["n"] <= 3 * max(r["nt"], 1) for r in rows)))
    for label, v in ok:
        print(f"  [{'PASS' if v else 'FAIL'}] {label}")
    print(f"\nSTEP 5 {'PASS' if all(v for _, v in ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
