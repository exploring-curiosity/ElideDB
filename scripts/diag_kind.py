"""KIND-GAP diagnostic: which descriptor variant separates object kinds?

The wall, measured three times: the store's object descriptors are
instance-ReID features - best cross-seed object match (med 0.75-0.85)
barely clears random background episodes (0.70-0.72), and no query-time
statistic retrieves at top-1% precision through a ~0.1 gap. Before any
full rebuild, this measures candidate descriptors on a small CROP BANK
and reports the gap each one achieves. The variant that materially
widens it earns the corpus build; if none does, the answer is a
different model, not a different pipeline.

Crops come from the TRAJECTORIES table (per-frame boxes are already
persisted), event-bound tracks only - the manipulated objects, which is
what the low-support queries are about. No re-tracking, no detector.

Variants:
    base      the shipped pipeline reproduced: crop -> ConvNeXt-Tiny,
              mean over views                        (expects gap ~0.09)
    pad       +30% context around the box
    vits      DINOv3 ViT-S fp32 (the A/B loser for INSTANCES - but kind
              is a different task and fp32 ViT features differ)
    maxv      max over views instead of mean (best view, not blur)
    color     base + Lab colour moments (generic image statistic,
              corpus-free) as a parallel similarity, averaged in

Metric per query (seed group 0, protocol seeds): for each seed-episode
event-bound track, its similarity to every episode in the bank
(4 other seed episodes = positives, 40 background = negatives) ->
AUC + median gap. Grades are never read beyond the protocol seed list.

    python scripts/diag_kind.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                      # noqa: E402
from elidedb.video import FrameSet                             # noqa: E402

QUERIES = (0, 1, 2, 7, 8)
NBG = 40
NVIEW = 4


def bank_episodes(db):
    """Seed episodes per query (protocol) + shared background set."""
    import pyarrow.parquet as pq
    ep = db.table("episodes").scan()
    keys = [(str(s), int(a), int(b)) for s, a, b in
            zip(ep.column("stream").to_pylist(), ep.column("ts").to_pylist(),
                ep.column("t1").to_pylist())]
    eidx = [int(i) for i in ep.column("episode_index").to_pylist()]
    t = pq.read_table(ROOT / "eval/truthsets/graded.parquet").to_pydict()
    G = {(int(q), int(e)): int(v) for q, e, v in
         zip(t["query_id"], t["episode_index"], t["true"])}
    seeds_of = {}
    all_seed_eps = set()
    for qi in QUERIES:
        truths = [i for i, e in enumerate(eidx) if G.get((qi, e)) == 1]
        rs = np.random.RandomState(0)
        ns = min(5, len(truths))
        grp = sorted(set(int(x) for x in rs.choice(truths, ns,
                                                   replace=False)))
        seeds_of[qi] = grp
        all_seed_eps |= set(grp)
    rng = np.random.default_rng(0)
    pool = [i for i in range(len(keys)) if i not in all_seed_eps]
    bg = sorted(int(x) for x in rng.choice(pool, NBG, replace=False))
    return keys, seeds_of, bg, sorted(all_seed_eps | set(bg))


def crops_for(db, keys, wanted_eps):
    """{episode -> [(track_key, [crop_bgr, ...])]} for event-bound tracks,
    NATIVE resolution, views spread across the track."""
    ev = db.table("events").scan().to_pydict()
    bound = {(str(s), int(o)) for s, o in zip(ev["stream"], ev["object_id"])
             if int(o) >= 0}
    tr = db.table("trajectories").scan().to_pydict()
    ep_span = {i: keys[i] for i in wanted_eps}
    span_of = defaultdict(list)
    for i, (s, a, b) in ep_span.items():
        span_of[s].append((a, b, i))
    # collect per-track samples inside wanted episodes
    tracks = defaultdict(list)
    for j in range(len(tr["ts"])):
        s = str(tr["stream"][j])
        oid = int(tr["object_id"][j])
        if tr["is_agent"][j] or (s, oid) not in bound:
            continue
        ts = int(tr["ts"][j])
        for a, b, i in span_of.get(s, ()):
            if a <= ts <= b:
                tracks[(i, s, int(tr["track_ts"][j]), oid)].append(
                    (ts, (int(tr["x0"][j]), int(tr["y0"][j]),
                          int(tr["x1"][j]), int(tr["y1"][j]))))
                break
    # frames to decode, grouped per episode
    need = defaultdict(set)
    for (i, s, t0, oid), pts in tracks.items():
        pts.sort()
        pick = np.unique(np.linspace(0, len(pts) - 1, NVIEW)
                         .round().astype(int))
        for k in pick:
            need[i].add(pts[k][0])
    ft = db.table("frames").scan()
    out = defaultdict(list)
    from tqdm import tqdm
    for i in tqdm(sorted(need), desc="decode", unit="ep"):
        s, a, b = keys[i]
        sel = ft.filter(pc.and_(
            pc.equal(ft.column("stream"), s),
            pc.and_(pc.greater_equal(ft.column("ts"), a),
                    pc.less_equal(ft.column("ts"), b))))
        tsl = [int(v) for v in sel.column("ts").to_pylist()]
        idx = [tsl.index(t) for t in sorted(need[i]) if t in tsl]
        if not idx:
            continue
        dec = FrameSet(db, "frames", sel.take(np.array(idx))).decode()
        frame_at = {int(t): im for t, im in dec}
        for key, pts in tracks.items():
            if key[0] != i:
                continue
            pts.sort()
            pick = np.unique(np.linspace(0, len(pts) - 1, NVIEW)
                             .round().astype(int))
            cs = []
            for k in pick:
                t, (x0, y0, x1, y1) = pts[k]
                im = frame_at.get(t)
                if im is None:
                    continue
                cs.append((im, (x0, y0, x1, y1)))
            if cs:
                out[i].append((key, cs))
    return out


# ------------------------------------------------------------ descriptors
def _crop(im, box, pad=0.0):
    x0, y0, x1, y1 = box
    if pad:
        w, h = x1 - x0, y1 - y0
        x0 -= int(w * pad); x1 += int(w * pad)
        y0 -= int(h * pad); y1 += int(h * pad)
    H, W = im.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return im[y0:y1, x0:x1]


def color_moments(c):
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    v = np.concatenate([lab.mean(0), lab.std(0)])
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def embed_bank(bank, variant):
    """{episode -> matrix of per-track descriptors} for one variant."""
    from elidedb import dinov3
    pad = 0.30 if "pad" in variant else 0.0
    mid = None
    if "vits" in variant:
        mid = "facebook/dinov3-vits16-pretrain-lvd1689m"
    elif "vitb" in variant:
        mid = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    agg = "max" if variant == "maxv" else "mean"
    out = {}
    from tqdm import tqdm
    for i, tracks in tqdm(bank.items(), desc=f"embed {variant}",
                          unit="ep"):
        vecs = []
        for key, cs in tracks:
            crops = []
            for im, box in cs:
                c = _crop(im, box, pad)
                if c is not None:
                    crops.append(c)
            if not crops:
                continue
            E = dinov3.embed(crops, mid=mid) if mid else dinov3.embed(crops)
            if agg == "max":
                # per-view vectors kept; similarity later takes max over
                # view pairs - here approximate with the top view by norm
                v = E[np.linalg.norm(E, axis=1).argmax()]
            else:
                v = E.mean(0)
            v = v / (np.linalg.norm(v) + 1e-8)
            if variant == "color":
                cm = np.mean([color_moments(c) for c in crops], 0)
                cm = cm / (np.linalg.norm(cm) + 1e-8)
                v = np.concatenate([v, cm * 0.5])
                v = v / np.linalg.norm(v)
            vecs.append(v)
        if vecs:
            out[i] = np.stack(vecs)
    return out


def score(desc, seeds_of, bg):
    print(f"  {'q':<5}{'gap':>8}{'AUC':>8}   (cross-seed vs background)")
    aucs = []
    for qi in QUERIES:
        S = [e for e in seeds_of[qi] if e in desc]
        if len(S) < 3:
            print(f"  q{qi:02d}   (too few seed episodes with tracks)")
            continue
        cross, back = [], []
        for e in S:
            for v in desc[e]:
                for e2 in S:
                    if e2 == e:
                        continue
                    cross.append(float((desc[e2] @ v).max()))
                for e2 in bg:
                    if e2 in desc:
                        back.append(float((desc[e2] @ v).max()))
        cross, back = np.array(cross), np.array(back)
        lab = np.r_[np.ones(len(cross)), np.zeros(len(back))]
        val = np.r_[cross, back]
        r = val.argsort().argsort()
        auc = (r[lab == 1].mean() - (len(cross) - 1) / 2) / max(len(back), 1)
        gap = float(np.median(cross) - np.median(back))
        aucs.append(auc)
        print(f"  q{qi:02d} {gap:>8.3f}{auc:>8.3f}")
    if aucs:
        print(f"  mean AUC {np.mean(aucs):.3f}")
    return float(np.mean(aucs)) if aucs else 0.0


def main():
    db = Store.open(str(ROOT / "lake/fresh_bench"))
    keys, seeds_of, bg, wanted = bank_episodes(db)
    print(f"crop bank: {len(wanted)} episodes "
          f"({sum(len(v) for v in seeds_of.values())} seed slots, {NBG} bg)")
    bank = crops_for(db, keys, wanted)
    n_tracks = sum(len(v) for v in bank.values())
    print(f"{len(bank)} episodes with event-bound tracks, "
          f"{n_tracks} tracks", flush=True)
    results = {}
    want = (sys.argv[sys.argv.index("--variants") + 1].split(",")
            if "--variants" in sys.argv
            else ["base", "pad", "maxv", "color", "vits"])
    for variant in want:
        print(f"== {variant}")
        desc = embed_bank(bank, variant)
        results[variant] = score(desc, seeds_of, bg)
    print("\nmean AUC by variant: " + "  ".join(
        f"{k}={v:.3f}" for k, v in sorted(results.items(),
                                          key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
