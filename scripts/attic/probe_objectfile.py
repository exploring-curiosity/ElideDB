"""OBJECT FILES: does identity pooled ALONG A TRACK beat single crops?

Single-frame crop identity is dead: across 51,101 crops the best cosine
to "an eggplant" was 0.184 (mean 0.048), and the crops in episodes that
genuinely contain one ranked 75th and 110th corpus-wide. That approach
asked a 30-pixel patch, once, in isolation.

Human vision does not work that way. Treisman/Kahneman's OBJECT FILE:
the visual system binds features to a persistent token that survives
motion, and accumulates evidence across views rather than re-deciding
each glimpse. Gestalt COMMON FATE supplies the token without any
recognition - pixels that move together are one thing.

So this pipeline is two-stream by construction:
  dorsal ("where/how")  common-fate motion -> a persistent track
  ventral ("what")      crops ALONG that track, embedded and POOLED

The measurement is a straight comparison on identical episodes:
  single   best cosine from any one crop      (what already failed)
  pooled   cosine of the track-mean embedding (the object file)

If pooling does not separate the true episodes, identity from pixels is
finished for this corpus and the honest move is to say so.

  python scripts/probe_objectfile.py [--q 9] [--n 60]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from elidedb.fftools import find  # noqa: E402
from elidedb.store import Store  # noqa: E402

EPOCH_NS = 1_704_067_200_000_000_000
FILE_STRIDE_NS = 20_000_000_000_000
FPS = 5.0
SRC = ROOT / "data/bridge/videos/observation.images.image_0/chunk-000"
# full source resolution: the rendition was downscaled, and a small
# object cannot survive that. Resolution is the one untested variable.
W = H = 480
NF = 12
MIN_BLOB_FRAC = 4e-4


def frames_of(stream, t0, n=NF):
    fidx = int(stream.rsplit("-", 1)[1])
    base = EPOCH_NS + fidx * FILE_STRIDE_NS
    t_sec = max((t0 - base) / 1e9, 0.0)
    src = SRC / f"file-{fidx:03d}.mp4"
    if not src.exists():
        return []
    r = subprocess.run(
        [find("ffmpeg"), "-v", "error", "-ss", f"{t_sec:.3f}", "-i", str(src),
         "-frames:v", str(n), "-vf", f"scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
    fb = W * H * 3
    return [np.frombuffer(r.stdout[i*fb:(i+1)*fb], np.uint8).reshape(H, W, 3)
            for i in range(len(r.stdout) // fb)]


def tracks_of(frames):
    """Common fate: coherent motion blobs linked across frames.

    Returns list of tracks; each track is [(frame_idx, bbox), ...]."""
    import cv2
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    tracks, live = [], []
    for i in range(len(grey) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grey[i], grey[i+1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med))) + 1e-6
        mask = (mag > max(1.0, med + 4 * 1.4826 * mad)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        nlab, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        min_blob = MIN_BLOB_FRAC * mask.shape[0] * mask.shape[1]
        blobs = []
        for j in range(1, nlab):
            if stats[j, cv2.CC_STAT_AREA] < min_blob:
                continue
            x, y, w, h = (stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP],
                          stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT])
            blobs.append((tuple(cents[j]), (x, y, w, h)))
        nxt = []
        for c, bb in blobs:
            best, bd = None, 1e9
            for tr in live:
                d = float(np.hypot(*(np.array(tr[-1][2]) - np.array(c))))
                if d < bd:
                    best, bd = tr, d
            if best is not None and bd < 60:
                best.append((i, bb, c))
                nxt.append(best)
            else:
                tr = [(i, bb, c)]
                tracks.append(tr)
                nxt.append(tr)
        live = nxt
    return tracks


def embed_crops(frames, track, embed, pad=0.35, max_views=8):
    """Ventral stream: the SAME object seen across the track."""
    views = []
    step = max(1, len(track) // max_views)
    for (fi, (x, y, w, h), _) in track[::step][:max_views]:
        if fi >= len(frames):
            continue
        px, py = int(w * pad), int(h * pad)
        x0, y0 = max(x - px, 0), max(y - py, 0)
        x1, y1 = min(x + w + px, W), min(y + h + py, H)
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        views.append(frames[fi][y0:y1, x0:x1])
    if not views:
        return None, None
    V = embed(views)                       # (v, d) already normalised
    single = V                              # each view on its own
    pooled = V.mean(0)
    pooled /= np.linalg.norm(pooled) + 1e-8
    return single, pooled


def main():
    argv = sys.argv
    qi = int(argv[argv.index("--q") + 1]) if "--q" in argv else 9
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 60
    db = Store.open("lake/bench")
    t = pq.read_table(ROOT / "eval/truthsets/bridge4h.parquet").to_pydict()
    truth = {(int(q), s, int(a)): int(v) for q, s, a, v in
             zip(t["query_id"], t["stream"], t["t0"], t["true"])}
    noun = {9: "an eggplant", 10: "a banana", 8: "a spoon",
            0: "a green object", 1: "a yellow object"}[qi]

    from elidedb.embeddings import DEFAULT_MODEL, _embed_images, embed_text

    def embed(imgs):
        from PIL import Image
        V = _embed_images([Image.fromarray(im) for im in imgs], DEFAULT_MODEL)
        V = np.asarray(V, np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        return V

    qv = np.asarray(embed_text(noun), np.float32)
    qv /= np.linalg.norm(qv) + 1e-8

    ep = db.table("episodes").scan()
    keys = list(zip(ep.column("stream").to_pylist(),
                    [int(v) for v in ep.column("ts").to_pylist()]))
    pos = [k for k in keys if truth.get((qi, k[0], k[1])) == 1]
    rng = np.random.default_rng(0)
    rest = [k for k in keys if k not in pos]
    negs = [rest[i] for i in rng.choice(len(rest), min(n - len(pos), len(rest)),
                                        replace=False)]
    sample = [(k, 1) for k in pos] + [(k, 0) for k in negs]
    print(f"q{qi:02d} noun={noun!r}  {len(pos)} true + {len(negs)} random "
          f"@ {W}x{H} source resolution", flush=True)

    rows = []
    for i, (k, lab) in enumerate(sample):
        f = frames_of(*k)
        if len(f) < 4:
            continue
        trs = [tr for tr in tracks_of(f) if len(tr) >= 3]
        if not trs:
            continue
        bs, bp = -1.0, -1.0
        for tr in sorted(trs, key=len, reverse=True)[:4]:
            single, pooled = embed_crops(f, tr, embed)
            if single is None:
                continue
            bs = max(bs, float(np.max(single @ qv)))
            bp = max(bp, float(pooled @ qv))
        if bs < 0:
            continue
        rows.append((lab, bs, bp))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)

    A = np.array(rows, float)
    lab = A[:, 0].astype(bool)
    # THE metric: true / min(k, support) at k=10, the ledger's definition.
    # Separation is diagnostic only - it dilutes as the base rate rises,
    # so it cannot be compared across queries.
    K = 10
    npos = int(lab.sum())
    denom = min(K, npos)
    print(f"\n{'method':>10} {'true mean':>10} {'rest mean':>10} "
          f"{'sep':>7} {'yield@10':>9}   true ranks (of %d)" % len(A))
    for j, nm in enumerate(["single", "POOLED"]):
        x = A[:, j + 1]
        order = np.argsort(-x)
        ranks = sorted(int(np.where(order == i)[0][0]) + 1
                       for i in np.where(lab)[0])
        hit = sum(1 for r in ranks if r <= K)
        sep = (x[lab].mean() - x[~lab].mean()) / (x[~lab].std() + 1e-9)
        print(f"{nm:>10} {x[lab].mean():>10.3f} {x[~lab].mean():>10.3f} "
              f"{sep:>6.2f}s {hit}/{denom} = {hit/denom:>4.2f}   {ranks[:10]}")


if __name__ == "__main__":
    main()
