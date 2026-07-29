"""Train FDNN-V by distillation, benchmarking every iteration.

PROTOCOL
--------
Split is by TIME per stream (last 30% held out) — never random, because
consecutive frames are near-duplicates and a random split scores memorisation.

Stage 0  ridge regression from 24x32 grayscale pixels -> 1152. The floor any
         learned model must clear; if a linear map matches the teacher on this
         corpus, a neural student is decoration.
Stage 1  per-frame distillation, BOTH stems raced for real (short budget),
         winner trained to convergence. The temporal head is zero at init, so
         this stage is exactly the per-frame model.
Stage 2  recurrent distillation on sequences. The only new thing stage 2 can
         learn is to use state, so its margin over stage 1 IS the measured
         value of temporal context — the claim "video model beats frame
         model" gets a number or it gets dropped.

Every eval row: held-out fidelity (mean + p10 cosine to teacher), kNN
retrieval agreement (top-10 neighbour overlap vs the teacher's neighbours,
same-stream near-in-time pairs excluded — retrieval behaviour, not vector
closeness), and ms/frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import mlx.core as mx                                        # noqa: E402
import mlx.nn as nn                                          # noqa: E402
import mlx.optimizers as optim                               # noqa: E402
from mlx.utils import tree_flatten                           # noqa: E402

from elidedb.fdnnvideo import (FDNNVideoEncoder, distill_loss,  # noqa: E402
                               save_encoder)

MU = None          # train-set teacher mean: the common mode to see past
ANCHORS = None     # caption-text embeddings: the directions queries live in


def _loss(student, teacher):
    # anchor_w: 50 measured stable; 800 destabilised the recurrent stage
    # (fid collapsed to 0.63). The binding constraint turned out to be
    # CAPACITY, not loss weighting — see the teacher-224 calibration: the
    # fid->knn curve is near-vertical around fid 0.92, so the lever is a
    # bigger stem, not a louder loss term.
    return distill_loss(student, teacher, mu=MU, anchors=ANCHORS,
                        anchor_w=50.0)

CACHE = Path("data/cache/fdnnv_train.npz")
# NOT inside a store: a trained model is not store data, and keeping it
# under lake/ is how it was lost when the stores were cleared (2026-07-28).
OUT = Path("models/fdnnv")


def load_split(val_frac=0.3):
    z = np.load(CACHE)
    px, vec, sid, ts = z["px"], z["vec"].astype(np.float32), z["sid"], z["ts"]
    val = np.zeros(len(px), bool)
    for s in np.unique(sid):
        m = sid == s
        cut = np.quantile(ts[m], 1 - val_frac)
        val |= m & (ts >= cut)
    return px, vec, sid, ts, val


def fidelity_np(student, teacher):
    t = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-8)
    cos = (student * t).sum(1)
    return float(cos.mean()), float(np.percentile(cos, 10))


def knn_agreement(student, teacher, ts, sid, k=10, n=1500, seed=0):
    """Top-k neighbour overlap, teacher vs student, temporal near-duplicates
    excluded (same stream within 5 s) so agreement measures retrieval
    behaviour rather than 'the next frame looks like this frame'."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(student), min(n, len(student)), replace=False)
    t = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-8)
    s = student
    St, Ss = t[idx] @ t.T, s[idx] @ s.T
    hits = 0
    for row, i in enumerate(idx):
        ban = (sid == sid[i]) & (np.abs(ts - ts[i]) < 5_000_000_000)
        St[row, ban] = -2
        Ss[row, ban] = -2
        a = set(np.argpartition(-St[row], k)[:k].tolist())
        b = set(np.argpartition(-Ss[row], k)[:k].tolist())
        hits += len(a & b)
    return hits / (len(idx) * k)


def eval_model(model, px, vec, sid, ts, val, batch=128):
    """Held-out eval in the STREAMING regime: per stream, in time order,
    state carried — exactly how ingest will run it."""
    outs = np.zeros((int(val.sum()), vec.shape[1]), np.float32)
    where = np.where(val)[0]
    pos = {int(i): r for r, i in enumerate(where)}
    for s in np.unique(sid[val]):
        m = np.where(val & (sid == s))[0]
        m = m[np.argsort(ts[m])]
        e = model.embed_frames_np(px[m], batch=batch)
        for r, i in zip(range(len(m)), m):
            outs[pos[int(i)]] = e[r]
    fid, p10 = fidelity_np(outs, vec[val])
    knn = knn_agreement(outs, vec[val], ts[val], sid[val])
    return fid, p10, knn


def speed(model, batch=64, reps=5):
    x = mx.array(np.random.rand(1, batch, 144, 192, 3).astype(np.float32))
    e, h = model(x)
    mx.eval(e, h)
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        e, h = model(x)
        mx.eval(e, h)
        best = min(best, (time.perf_counter() - t) / batch * 1000)
    return best


def ridge_baseline(px, vec, val):
    g = px.mean(axis=3)[:, ::6, ::6].reshape(len(px), -1).astype(np.float32)
    g = (g - g.mean(0)) / (g.std(0) + 1e-6)
    Xtr, Ytr = g[~val], vec[~val]
    W = np.linalg.solve(Xtr.T @ Xtr + 10.0 * np.eye(g.shape[1]), Xtr.T @ Ytr)
    pred = g[val] @ W
    pred /= np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8
    return fidelity_np(pred, vec[val])


def train_stage1(model, px, vec, val, epochs, lr=1e-3, batch=256, seed=0,
                 log=print):
    rng = np.random.default_rng(seed)
    tr = np.where(~val)[0]
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    def loss_fn(m, x, y):
        e, _ = m(x[:, None])                     # T=1: pure per-frame
        return _loss(e[:, 0], y)

    lg = nn.value_and_grad(model, loss_fn)
    for ep in range(epochs):
        rng.shuffle(tr)
        tl = 0.0
        for i in range(0, len(tr), batch):
            b = tr[i:i + batch]
            x = mx.array(px[b].astype(np.float32) / 127.5 - 1.0)
            y = mx.array(vec[b])
            loss, grads = lg(model, x, y)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            tl += float(loss.item())
        log(ep, tl / max(len(tr) // batch, 1))
    return model


def train_stage2(model, px, vec, sid, ts, val, epochs, T=32, B=8, lr=2e-4,
                 seed=0, log=print):
    rng = np.random.default_rng(seed)
    # sequence starts: any train frame with T consecutive cached frames of the
    # same stream after it (cache is per-stream time-ordered)
    starts = []
    for s in np.unique(sid):
        m = np.where(sid == s)[0]
        ok = ~val[m]
        for j in range(len(m) - T):
            if ok[j:j + T].all():
                starts.append(m[j])
    starts = np.array(starts)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    burn = 4                                     # state needs a run-up

    def loss_fn(m, x, y):
        e, _ = m(x)
        return _loss(e[:, burn:], y[:, burn:])

    lg = nn.value_and_grad(model, loss_fn)
    steps = max(len(starts) // (B * T // 4), 40)
    for ep in range(epochs):
        tl = 0.0
        for _ in range(steps):
            b = rng.choice(starts, B, replace=False)
            x = np.stack([px[i:i + T] for i in b]).astype(np.float32)
            y = np.stack([vec[i:i + T] for i in b])
            loss, grads = lg(model, mx.array(x / 127.5 - 1.0), mx.array(y))
            try:
                grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            except AttributeError:
                pass                     # older MLX: train unclipped
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            tl += float(loss.item())
        log(ep, tl / steps)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race-epochs", type=int, default=4)
    ap.add_argument("--s1-epochs", type=int, default=14)
    ap.add_argument("--s2-epochs", type=int, default=8)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--vit-depth", type=int, default=4)
    ap.add_argument("--skip-race", action="store_true",
                    help="go straight to the vit stem (already raced twice)")
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    px, vec, sid, ts, val = load_split()
    print(f"pairs: {len(px):,} | train {int((~val).sum()):,} "
          f"| val {int(val.sum()):,} (time split)", flush=True)

    global MU, ANCHORS
    tn = vec[~val] / (np.linalg.norm(vec[~val], axis=1, keepdims=True) + 1e-8)
    MU = mx.array(tn.mean(0).astype(np.float32))
    try:
        from elidedb import Store
        caps = Store.open("lake/bridge").table("context_captions").scan()
        av = np.asarray(caps.column("vector").to_pylist(), np.float32)
        av /= np.linalg.norm(av, axis=1, keepdims=True) + 1e-8
        rng_a = np.random.default_rng(0)
        ANCHORS = mx.array(av)          # all of them; they are the query space
        print(f"caption anchors: {ANCHORS.shape[0]}", flush=True)
    except Exception as e:
        print(f"no caption anchors ({e}); training without", flush=True)

    report = {"iterations": []}

    fid, p10 = ridge_baseline(px, vec, val)
    print(f"[stage0] ridge-from-pixels  fid={fid:.4f} p10={p10:.4f}  "
          "<- the bar a neural student must clear", flush=True)
    report["ridge"] = {"fid": fid, "p10": p10}

    # ---- stage 1: race the stems, believe the measurement ------------------
    racers = {}
    for stem in (("vit",) if args.skip_race else ("conv", "vit")):
        m = FDNNVideoEncoder(stem=stem, stem_width=args.width,
                             channels=args.channels,
                             vit_depth=args.vit_depth, seed=args.seed)
        t0 = time.time()
        train_stage1(m, px, vec, val, args.race_epochs,
                     log=lambda e, l: None)
        fid, p10, knn = eval_model(m, px, vec, sid, ts, val)
        ms = speed(m)
        n = sum(int(np.prod(v.shape))
                for _, v in tree_flatten(m.parameters()))
        row = {"stage": f"race-{stem}", "fid": fid, "p10": p10, "knn": knn,
               "ms_frame": ms, "params": n,
               "train_s": round(time.time() - t0, 1)}
        report["iterations"].append(row)
        racers[stem] = (m, fid)
        print(f"[race ] {stem:4s} fid={fid:.4f} p10={p10:.4f} knn={knn:.3f} "
              f"{ms:.3f} ms/f {n/1e6:.2f}M ({row['train_s']}s)", flush=True)

    stem = max(racers, key=lambda s: racers[s][1])
    model = racers[stem][0]
    print(f"-> winner: {stem}", flush=True)

    t0 = time.time()
    train_stage1(model, px, vec, val, args.s1_epochs - args.race_epochs,
                 lr=5e-4, log=lambda e, l: print(
                     f"  s1 ep{e} loss {l:.4f}", flush=True))
    fid1, p10_1, knn1 = eval_model(model, px, vec, sid, ts, val)
    ms = speed(model)
    report["iterations"].append(
        {"stage": "stage1-final", "fid": fid1, "p10": p10_1, "knn": knn1,
         "ms_frame": ms, "train_s": round(time.time() - t0, 1)})
    print(f"[s1   ] fid={fid1:.4f} p10={p10_1:.4f} knn={knn1:.3f} "
          f"{ms:.3f} ms/f", flush=True)

    # ---- stage 2: the temporal claim, measured -----------------------------
    from mlx.utils import tree_flatten as _tf
    s1_weights = {k: np.array(v) for k, v in _tf(model.parameters())}
    t0 = time.time()
    train_stage2(model, px, vec, sid, ts, val, args.s2_epochs,
                 log=lambda e, l: print(f"  s2 ep{e} loss {l:.4f}",
                                        flush=True))
    fid2, p10_2, knn2 = eval_model(model, px, vec, sid, ts, val)
    ms = speed(model)
    report["iterations"].append(
        {"stage": "stage2-final", "fid": fid2, "p10": p10_2, "knn": knn2,
         "ms_frame": ms, "train_s": round(time.time() - t0, 1)})
    print(f"[s2   ] fid={fid2:.4f} p10={p10_2:.4f} knn={knn2:.3f} "
          f"{ms:.3f} ms/f", flush=True)
    print(f"[gain ] temporal state: fid {fid2 - fid1:+.4f} "
          f"p10 {p10_2 - p10_1:+.4f} knn {knn2 - knn1:+.3f}", flush=True)
    report["temporal_gain"] = {"fid": fid2 - fid1, "p10": p10_2 - p10_1,
                               "knn": knn2 - knn1}
    if fid2 < fid1 - 0.002 or knn2 < knn1 - 0.005:
        # A search returns its best point, not its last: if the temporal
        # stage did not earn its place on held-out data, ship stage 1.
        from mlx.utils import tree_unflatten as _tu
        model.update(_tu([(k, mx.array(v)) for k, v in s1_weights.items()]))
        model.cell.freeze(keys=["basis_types", "mask"], recurse=False)
        mx.eval(model.parameters())
        report["kept"] = "stage1"
        print("[keep ] stage 2 did not earn its place -> shipping stage 1",
              flush=True)
    else:
        report["kept"] = "stage2"

    save_encoder(model, {"teacher": "mlx-community/siglip-so400m-patch14-384",
                         "report": report, "stem": stem,
                         "input_hw": [144, 192]}, OUT)
    Path("bench_fdnnv.json").write_text(json.dumps(report, indent=2))
    print(f"saved -> {OUT}  |  bench_fdnnv.json", flush=True)


if __name__ == "__main__":
    main()
