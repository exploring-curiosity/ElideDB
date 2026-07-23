"""Train FDNN-V2. Warm-started from V1; gates from docs/FDNNV2_PLAN.md §4.

Stage A (no captions needed): predictive (two horizons) + arrow-of-time +
          appearance anchor.
Stage B (needs 7B captions): + verb-focused contrastive with hard negatives,
          training the text adapter jointly.

Probes after every stage, all on HELD-OUT time, labels from outside the
store, eval-only:
  close/open   nearest-centroid accuracy in context space (gate ≥ 0.80;
               pixels scored 0.60, the corpus's robot-state bound is 0.85)
  aot          forward-vs-reversed AUC (gate ≥ 0.90)
  app_fid      appearance-head fidelity to its index (gate ≥ 0.90 — the
               existing indexes must keep working)
  ms/frame     (gate ≤ 0.5 batched)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import mlx.core as mx                                        # noqa: E402
import mlx.nn as nn                                          # noqa: E402
import mlx.optimizers as optim                               # noqa: E402

from elidedb.fdnnvideo import load_encoder                   # noqa: E402
from elidedb.fdnnv2 import (CTX_DIM, FDNNv2, TextAdapter,    # noqa: E402
                            VERB_SWAPS, pool_event, save_v2, swap_verbs)

CACHE = Path("data/cache/b4h")
OUT = Path("lake/bridge4h/models/fdnnv2")


def load_cache():
    px = np.load(CACHE / "px.npy", mmap_mode="r")
    vec = np.load(CACHE / "vec.npy")
    sid = np.load(CACHE / "sid.npy")
    ts = np.load(CACHE / "ts.npy")
    val = np.zeros(len(ts), bool)
    for s in np.unique(sid):
        m = sid == s
        cut = np.quantile(ts[m], 0.7)
        val |= m & (ts >= cut)
    return px, vec, sid, ts, val


def episode_groups(sid, ts, streams):
    """close/open/put-in episode -> cache index ranges (EVAL ONLY)."""
    import pyarrow.parquet as pq
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    from elidedb import Store
    db = Store.open("lake/bridge4h")
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(),
                         epm.column("stream").to_pylist()))
    smap = {s: i for i, s in enumerate(streams)}
    groups = {"close": [], "open": [], "putin": []}
    for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"], t["task"]):
        k = (k or "").lower()
        s = stream_of.get(int(i))
        if s not in smap or "drawer" not in k:
            continue
        which = ("close" if "close" in k else
                 "open" if "open" in k else
                 "putin" if ("put" in k or "place" in k) else None)
        if which is None:
            continue
        m = np.where((sid == smap[s]) & (ts >= int(a)) & (ts <= int(b)))[0]
        if len(m) >= 4:
            groups[which].append(m)
    return groups


def probes(model, px, vec, sid, ts, val, groups, rng):
    """The gate metrics. Context embeddings are novelty-pooled per episode —
    the same pooling retrieval will use."""
    def ep_ctx(m):
        x = px[m][None].astype(np.float32) / 127.5 - 1.0
        out = model.run(mx.array(x))
        return pool_event(np.array(out["ctx"][0]),
                          np.array(out["gates"][0]), 0, len(m))

    # close vs open, held-out episodes only
    accs = {}
    for a_name, b_name in (("close", "open"), ("close", "putin")):
        A = [g for g in groups[a_name] if val[g].mean() > 0.5][:60]
        B = [g for g in groups[b_name] if val[g].mean() > 0.5][:60]
        if len(A) < 6 or len(B) < 6:
            accs[f"{a_name}_vs_{b_name}"] = float("nan")
            continue
        va = np.stack([ep_ctx(m) for m in A])
        vb = np.stack([ep_ctx(m) for m in B])
        hit = tot = 0
        for (X, Y) in ((va, vb), (vb, va)):
            p = rng.permutation(len(X))
            tr, te = X[p[:len(X)//2]], X[p[len(X)//2:]]
            ca, cb = tr.mean(0), Y.mean(0)
            ca /= np.linalg.norm(ca) + 1e-8
            cb /= np.linalg.norm(cb) + 1e-8
            for x in te:
                hit += (x @ ca > x @ cb)
                tot += 1
        accs[f"{a_name}_vs_{b_name}"] = hit / tot

    # arrow of time on held-out clips
    vi = np.where(val)[0]
    scores, labels = [], []
    for _ in range(40):
        s = rng.choice(np.unique(sid[vi]))
        m = vi[sid[vi] == s]
        j = rng.integers(0, len(m) - 24)
        clip = px[m[j:j + 24]].astype(np.float32) / 127.5 - 1.0
        for arr, lab in ((clip, 1.0), (clip[::-1].copy(), 0.0)):
            out = model.run(mx.array(arr[None]))
            scores.append(float(out["aot"].item()))
            labels.append(lab)
    scores, labels = np.array(scores), np.array(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    auc = float(np.mean([[p > n for n in neg] for p in pos]))

    # appearance fidelity on a held-out sample
    pick = rng.choice(vi, 800, replace=False)
    x = px[np.sort(pick)].astype(np.float32) / 127.5 - 1.0
    out = model.run(mx.array(x[None]))
    app = np.array(out["app"][0])
    tgt = vec[np.sort(pick)]
    tgt = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-8)
    fid = float((app * tgt).sum(1).mean())

    # speed
    xb = mx.array(np.random.rand(1, 64, 144, 192, 3).astype(np.float32))
    out = model.run(xb)
    mx.eval(out["app"])
    t0 = time.perf_counter()
    for _ in range(3):
        out = model.run(xb)
        mx.eval(out["app"], out["ctx"])
    ms = (time.perf_counter() - t0) / 3 / 64 * 1000
    return {**{k: round(v, 3) for k, v in accs.items()},
            "aot_auc": round(auc, 3), "app_fid": round(fid, 4),
            "ms_frame": round(ms, 3)}


def stage_a(model, px, vec, sid, ts, val, epochs=8, T=24, B=8, lr=4e-4,
            seed=0, freeze_trunk=False):
    rng = np.random.default_rng(seed)
    if freeze_trunk:
        # A3/A4 measured it twice: training the trunk with these objectives
        # DEGRADES the episode separability the warm start already carries
        # (0.70 untrained -> 0.42-0.47 trained), because the objectives
        # reorganise the state the ctx head projects. The trunk already
        # knows; only the new heads need to learn.
        model.base.freeze()
    tr = np.where(~val)[0]
    starts = []
    for s in np.unique(sid):
        m = np.where(sid == s)[0]
        ok = ~val[m]
        for j in range(0, len(m) - T - 12, 3):
            if ok[j:j + T + 12].all():
                starts.append(m[j])
    starts = np.array(starts)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    frozen = np.array(model.base.cell.mask)
    K = model.horizons

    Kmax = max(K)

    def loss_fn(m, x, xr, tgt_now, y_aot):
        out = m.run(x)                     # x has T + Kmax frames
        outr = m.run(xr)
        T_eff = x.shape[1] - Kmax
        # L_app anchor over the scored prefix
        l = 0.5 * mx.mean(1.0 - mx.sum(
            out["app"][:, :T_eff] * tgt_now, axis=-1))
        # L1: predict the model's OWN future glimpse-change, stop-gradded
        g = out["g"]
        for p, k in zip(out["preds"], K):
            tk = mx.stop_gradient(g[:, k:k + T_eff] - g[:, :T_eff])
            tk = tk * mx.rsqrt(mx.sum(tk * tk, axis=-1, keepdims=True) + 1e-8)
            pn = p[:, :T_eff]
            pn = pn * mx.rsqrt(mx.sum(pn * pn, axis=-1, keepdims=True) + 1e-8)
            l = l + mx.mean(1.0 - mx.sum(pn * tk, axis=-1))
        # L2 arrow of time
        logits = mx.concatenate([out["aot"], outr["aot"]])
        l = l + 1.0 * mx.mean(nn.losses.binary_cross_entropy(
            logits, y_aot, with_logits=True))
        return l

    lg = nn.value_and_grad(model, loss_fn)
    steps = max(len(starts) // B // 4, 60)
    for ep in range(epochs):
        tl = 0.0
        for _ in range(steps):
            b = rng.choice(starts, B, replace=False)
            x = np.stack([px[i:i + T + Kmax] for i in b]).astype(np.float32)
            x = x / 127.5 - 1.0
            xr = x[:, ::-1].copy()
            tgt = np.stack([vec[i:i + T] for i in b])
            tgt /= np.linalg.norm(tgt, axis=-1, keepdims=True) + 1e-8
            y = mx.array(np.concatenate([np.ones(B), np.zeros(B)])
                         .astype(np.float32))
            loss, grads = lg(model, mx.array(x), mx.array(xr),
                             mx.array(tgt), y)
            try:
                grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            except AttributeError:
                pass
            opt.update(model, grads)
            model.base.cell.mask = mx.array(frozen)
            mx.eval(model.parameters(), opt.state)
            tl += float(loss.item())
        print(f"  A ep{ep} loss {tl / steps:.4f}", flush=True)
    return model


def stage_b(model, adapter, px, sid, ts, streams, val, epochs=6, lr=3e-4,
            seed=0, freeze_trunk=True):
    """Verb-focused contrastive against 7B captions with swap negatives.

    Trunk frozen by default: A5 measured that the warm-start trunk already
    carries the separability (0.70 untrained) and that trunk training spends
    it. L3 trains the two things the probes actually measure — the context
    projection and the text adapter — with supervision aimed at retrieval."""
    from elidedb import Store
    from elidedb.context import embed_texts
    rng = np.random.default_rng(seed)
    if freeze_trunk:
        model.base.freeze()
    db = Store.open("lake/bridge4h")
    caps = db.table("context_captions").scan()
    smap = {s: i for i, s in enumerate(streams)}
    wins, texts = [], []
    for s, a, b, c in zip(caps.column("stream").to_pylist(),
                          caps.column("ts").to_pylist(),
                          caps.column("t1").to_pylist(),
                          caps.column("caption").to_pylist()):
        if s not in smap or not c:
            continue
        m = np.where((sid == smap[s]) & (ts >= int(a)) & (ts <= int(b)))[0]
        if len(m) >= 4 and not val[m].any():
            wins.append(m)
            texts.append(c)
    negs = [swap_verbs(t, rng) for t in texts]
    print(f"  B: {len(wins)} caption windows, "
          f"{sum(n is not None for n in negs)} hard negatives", flush=True)
    # VERB-BALANCED SAMPLING. Measured: this corpus's captions run 16:1
    # pick/place vs close, so uniform sampling teaches the adapter to map
    # every query toward the majority verb mode (put-in AUC 0.83 while close
    # and open sank below chance). Each caption is weighted by the inverse
    # sqrt document-frequency of its rarest verb from the GENERIC swap
    # vocabulary — frequencies come from the corpus, words from plain
    # English; nothing dataset-specific enters.
    vocab = {w for pair in VERB_SWAPS for term in pair for w in term.split()}
    docs = [set(re.findall(r"[a-z]+", t.lower())) for t in texts]
    df = {}
    for d in docs:
        for w in d & vocab:
            df[w] = df.get(w, 0) + 1
    med = float(np.median(list(df.values()))) if df else 1.0
    wt = np.array([1.0 / np.sqrt(min((df[w] for w in d & vocab),
                                     default=med)) for d in docs])
    wt = wt / wt.sum()
    tpos = embed_texts(texts)
    tneg_idx = [i for i, n in enumerate(negs) if n]
    tneg = embed_texts([negs[i] for i in tneg_idx]) if tneg_idx else None
    neg_of = {i: j for j, i in enumerate(tneg_idx)}

    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    frozen = np.array(model.base.cell.mask)
    B = 8

    def loss_fn(m, a, xs, tp, tn, has_n):
        # IN-BATCH sigmoid contrastive (the actual SigLIP loss). The first
        # run used 1 positive + 1 swap negative per clip and the model found
        # the degenerate optimum: EVERY clip vector collapsed to one constant
        # direction (measured: pairwise cos 0.9993 across episodes, centroids
        # at 1.0000) while the adapter split caption verbs across two poles —
        # loss 0.77 with zero grounding. Off-diagonal terms make a constant
        # clip vector unsatisfiable: it would score every caption equally.
        cvs = []
        for x in xs:
            out = m.run(x[None])
            w = out["gates"][0] + 1e-3
            w = w / mx.sum(w)
            cv = mx.sum(out["ctx"][0] * w[:, None], axis=0)
            cvs.append(cv * mx.rsqrt(mx.sum(cv * cv) + 1e-8))
        CV = mx.stack(cvs)                       # (B, ctx)
        ZP = a(tp)                               # (B, ctx)
        logits = 10.0 * CV @ ZP.T - 2.0
        eye = mx.eye(len(xs))
        pair = nn.softplus(-(2 * eye - 1) * logits)
        nb = max(len(xs) - 1, 1)
        l = (mx.sum(pair * eye) / len(xs)
             + mx.sum(pair * (1 - eye)) / (len(xs) * nb))
        # verb-swap hard negatives on top: same clip, inverted caption
        zn = a(tn)
        hn = [nn.softplus(10.0 * mx.sum(CV[j] * zn[h]) - 2.0)
              for j, h in enumerate(has_n) if h >= 0]
        if hn:
            l = l + mx.sum(mx.stack(hn)) / len(hn)
        return l

    params = {"m": model, "a": adapter}

    def joint_loss(p, xs, tp, tn, has_n):
        return loss_fn(p["m"], p["a"], xs, tp, tn, has_n)

    lg = nn.value_and_grad(params, joint_loss) if False else None
    # MLX value_and_grad over a dict of Modules: grad both via a wrapper
    class Joint(nn.Module):
        def __init__(self, m, a):
            super().__init__()
            self.m = m
            self.a = a
    joint = Joint(model, adapter)

    def jl(j, xs, tp, tn, has_n):
        return loss_fn(j.m, j.a, xs, tp, tn, has_n)
    lg = nn.value_and_grad(joint, jl)

    steps = max(len(wins) // B, 40)
    for ep in range(epochs):
        tl = 0.0
        for st in range(steps):
            idx = rng.choice(len(wins), B, replace=False, p=wt)
            xs = [mx.array(px[wins[i]].astype(np.float32) / 127.5 - 1.0)
                  for i in idx]
            tp = mx.array(tpos[idx])
            hn = [neg_of.get(int(i), -1) for i in idx]
            loss, grads = lg(joint, xs, tp,
                             mx.array(tneg) if tneg is not None
                             else mx.zeros((1, 1152)), hn)
            try:
                grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            except AttributeError:
                pass
            opt.update(joint, grads)
            model.base.cell.mask = mx.array(frozen)
            mx.eval(joint.parameters(), opt.state)
            tl += float(loss.item())
        # collapse monitor: mean pairwise cos of clip vectors in a probe
        # batch. ~1.0 means the constant-vector shortcut is winning again.
        pi = rng.choice(len(wins), min(B, len(wins)), replace=False)
        cvs = []
        for i in pi:
            out = model.run(mx.array(
                px[wins[i]].astype(np.float32) / 127.5 - 1.0)[None])
            w = out["gates"][0] + 1e-3
            w = w / mx.sum(w)
            cv = mx.sum(out["ctx"][0] * w[:, None], axis=0)
            cvs.append(cv * mx.rsqrt(mx.sum(cv * cv) + 1e-8))
        C = mx.stack(cvs)
        n = len(pi)
        spread = float((mx.sum(C @ C.T) - n) / (n * (n - 1)))
        print(f"  B ep{ep} loss {tl / steps:.4f} clip-cos {spread:.3f}",
              flush=True)
    return model, adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="a", choices=["a", "b", "ab"])
    ap.add_argument("--a-epochs", type=int, default=8)
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--b-epochs", type=int, default=6)
    ap.add_argument("--reset-heads", action="store_true",
                    help="reinit ctx head, motion, adapter before stage B "
                         "(escape a collapsed checkpoint)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    px, vec, sid, ts, val = load_cache()
    streams = list(np.load(CACHE / "streams.npy"))
    print(f"cache: {len(px):,} frames | train {int((~val).sum()):,} "
          f"| val {int(val.sum()):,}", flush=True)
    groups = episode_groups(sid, ts, streams)
    print("probe episodes:", {k: len(v) for k, v in groups.items()},
          flush=True)

    if OUT.exists() and args.stage == "b":
        from elidedb.fdnnv2 import load_v2
        model, adapter, _ = load_v2(OUT)
        print("loaded existing V2 for stage B", flush=True)
        if args.reset_heads:
            import mlx.nn as mnn
            G = model.base.cfg["glimpse"]
            C = model.base.channels
            model.motion = mnn.Linear(G, model.motion_dim)
            model.head_ctx = mnn.Linear(G + C + model.motion_dim,
                                        model.cfg["ctx_dim"])
            adapter = TextAdapter()
            mx.eval(model.parameters(), adapter.parameters())
            print("reset ctx-path heads + adapter", flush=True)
    else:
        base, _ = load_encoder("lake/bridge/models/fdnnv")
        model = FDNNv2(base)
        adapter = TextAdapter()
        mx.eval(model.parameters(), adapter.parameters())

    print("[baseline]", probes(model, px, vec, sid, ts, val, groups, rng),
          flush=True)

    report = {}
    if args.stage in ("a", "ab"):
        t0 = time.time()
        stage_a(model, px, vec, sid, ts, val, epochs=args.a_epochs,
                seed=args.seed, freeze_trunk=args.freeze_trunk)
        report["stage_a"] = probes(model, px, vec, sid, ts, val, groups, rng)
        report["stage_a"]["train_s"] = round(time.time() - t0, 1)
        print("[stage A ]", report["stage_a"], flush=True)
        save_v2(model, adapter, {"report": report}, OUT)
    if args.stage in ("b", "ab"):
        t0 = time.time()
        model, adapter = stage_b(model, adapter, px, sid, ts, streams, val,
                                 epochs=args.b_epochs, seed=args.seed,
                                 freeze_trunk=args.freeze_trunk)
        report["stage_b"] = probes(model, px, vec, sid, ts, val, groups, rng)
        report["stage_b"]["train_s"] = round(time.time() - t0, 1)
        print("[stage B ]", report["stage_b"], flush=True)
        save_v2(model, adapter, {"report": report}, OUT)
    Path("bench_fdnnv2.json").write_text(json.dumps(report, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
