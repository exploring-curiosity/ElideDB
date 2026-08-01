"""Distill V-JEPA2 into an FDNN student, and report both numbers.

One channel at a time, and each one answers the same two questions:
does the student RANK like the teacher, and what does it cost. Cosine
alone is not an answer - on PE a constant prediction of the corpus mean
scored 0.8629, so a student at 0.9168 had learned nothing. Every report
carries the mean baseline beside it.

V-JEPA2 first because it is the user's example and because it backs TWO
channels: the `vjepa` world-model channel and the `act` action probe
share one ViT-L encoder, so one student replaces both.

  python scripts/distill_vjepa.py --episodes 300 --epochs 40
  python scripts/distill_vjepa.py --teacher-only --episodes 300
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.video import FrameSet                           # noqa: E402

CACHE = ROOT / "scratch_distill"
SIZE, FRAMES = 128, 16


def clips(db, n_ep, size=SIZE, frames=FRAMES):
    """One fixed-shape clip per episode: the student's and teacher's input.

    Uniform sampling across the episode, not the first N frames - an
    event that happens late is invisible to a prefix.
    """
    import cv2
    ep = db.table("episodes").scan().to_pydict()
    ft = db.table("frames").scan()
    sel = np.arange(min(n_ep, len(ep["ts"])))
    out, keys = [], []
    for i in sel:
        s, a, b = str(ep["stream"][i]), int(ep["ts"][i]), int(ep["t1"][i])
        sub = ft.filter(pc.and_(
            pc.equal(ft.column("stream"), s),
            pc.and_(pc.greater_equal(ft.column("ts"), a),
                    pc.less_equal(ft.column("ts"), b))))
        if len(sub) < 4:
            continue
        try:
            dec = sorted(FrameSet(db, "frames", sub).decode())
        except Exception:
            continue
        idx = np.linspace(0, len(dec) - 1, frames).round().astype(int)
        v = np.stack([cv2.resize(dec[j][1], (size, size),
                                 interpolation=cv2.INTER_AREA) for j in idx])
        out.append(v.astype(np.uint8))
        keys.append((s, a))
    return np.stack(out), keys


def teacher(clips_u8, batch=4, pool="mean"):
    """V-JEPA2 embeddings. Paid ONCE, cached, never at ingest.

    pool="mean"      mean over patch tokens - what the shipping vjepa
                     channel does, and what collapses to effective rank
                     16 of 1024 dims.
    pool="attentive" the SSv2 attentive pooler's OUTPUT, taken before
                     its 174-class head. A learned cross-attention query
                     over the tokens - the aggregation the teacher was
                     actually trained with, rather than the one that is
                     convenient. Mean pooling weights the unchanged room
                     the same as the hand that moves.
    """
    import torch
    from transformers import AutoModel
    from elidedb.device import pick
    MID = "facebook/vjepa2-vitl-fpc64-256"
    dev, dtype = pick()
    m = AutoModel.from_pretrained(MID, dtype=dtype,
                                  low_cpu_mem_usage=True).to(dev).eval()
    probe = None
    if pool == "attentive":
        from elidedb.action_probe import _build_probe, PROBE_CKPT
        probe = _build_probe()
        sd = torch.load(PROBE_CKPT, map_location="cpu",
                        weights_only=False)["classifiers"][0]
        probe.load_state_dict({k.replace("module.", ""): v
                               for k, v in sd.items()}, strict=True)
        probe = probe.to(dev).float().eval()
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    out, t0 = [], time.time()
    for i in range(0, len(clips_u8), batch):
        c = clips_u8[i:i + batch].astype(np.float32) / 255.0
        c = (c - mean) / std
        x = torch.tensor(c, device=dev, dtype=dtype).permute(0, 1, 4, 2, 3)
        with torch.no_grad():
            h = m.get_vision_features(x)
            if probe is None:
                v = h.float().mean(1)
            else:
                # the pooled feature, NOT the 174 logits: the logits are
                # SSv2's label space, the pooled feature is the
                # representation the label space was read off
                v = probe.pooler(h.float()).squeeze(1)
        out.append(v.cpu().numpy())
        if (i // batch) % 20 == 0:
            print(f"  teacher {i}/{len(clips_u8)}  "
                  f"{(time.time() - t0):.0f}s", flush=True)
    T = np.concatenate(out)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8), \
        (time.time() - t0) / len(clips_u8)


def rank_stats(P, Y):
    """Rank agreement with the teacher - the only thing that matters for
    a retrieval channel - plus the trivial baseline cosine."""
    n = min(len(P), 400)
    Sp, St = P[:n] @ P[:n].T, Y[:n] @ Y[:n].T
    np.fill_diagonal(Sp, -9); np.fill_diagonal(St, -9)
    k = min(10, n - 1)
    rp, rt = np.argsort(-Sp, 1)[:, :k], np.argsort(-St, 1)[:, :k]
    mu = Y.mean(0); mu /= np.linalg.norm(mu) + 1e-8
    return {"cosine": round(float(np.mean(np.sum(P * Y, 1))), 4),
            "mean_baseline": round(float(np.mean(Y @ mu)), 4),
            "nn_top1": round(float(np.mean(Sp.argmax(1) == St.argmax(1))), 4),
            "nn_recall@10": round(float(np.mean(
                [len(set(a) & set(b)) / k for a, b in zip(rp, rt)])), 4)}


def main():
    argv = sys.argv
    store = argv[argv.index("--store") + 1] if "--store" in argv else "lake/bridge4h"
    n_ep = int(argv[argv.index("--episodes") + 1] if "--episodes" in argv else 300)
    epochs = int(argv[argv.index("--epochs") + 1] if "--epochs" in argv else 40)
    CACHE.mkdir(exist_ok=True)
    pool = argv[argv.index("--pool") + 1] if "--pool" in argv else "mean"
    cf = CACHE / f"vjepa_{Path(store).name}_{n_ep}_{pool}.npz"

    if cf.exists():
        z = np.load(cf)
        V, Y, t_teach = z["V"], z["Y"], float(z["t_teach"])
        print(f"cached: {len(V)} clips, teacher {t_teach * 1000:.0f} ms/clip")
    else:
        db = Store.open(store)
        a = time.time()
        V, keys = clips(db, n_ep)
        print(f"{len(V)} clips decoded in {time.time() - a:.0f}s")
        Y, t_teach = teacher(V, pool=pool)
        np.savez(cf, V=V, Y=Y, t_teach=t_teach,
                 keys=np.array(keys, dtype=object))
        print(f"teacher: {t_teach * 1000:.0f} ms/clip")
    if "--teacher-only" in argv:
        return

    import mlx.core as mx
    import mlx.nn as mnn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from elidedb.fdnnstudent import VideoStudent

    ntr = int(len(V) * 0.8)
    Xtr = mx.array(V[:ntr].astype(np.float32) / 127.5 - 1.0)
    Xte = mx.array(V[ntr:].astype(np.float32) / 127.5 - 1.0)
    Ytr, Yte = mx.array(Y[:ntr]), mx.array(Y[ntr:])

    net = VideoStudent(out_dim=Y.shape[1], size=SIZE, frames=FRAMES)
    n_par = sum(x.size for _, x in tree_flatten(net.parameters()))
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    TAU = 0.05

    def loss_fn(model, x, y):
        p = model(x)
        # InfoNCE: the student must rank ITS clip's teacher vector above
        # every other clip's in the batch. Cosine regression cannot -
        # on an anisotropic teacher the corpus mean already scores high.
        logit = (p @ y.T) / TAU
        lab = mx.arange(x.shape[0])
        return 0.5 * (mnn.losses.cross_entropy(logit, lab, reduction="mean")
                      + mnn.losses.cross_entropy(logit.T, lab,
                                                 reduction="mean"))

    step = mnn.value_and_grad(net, loss_fn)
    bs = 16
    a = time.time()
    for ep in range(epochs):
        perm = np.random.permutation(ntr)
        tot = 0.0
        for i in range(0, ntr - bs + 1, bs):
            s = mx.array(perm[i:i + bs])
            l, g = step(net, Xtr[s], Ytr[s])
            opt.update(net, g)
            mx.eval(net.parameters(), opt.state)
            tot += float(l)
        if (ep + 1) % 10 == 0:
            P = np.array(net(Xte))
            print(f"  epoch {ep + 1:3d}  loss {tot / max(ntr // bs, 1):.4f}  "
                  f"{rank_stats(P, np.array(Yte))}", flush=True)
    t_train = time.time() - a

    P = np.array(net(Xte))
    fid = rank_stats(P, np.array(Yte))
    a = time.time()
    for _ in range(5):
        mx.eval(net(Xte[:16]))
    t_stu = (time.time() - a) / 5 / 16

    out = {"channel": "vjepa", "teacher": "facebook/vjepa2-vitl-fpc64-256",
           "clips": len(V), "student_params_M": round(n_par / 1e6, 2),
           "teacher_params_M": 300,
           "train_seconds": round(t_train, 1), "fidelity": fid,
           "teacher_ms_per_clip": round(t_teach * 1000, 2),
           "student_ms_per_clip": round(t_stu * 1000, 3),
           "speedup": round(t_teach / max(t_stu, 1e-9), 1)}
    print(json.dumps(out, indent=1))
    (ROOT / "bench" / "bench_distill_vjepa.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
