"""EXTRACTION LADDER: probe every pipeline stage against sim truth.

The question that was never asked: does the latent actually CONTAIN
the physical state the simulator hands over for free? End-to-end
retrieval numbers cannot answer it - a chain is judged stage by stage
or not at all. Each stage of the shipped pipeline is probed (ridge,
5-fold by episode, R^2) against per-frame ground truth from the
generator's trace: hand position, gripper, block positions, tower
height, velocities. Where R^2 collapses is where extraction dies.

Stages (exactly the shipped operators):
  S0 pixels   32x24 grey thumbnail        (what a camera trivially has)
  S1 grid     20x20x384 DINOv3 patches    (frozen encoder, full)
  S2 c5       5x5 pooled grid             (store field)
  S3 r20      row marginal                (the shipped profile)
  S4 g        global mean                 (1 vector/frame)
  S5 rp       JL(768) of r20              (the shipped query space)

    SDX_ENC=vits SDX_RES=320 python native/vwm_extract.py --encode
    python native/vwm_extract.py --probe
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data" / "sim_probe"
CACHE = ROOT / "data" / "cache" / "vwm_probe"
FPS = 10
W, H = 640, 480


def _l2(V, ax=-1):
    V = np.asarray(V, np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=ax, keepdims=True), 1e-8)


def read_frames(mp4):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True)
    n = len(p.stdout) // (W * H * 3)
    return np.frombuffer(p.stdout, np.uint8).reshape(n, H, W, 3)


def encode_all():
    sys.path.insert(0, str(ROOT / "native"))
    import vcore
    from vtrans import encode
    vcore.feature_dim()
    CACHE.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    eps = sorted(PROBE.glob("ep*"))
    for ep in tqdm(eps, unit="ep", desc="encode"):
        fp = CACHE / f"{ep.name}.npz"
        if fp.exists():
            continue
        cams = sorted(ep.glob("cam*.mp4"))
        F = read_frames(cams[0])
        G = encode(F)                          # (T, 20, 20, 384)
        T = len(G)
        thumb = np.stack([f[::20, ::20].mean(-1) for f in F])  # 24x32
        np.savez(fp, G=G.astype(np.float16),
                 thumb=thumb.astype(np.float16), cam=cams[0].name)


def targets_from_trace(tr, T):
    """Physical quantities per frame, identity-free."""
    n_blk = (tr.shape[1] - 5) // 7
    hand = tr[:, 1:4]
    grip = tr[:, 4:5]
    bpos = tr[:, 5:].reshape(len(tr), n_blk, 7)[:, :, :3]
    L = min(T, len(tr))
    hand, grip, bpos = hand[:L], grip[:L], bpos[:L]
    d = np.linalg.norm(bpos - hand[:, None], axis=2)
    near = d.argmin(1)
    nb = bpos[np.arange(L), near]
    out = {
        "hand_x": hand[:, 0], "hand_y": hand[:, 1], "hand_z": hand[:, 2],
        "grip": grip[:, 0],
        "tower_top_z": bpos[:, :, 2].max(1),
        "nearblk_x": nb[:, 0], "nearblk_y": nb[:, 1],
        "nearblk_z": nb[:, 2],
        "hand_blk_dist": d.min(1),
    }
    v = np.zeros((L, 3), np.float32)
    v[1:] = (hand[1:] - hand[:-1]) * FPS
    out["handvel_x"] = v[:, 0]
    out["handvel_z"] = v[:, 2]
    bv = np.zeros(L, np.float32)
    bv[1:] = np.linalg.norm(bpos[1:] - bpos[:-1], axis=2).max(1) * FPS
    out["blk_speed_max"] = bv
    return out, L


def stages(z):
    G = z["G"].astype(np.float32)
    T = len(G)
    thumb = z["thumb"].astype(np.float32).reshape(T, -1)
    c5 = G.reshape(T, 5, 4, 5, 4, -1).mean((2, 4)).reshape(T, -1)
    r20 = G.mean(2).reshape(T, -1)
    g = _l2(G.reshape(T, -1, 384).mean(1))
    rs = np.random.RandomState(17)
    P = (rs.randn(r20.shape[1], 768) / np.sqrt(768)).astype(np.float32)
    rp = _l2(_l2(r20) @ P)
    rs2 = np.random.RandomState(23)
    PG = (rs2.randn(G.shape[1] * G.shape[2] * G.shape[3], 4096)
          / 64).astype(np.float32)
    grid = _l2(G.reshape(T, -1) @ PG)
    return {"S0_pixels": _l2(thumb), "S1_grid4096": grid,
            "S2_c5x5": _l2(c5), "S3_r20": _l2(r20), "S4_g384": g,
            "S5_rp768": rp}


def probe():
    from sklearn.linear_model import Ridge
    eps = sorted(CACHE.glob("ep*.npz"))
    X = {k: [] for k in ("S0_pixels", "S1_grid4096", "S2_c5x5",
                         "S3_r20", "S4_g384", "S5_rp768")}
    Y, ep_id = {}, []
    from tqdm import tqdm
    for f in tqdm(eps, unit="ep", desc="stages"):
        z = np.load(f)
        st = stages(z)
        tr = np.load(PROBE / f.stem / "trace.npy")
        tg, L = targets_from_trace(tr, len(z["G"]))
        for k in X:
            X[k].append(st[k][:L])
        for k, v in tg.items():
            Y.setdefault(k, []).append(v)
        ep_id += [f.stem] * L
    ep_id = np.array(ep_id)
    for k in X:
        X[k] = np.concatenate(X[k])
    for k in Y:
        Y[k] = np.concatenate(Y[k])
    print(f"{len(ep_id)} frames from {len(eps)} episodes\n")

    uniq = sorted(set(ep_id))
    rs = np.random.RandomState(0)
    rs.shuffle(uniq)
    folds = np.array_split(uniq, 5)
    names = list(Y)
    print(f"{'stage':12s}" + "".join(f"{n:>14s}" for n in names))
    for sk, Xs in X.items():
        r2s = []
        for tk in names:
            y = Y[tk]
            press, tss = 0.0, 0.0
            for f5 in folds:
                te = np.isin(ep_id, f5)
                m = Ridge(alpha=10.0)
                m.fit(Xs[~te], y[~te])
                p = m.predict(Xs[te])
                press += float(((y[te] - p) ** 2).sum())
                tss += float(((y[te] - y[te].mean()) ** 2).sum())
            r2s.append(1.0 - press / max(tss, 1e-9))
        print(f"{sk:12s}" + "".join(f"{v:14.3f}" for v in r2s))


if __name__ == "__main__":
    if "--encode" in sys.argv:
        encode_all()
    if "--probe" in sys.argv:
        probe()
