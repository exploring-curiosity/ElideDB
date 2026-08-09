"""Cross-view invariance head. The supervision is FREE and physical:
two cameras filmed the same moment, so their window representations
must agree - InfoNCE across views, in-batch negatives, plus a
temporal-jitter positive (same moment, boundaries +-15%) for phase
robustness. No labels, self-generated training corpus (seeds 6000+),
eval corpus untouched.

Why a trained head now, after three measured nulls on hand-operators:
the oracle-cut ceiling said RANKING DEPTH is the whole yield/prec
game, and the deep tail is invariance the fixed operator lacks -
colour/shape/row-position were each ruled out by measurement, so the
residual is structure only a learned map can absorb.

    python native/vwm_head.py            # train, save wm_head.pt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "vwm"
OUT = ROOT / "native" / "wm_head.pt"

D_ROW = 20 * 384
D_P = 768              # per-frame JL projection of the row profile
SEGS = (3, 5, 8)
D_IN = sum(SEGS) * D_P
D_H = 2048
D_OUT = 512
TEMP = 0.07
BS = 256
STEPS = 2500
LR = 1e-3
SEED = 0


def rproj_row():
    rs = np.random.RandomState(17)
    return (rs.randn(D_ROW, D_P) / np.sqrt(D_P)).astype(np.float32)


def _l2(V, ax=-1):
    V = np.asarray(V, np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=ax, keepdims=True), 1e-8)


def dmulti_p(rp, s, e):
    """(T, D_P) projected rows -> ordered-change rep of window [s,e)."""
    parts = []
    for n2 in SEGS:
        cuts = [s + (e - s) * i // n2 for i in range(n2 + 1)]
        cuts[-1] = e - 1
        parts += [_l2(rp[cuts[i + 1]] - rp[cuts[i]]) for i in range(n2)]
    return np.concatenate(parts)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, D_H), nn.GELU(),
            nn.Linear(D_H, D_H), nn.GELU(),
            nn.Linear(D_H, D_OUT))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def load_train():
    """episode -> list of per-view projected row-profile series."""
    P = rproj_row()
    eps = {}
    files = sorted(CACHE.glob("sim_train_*_ep*_cam*_v3.npz"))
    from tqdm import tqdm
    for f in tqdm(files, unit="file", desc="load"):
        key = f.stem.rsplit("_cam", 1)[0]
        r = np.load(f)["r20"].astype(np.float32) @ P
        eps.setdefault(key, []).append(_l2(r))
    return {k: v for k, v in eps.items() if len(v) >= 2}


def main():
    torch.manual_seed(SEED)
    eps = load_train()
    keys = sorted(eps)
    print(f"{len(keys)} train episodes with 2 views")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Head().to(dev)
    print(f"head {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"params on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    rs = np.random.RandomState(SEED)
    from tqdm import tqdm
    losses = []
    for step in tqdm(range(STEPS), unit="step", desc="train"):
        A, B = [], []
        for _ in range(BS):
            k = keys[rs.randint(len(keys))]
            views = eps[k]
            T = min(len(v) for v in views)
            L = rs.randint(20, min(81, T))
            s = rs.randint(0, T - L + 1)
            # view A: as-is; view B: OTHER camera + boundary jitter
            j = max(1, int(0.15 * L))
            s2 = int(np.clip(s + rs.randint(-j, j + 1), 0, T - L))
            e2 = min(T, s2 + L + rs.randint(-j, j + 1))
            if e2 - s2 < 12:
                e2 = s2 + 12
            va, vb = rs.permutation(2)[:2]
            A.append(dmulti_p(views[va], s, s + L))
            B.append(dmulti_p(views[vb], s2, min(e2, T)))
        xa = torch.as_tensor(np.stack(A), device=dev)
        xb = torch.as_tensor(np.stack(B), device=dev)
        za, zb = model(xa), model(xb)
        logits = za @ zb.T / TEMP
        tgt = torch.arange(len(za), device=dev)
        loss = 0.5 * (nn.functional.cross_entropy(logits, tgt)
                      + nn.functional.cross_entropy(logits.T, tgt))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(float(loss))
        if step % 250 == 0:
            print(f"  step {step}: loss {np.mean(losses[-250:]):.3f}")
    torch.save(model.state_dict(), OUT)
    (ROOT / "native" / "wm_head.json").write_text(json.dumps(dict(
        d_p=D_P, segs=list(SEGS), d_h=D_H, d_out=D_OUT, temp=TEMP,
        bs=BS, steps=STEPS, lr=LR, seed=SEED,
        loss_tail=float(np.mean(losses[-250:])),
        time=time.strftime("%F %T")), indent=1))
    print(f"saved {OUT.name}; final loss {np.mean(losses[-250:]):.3f}")


def load(device=None):
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    m = Head().to(dev)
    m.load_state_dict(torch.load(OUT, map_location=dev, weights_only=True))
    m.eval()
    return m


if __name__ == "__main__":
    main()
