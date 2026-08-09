"""World-model core v1: a causal predictor over frozen per-frame latents.

The design conclusion of RESEARCH_WM.md, built: the latent of a clip is
the PREDICTOR'S STATE TRAJECTORY over it, not a pooled frame embedding.
This file trains that predictor with the LeWorldModel recipe - exactly
two losses:

  1. next-latent prediction (1 - cosine) - the state keeps what the
     future depends on, which is a definition of "worth remembering"
     that needs no vocabulary;
  2. an isotropic-Gaussian regularizer on the states (SIGReg-style:
     random 1-D projections tested against N(0,1) via an Epps-Pulley
     characteristic-function statistic) - the LeJEPA anti-collapse
     term, one trade-off knob, no EMA, no stop-gradient.

No labels, no verbs, no domain info touch training - the inputs are
frozen vits latents of self-generated MuJoCo episodes (seed-disjoint
from the eval corpus), and ground truth exists only inside the gates
(vwm_gates.py). Read and write paths use the same operator: run the
predictor over any clip's latents, store/compare its states.

    python native/vwm.py                # train, save wm_v1.pt
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
OUT = ROOT / "native" / "wm_v1.pt"

L = 64            # window length (6.4s at 10fps)
STRIDE = 16
D_IN = 384        # vits16 global latent
# prediction horizons in FRAMES (0.1/0.5/1.5s at 10fps). At horizon 1
# alone the task was a COPY: val 1-cos hit 0.018 because the next
# frame is nearly the current one, so the state could be a low-pass of
# the frozen latent and retrieval gained nothing over frozen features
# (measured: P@10 0.352 vs frozen 0.416 once position leakage was
# removed). Longer horizons force the state to carry dynamics.
HORIZONS = (1, 5, 15)
D = 384
LAYERS = 6
HEADS = 6
FF = 1536
LAMBDA = 0.5      # the one trade-off knob (pred vs gaussian)
LR = 3e-4
BS = 64
EPOCHS = 12
SEED = 0


class Predictor(nn.Module):
    """Block-causal next-latent predictor. states() is the READ/WRITE
    operator: per-frame hidden state + prediction residual."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(D_IN, D)
        # NO positional embedding (NoPE): with a learned absolute
        # position the state carried WHERE-IN-THE-FILE it sat, and
        # retrieval degenerated - every query matched episode-START
        # spans because query states are recomputed from position 0.
        # The causal mask already supplies order asymmetry, and a
        # memory must be time-shift invariant: the state is the
        # content of the past, not its file offset.
        layer = nn.TransformerEncoderLayer(
            D, HEADS, FF, dropout=0.0, batch_first=True,
            norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, LAYERS)
        self.out = nn.ModuleList(
            [nn.Linear(D, D_IN) for _ in HORIZONS])

    def forward(self, x):                      # (B, T, D_IN)
        T = x.shape[1]
        h = self.inp(x)
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device)
        h = self.enc(h, mask=mask, is_causal=True)
        return h, [o(h) for o in self.out]     # states, per-horizon preds

    @torch.no_grad()
    def states(self, g, device=None, chunk=512):
        """(T, D_IN) numpy -> per-frame states (T, D) + surprise (T,).
        surprise[t] = 1 - cos(pred[t-1], g[t]); the write-path record."""
        self.eval()
        dev = device or next(self.parameters()).device
        x = torch.as_tensor(np.asarray(g, np.float32), device=dev)[None]
        hs, ps = [], []
        for s in range(0, x.shape[1], chunk):
            xx = x[:, max(0, s - L):s + chunk]
            h, p = self._fwd_h1(xx)
            off = s - max(0, s - L)
            hs.append(h[0, off:])
            ps.append(p[0, off:])
        h = torch.cat(hs); p = torch.cat(ps)
        tgt = x[0, 1:]
        pn = torch.nn.functional.normalize(p[:-1], dim=-1)
        tn = torch.nn.functional.normalize(tgt, dim=-1)
        sur = torch.zeros(x.shape[1], device=dev)
        sur[1:] = 1.0 - (pn * tn).sum(-1)
        return h.cpu().numpy(), sur.cpu().numpy()

    @torch.no_grad()
    def _fwd_h1(self, xx):
        h, ps = self(xx)
        return h, ps[0]


def sigreg(h, k=64, ts=(0.5, 1.0, 1.5, 2.0, 2.5)):
    """Epps-Pulley against N(0,1) over k random 1-D projections of the
    states. Differentiable: characteristic function via cos/sin means."""
    z = h.reshape(-1, h.shape[-1])
    u = torch.randn(h.shape[-1], k, device=h.device)
    u = u / u.norm(dim=0, keepdim=True)
    p = z @ u                                   # (N, k)
    loss = 0.0
    for t in ts:
        re = torch.cos(t * p).mean(0)
        im = torch.sin(t * p).mean(0)
        target = float(np.exp(-t * t / 2.0))
        loss = loss + ((re - target) ** 2 + im ** 2).mean()
    return loss / len(ts)


def load_windows(roots):
    eps = []
    for r in roots:
        pref = "_".join((ROOT / r).resolve()
                        .relative_to(ROOT / "data").parts)
        eps += sorted(CACHE.glob(f"{pref}_ep*.npz"))
    rs = np.random.RandomState(SEED)
    rs.shuffle(eps)
    n_val = max(1, len(eps) // 10)
    val_eps, tr_eps = eps[:n_val], eps[n_val:]

    def wins(files):
        X = []
        for f in files:
            g = np.load(f)["g"].astype(np.float32)
            for s in range(0, max(1, len(g) - L), STRIDE):
                w = g[s:s + L]
                if len(w) == L:
                    X.append(w)
        return np.stack(X) if X else np.zeros((0, L, D_IN), np.float32)

    return wins(tr_eps), wins(val_eps), len(tr_eps), len(val_eps)


def main():
    torch.manual_seed(SEED)
    roots = sys.argv[1:] or ["data/sim_train/panda", "data/sim_train/xarm7",
                             "data/sim_train/vx300s"]
    Xtr, Xva, ntr, nva = load_windows(roots)
    print(f"train {len(Xtr)} windows / {ntr} eps   "
          f"val {len(Xva)} windows / {nva} eps")
    assert len(Xtr) > 0, "no cached latents - run vwm_encode.py first"
    dev = ("mps" if torch.backends.mps.is_available() else "cpu")
    model = Predictor().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"predictor {n_par/1e6:.1f}M params on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * max(1, len(Xtr) // BS))
    from tqdm import tqdm
    hist = []
    Xva_t = torch.as_tensor(Xva, device=dev)
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.RandomState(SEED + ep).permutation(len(Xtr))
        pl = gl = nb = 0
        for i in tqdm(range(0, len(idx), BS), unit="batch",
                      desc=f"epoch {ep}"):
            xb = torch.as_tensor(Xtr[idx[i:i + BS]], device=dev)
            h, ps2 = model(xb)
            pred = 0.0
            for hz, p in zip(HORIZONS, ps2):
                pn = torch.nn.functional.normalize(p[:, :-hz], dim=-1)
                tn = torch.nn.functional.normalize(xb[:, hz:], dim=-1)
                pred = pred + (1.0 - (pn * tn).sum(-1).mean())
            pred = pred / len(HORIZONS)
            gau = sigreg(h)
            loss = pred + LAMBDA * gau
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            pl += float(pred); gl += float(gau); nb += 1
        model.eval()
        with torch.no_grad():
            vp = 0.0
            for s in range(0, len(Xva_t), BS):
                xb = Xva_t[s:s + BS]
                _, ps2 = model(xb)
                v = 0.0
                for hz, p in zip(HORIZONS, ps2):
                    pn = torch.nn.functional.normalize(p[:, :-hz], dim=-1)
                    tn = torch.nn.functional.normalize(xb[:, hz:], dim=-1)
                    v += float(1.0 - (pn * tn).sum(-1).mean())
                vp += (v / len(HORIZONS)) * len(xb)
            vp /= max(1, len(Xva_t))
        hist.append(dict(epoch=ep, pred=pl / nb, sig=gl / nb, val=vp))
        print(f"  epoch {ep}: pred {pl/nb:.4f}  sig {gl/nb:.4f}  "
              f"val_pred {vp:.4f}")
    torch.save(model.state_dict(), OUT)
    manifest = dict(encoder="vits16@320", pos="none (NoPE)",
                    horizons=list(HORIZONS),
                    L=L, stride=STRIDE, d=D,
                    layers=LAYERS, heads=HEADS, ff=FF, lam=LAMBDA,
                    lr=LR, bs=BS, epochs=EPOCHS, seed=SEED,
                    params=int(n_par), roots=[str(r) for r in roots],
                    train_windows=int(len(Xtr)), hist=hist,
                    time=time.strftime("%F %T"))
    (ROOT / "native" / "wm_v1.json").write_text(
        json.dumps(manifest, indent=1))
    print(f"saved {OUT.name} + manifest")


def load(device=None):
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    m = Predictor().to(dev)
    m.load_state_dict(torch.load(OUT, map_location=dev, weights_only=True))
    m.eval()
    return m


if __name__ == "__main__":
    main()
