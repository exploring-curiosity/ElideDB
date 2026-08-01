"""Train one channel student and report what it costs to replace its teacher.

Held-out by EPISODE, not by row. PE emits eight windows per episode and
they share frames, so a random row split leaks the answer across the
boundary and reports a fidelity the student does not have.

  python scripts/distill_channel.py pe --store lake/bridge4h
  python scripts/distill_channel.py sig2 --store lake/fresh_bench
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.distill import DIMS, Head, fidelity, init, pairs  # noqa: E402

OUT = ROOT / "models/channels"
# InfoNCE temperature. Low enough that near-duplicate spans still
# separate; the batch is the negative pool, so 128 spans per step.
TAU = 0.05


def main():
    argv = sys.argv
    ch = argv[1]
    store = argv[argv.index("--store") + 1] if "--store" in argv else "lake/bridge4h"
    epochs = int(argv[argv.index("--epochs") + 1] if "--epochs" in argv else 60)
    hidden = int(argv[argv.index("--hidden") + 1] if "--hidden" in argv else 512)
    db = Store.open(store)

    a = time.time()
    X, Y, keys = pairs(db, ch)
    t_data = time.time() - a
    if not X:
        raise SystemExit(f"no training pairs for {ch} in {store}")
    d_out = Y.shape[1]

    # split by EPISODE: shared frames across a row split leak the answer
    eps = sorted({k[1] for k in keys})
    cut = eps[int(len(eps) * 0.8)]
    tr = [i for i, k in enumerate(keys) if k[1] < cut]
    te = [i for i, k in enumerate(keys) if k[1] >= cut]
    print(f"{ch}: {len(X)} spans over {len(eps)} episodes, "
          f"{len(tr)} train / {len(te)} test, teacher dim {d_out}, "
          f"pairs built in {t_data:.1f}s")

    import torch
    import torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    class Net(nn.Module):
        """Same skeleton the numpy Head runs: attention pool, pre-norm
        residual MLP, project, normalise."""

        def __init__(self, d_in, d_out, hidden):
            super().__init__()
            self.q = nn.Parameter(torch.randn(d_in) / d_in ** 0.5)
            self.norm = nn.LayerNorm(d_in)
            self.f = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                   nn.Linear(hidden, d_in))
            self.p = nn.Linear(d_in, d_out)

        def forward(self, X, mask):
            s = (X @ self.q).masked_fill(~mask, -1e9)
            a = torch.softmax(s, 1).unsqueeze(-1)
            h = (a * X).sum(1)
            h = self.norm(h)
            h = h + self.f(h)
            y = self.p(h)
            return y / (y.norm(dim=-1, keepdim=True) + 1e-8)

    d_in = X[0].shape[1]
    L = max(len(x) for x in X)
    def pad(idx):
        B = np.zeros((len(idx), L, d_in), np.float32)
        M = np.zeros((len(idx), L), bool)
        for j, i in enumerate(idx):
            B[j, :len(X[i])] = X[i]; M[j, :len(X[i])] = True
        return (torch.tensor(B, device=dev), torch.tensor(M, device=dev),
                torch.tensor(Y[idx], device=dev))

    Xtr, Mtr, Ytr = pad(tr)
    Xte, Mte, Yte = pad(te)
    net = Net(d_in, d_out, hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    n_par = sum(p.numel() for p in net.parameters())

    a = time.time()
    bs = 128
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), bs):
            s = perm[i:i + bs]
            P = net(Xtr[s], Mtr[s])
            # InfoNCE, NOT cosine regression. Cosine regression on this
            # teacher is nearly a no-op: PE's space is anisotropic (mean
            # pairwise cosine 0.885), so a constant prediction of the
            # corpus mean already scores 0.8629 and the first student
            # reached 0.9168 with nearest-neighbour agreement of 0.005.
            # It matched the mean and learned no episode.
            #
            # Contrastive loss cannot be satisfied that way: the student
            # must score ITS span's teacher vector above every OTHER
            # span's in the batch, which is the ranking the channel is
            # actually consumed for. The shared mean cancels because it
            # is in every term of the denominator.
            logit = P @ Ytr[s].T / TAU
            lab = torch.arange(len(s), device=dev)
            loss = 0.5 * (nn.functional.cross_entropy(logit, lab)
                          + nn.functional.cross_entropy(logit.T, lab))
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            net.eval()
            with torch.no_grad():
                Pt = net(Xte, Mte)
                c = (Pt * Yte).sum(-1).mean().item()
                S = Pt @ Yte.T
                r1 = (S.argmax(1) == torch.arange(len(te),
                                                  device=dev)).float().mean().item()
            print(f"  epoch {ep + 1:3d}  loss {loss.item():.4f}  "
                  f"test cosine {c:.4f}  retrieve-own-teacher@1 {r1:.4f}",
                  flush=True)
    t_train = time.time() - a

    # export to numpy: the write path runs two matmuls, no framework
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    w = {"q": sd["q"], "g": sd["norm.weight"], "beta": sd["norm.bias"],
         "W1": sd["f.0.weight"].T, "b1": sd["f.0.bias"],
         "W2": sd["f.2.weight"].T, "b2": sd["f.2.bias"],
         "Wp": sd["p.weight"].T, "bp": sd["p.bias"]}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{ch}.npz", **w)
    head = Head(w)

    Xte_l = [X[i] for i in te]
    fid = fidelity(head, Xte_l, Y[te])
    a = time.time()
    for _ in range(3):
        head.batch(Xte_l)
    ms_span = (time.time() - a) / 3 / max(len(Xte_l), 1) * 1000
    spans_per_ep = len(X) / len(eps)

    out = {"channel": ch, "store": store, "params": n_par,
           "train_seconds": round(t_train, 1),
           "test_episodes": len(eps) - len(set(k[1] for k in keys if k[1] < cut)),
           "fidelity": fid,
           "student_ms_per_span": round(ms_span, 4),
           "spans_per_episode": round(spans_per_ep, 2),
           "student_ms_per_episode": round(ms_span * spans_per_ep, 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "bench" / f"bench_distill_{ch}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
