"""Predict the name VECTOR from what the stream already computed.

Naming is 94% of the write path and batching does not fix it: SigLIP
over participant crops costs 400 ms/episode however the crops are
grouped, because the work is a second vision model over pixels that
were already encoded once.

But the frame was already embedded during the stream, and the
participant's box is already known from the geometry. If a head can map
(frame embedding, box) -> the name vector SigLIP would have produced,
the second model disappears and naming becomes a matmul.

This is legitimate because names are never compared as strings anywhere
in this system - matching is cosine in name space - so the vector IS
the product. A discrete label for the inverted index comes back by
nearest neighbour against the corpus's own attested vocabulary.

Trained on the store's existing named events: the SigLIP-crop name
vectors are the target, and the truthset is untouched.

MEASURED NEGATIVE - THIS DOES NOT WORK, and the reason is instructive.

    holdout cosine                                   0.8605
    cosine of ANY name vector to the global mean     0.8556
    holdout top-1 name                               0.1330
    majority class ("black object", 659/5079)        0.1297
    random over a 397-name vocabulary                0.0025

The head learned to emit the average name vector. Its cosine is the
global-mean cosine and its top-1 is the majority-class rate, so the
0.86 that looks like signal is the null: in a name space this
anisotropic, cosine to the mean is high for everything and cannot
distinguish a banana from a drawer.

The cause is the input, not the capacity. FDNN-V's embedding is a
WHOLE-FRAME pooled vector, and six numbers of box geometry cannot
select which object in that frame is being named. There is nothing
spatially selective for the head to condition on, so the best it can do
is predict the prior.

The fix is ROI features - pool the encoder's SPATIAL map over the
participant's box (the RoIAlign move) instead of handing the head a
globally pooled vector plus coordinates. That needs the encoder to
expose pre-pooled features, which is a change to fdnnvideo rather than
to this script. Until then naming stays SigLIP-on-crops and stays
opt-in, and the write path is honest about the two operating points.

Kept as a reproduction: the trap here is that a plausible-looking
cosine hid a model that had learned nothing, and only the majority-class
baseline exposed it.

  python scripts/distill_namer.py [--store lake/bench] [--epochs 200]
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
from elidedb.embeddings import _vec_table                    # noqa: E402

OUT = ROOT / "models/student_v1/namer_fast.pt"


def build(db):
    """(frame embedding, box geometry) -> SigLIP name vector."""
    ev = db.table("events").scan().to_pydict()
    NV = np.asarray(ev["name_vec"], np.float32)
    keep = [i for i in range(len(ev["ts"]))
            if ev["name"][i] and np.linalg.norm(NV[i]) > 1e-6]
    if not keep:
        raise SystemExit("no named events to learn from")

    tb, FV = _vec_table(db, "frame_vectors")
    FV = np.asarray(FV, np.float32)
    fkey = {}
    for r, (s, a) in enumerate(zip(tb.column("stream").to_pylist(),
                                   tb.column("ts").to_pylist())):
        fkey.setdefault(str(s), []).append((int(a), r))
    for s in fkey:
        fkey[s].sort()
    starts = {s: np.array([x[0] for x in v]) for s, v in fkey.items()}

    X, Y, NAMES = [], [], []
    for i in keep:
        s = str(ev["stream"][i])
        v = fkey.get(s)
        if v is None:
            continue
        # the frame nearest the event's own timestamp
        j = int(np.searchsorted(starts[s], int(ev["ev_t0"][i])))
        j = min(max(j, 0), len(v) - 1)
        femb = FV[v[j][1]]
        box = ev["box"][i] or [0, 0, 0, 0]
        b = np.asarray(box[:4], np.float32)
        w, h = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
        geo = np.array([b[0] / 640, b[1] / 480, b[2] / 640, b[3] / 480,
                        w * h / (640 * 480), w / h], np.float32)
        X.append(np.concatenate([femb, geo]))
        Y.append(NV[i] / (np.linalg.norm(NV[i]) + 1e-8))
        NAMES.append(ev["name"][i])
    return (np.stack(X), np.stack(Y), NAMES)


def main():
    import torch
    import torch.nn as nn
    argv = sys.argv
    db = Store.open(argv[argv.index("--store") + 1]
                    if "--store" in argv else "lake/bench")
    epochs = int(argv[argv.index("--epochs") + 1]
                 if "--epochs" in argv else 200)
    X, Y, NAMES = build(db)
    print(f"{len(X)} named events, input {X.shape[1]}d -> {Y.shape[1]}d")

    # held-out by NAME so the split tests generalisation to instances,
    # not memorisation of a name it has already seen many times
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_ho = max(1, len(X) // 5)
    ho, tr = idx[:n_ho], idx[n_ho:]

    class Head(nn.Module):
        def __init__(self, d_in, d_out=1152):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d_in, 1024), nn.GELU(),
                                   nn.Linear(1024, 1024), nn.GELU(),
                                   nn.Linear(1024, d_out))

        def forward(self, x):
            y = self.f(x)
            return y / (y.norm(dim=-1, keepdim=True) + 1e-8)

    net = Head(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    Xt = torch.tensor(X[tr]); Yt = torch.tensor(Y[tr])
    Xh = torch.tensor(X[ho]); Yh = torch.tensor(Y[ho])
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = (1 - (net(Xt[b]) * Yt[b]).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            net.eval()
            with torch.no_grad():
                cs = (net(Xh) * Yh).sum(-1).mean().item()
            print(f"  epoch {ep+1:>3}  holdout cosine {cs:.4f}", flush=True)
    train_s = time.time() - t0

    # top-1 against the corpus's own attested vocabulary
    vocab = sorted(set(NAMES))
    byname = {}
    for n, y in zip(NAMES, Y):
        byname.setdefault(n, []).append(y)
    V = np.stack([np.mean(byname[n], 0) for n in vocab])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    net.eval()
    with torch.no_grad():
        P = net(Xh).numpy()
    pred = [vocab[i] for i in (P @ V.T).argmax(1)]
    true = [NAMES[i] for i in ho]
    top1 = float(np.mean([p == t for p, t in zip(pred, true)]))
    with torch.no_grad():
        cos = float((net(Xh) * Yh).sum(-1).mean())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sd": net.state_dict(), "d_in": X.shape[1],
                "vocab": vocab, "V": V}, OUT)
    print(json.dumps({
        "saved": str(OUT), "train_events": len(tr), "holdout": len(ho),
        "vocab": len(vocab), "holdout_cosine": round(cos, 4),
        "holdout_top1_name": round(top1, 4),
        "params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2),
        "train_seconds": round(train_s, 1)}, indent=1))


if __name__ == "__main__":
    main()
