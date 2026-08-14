"""The learned experience recurrence.

    z_t = f(z_{t-1}, act(t), a_t, g_t)

    a_t = pred(t+1) - act(t)                       expectation, 1024-d VECTOR
    b_t = act(t+1) - act(t)                        realised change
    g_t = [ <a_hat, b_hat>, log(|b|/|a|) ]         reality, 2 SCALARS

WHY REALITY IS TWO SCALARS. pred(t+1) - act(t+1) is the barred error channel,
and it lies in the linear span of {act(t), pred(t+1), act(t+1)} - any MLP over
that concatenation forms it in its first layer, so the bar cannot be enforced
by policy, only by architecture. Passing reality as two scalars makes the
1024-d residual unreconstructable. It also matches what the audit found:
gating is worth +0.12 over no gate while a PERFECT gate is worth only +0.02
more, so reality's job is to weight, not to carry. Hence the update

    gate = sigmoid(W_g [z_{t-1}, u_t, g_t])        reality drives the gate
    cand = tanh   (W_c [z_{t-1}, u_t])             expectation drives content
    z_t  = (1-gate) * z_{t-1} + gate * cand

WHY PHYSICS AND NOT LABELS. Targets come from relmo.vjphys - openness rate,
contact, speed, motion in the gripper frame - read from sim state at TRAINING
TIME ONLY, never at serve, never indexed. Task-family labels are used solely to
grade and never reach this file. The bet is that physical targets transfer
where lexical ones cannot: "hinge angle increasing while contact holds" is the
same fact for an unseen fridge or a real cupboard. The feasibility gate
(relmo.vjfeas) confirmed the frozen descriptor already carries it - open vs
close reads at 0.923 per step and 0.974 per episode on held-out scenes against
a 0.505 shuffled control - which is exactly the axis retrieval was discarding.

THE RECONSTRUCTION TERM exists because physics alone is a 9-number bottleneck,
and 9 numbers cannot separate a hinged door from a sliding drawer (both
articulate, both open). Asking z to also reconstruct a_t keeps the rest of the
descriptor alive while the physics loss reshapes its geometry. Its weight is an
arm, not an assumption.

    python -m relmo.vjz --arm a1 --epochs 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjphys  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402
from relmo.vjzeval import evaluate, report  # noqa: E402

REC4 = R.BASE / "vjrec4" / "rcasa_L6"
SIG = R.BASE / "vjsig" / "rcasa"
CKPT = R.BASE / "models" / "vjz"


def channels(z):
    """npz -> (a_t, g_t). b_t is consumed here and never leaves as a vector."""
    a, b = z["pred_change"].astype(np.float32), z["obs_change"].astype(np.float32)
    na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9
    nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9
    cos = ((a / na) * (b / nb)).sum(-1, keepdims=True)
    g = np.concatenate([cos, np.log(nb / na)], -1).astype(np.float32)
    return a, g


def gather(ids, dataset="rcasa", want_y=True, rec_dir=None, sig_dir=None):
    rec_dir = rec_dir or REC4
    sig_dir = sig_dir or SIG
    out = {}
    for i in sorted(ids):
        f = rec_dir / f"{i}.npz"
        if not f.exists():
            continue
        y = vjphys.load(dataset, i) if want_y else None
        if want_y and y is None:
            continue
        a, g = channels(np.load(f))
        s = sig_dir / f"{i}.npz"
        sg = np.load(s)["sig"].astype(np.float32) if s.exists() else None
        out[i] = dict(a=a, g=g, sig=sg, y=y)
    return out


def build_model(torch, nn, d_in, d_sig, d, arm, n_phys=None):
    # n_phys comes from the CHECKPOINT, not from the current vjphys.KEYS -
    # the target set grew from 9 to 11 and older checkpoints must still load.
    n_phys = len(vjphys.KEYS) if n_phys is None else n_phys
    class ZNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d_in)
            self.proj = nn.Linear(d_in, d)
            self.use_sig = d_sig > 0
            if self.use_sig:
                self.snorm = nn.LayerNorm(d_sig)
                self.sproj = nn.Linear(d_sig, d)
            n_in = d + (d if self.use_sig else 0)
            self.gate = nn.Linear(d + n_in + 2, d)
            self.cand = nn.Linear(d + n_in, d)
            self.drop = nn.Dropout(0.1)
            self.head_phys = nn.Linear(d, n_phys)
            self.head_rec = nn.Linear(d, d_in)
            self.d = d

        def forward(self, a, g, sig):
            B, T, _ = a.shape
            u = self.drop(torch.nn.functional.gelu(self.proj(self.norm(a))))
            if self.use_sig:
                v = torch.nn.functional.gelu(self.sproj(self.snorm(sig)))
                u = torch.cat([u, v], -1)
            z = a.new_zeros(B, self.d)
            Z = []
            for t in range(T):
                ut = u[:, t]
                gt = torch.sigmoid(self.gate(torch.cat([z, ut, g[:, t]], -1)))
                ct = torch.tanh(self.cand(torch.cat([z, ut], -1)))
                z = (1 - gt) * z + gt * ct
                Z.append(z)
            Z = torch.stack(Z, 1)
            return Z, self.head_phys(Z), self.head_rec(Z)
    return ZNet()


def load_ckpt(tag):
    """-> (model, meta). Used by relmo.vjood to score any corpus."""
    import torch
    import torch.nn as nn
    ck = torch.load(CKPT / f"{tag}.pt", weights_only=False, map_location="cpu")
    m = build_model(torch, nn, 1024, ck["d_sig"], ck["dim"], ck["arm"],
                    n_phys=len(ck["ymu"]))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def encode(model, ck, recs, dev="cpu", bs=64):
    """{id: dict(a,g,sig)} -> {id: (T,d)}. No physics targets needed."""
    import torch
    ids = [i for i in sorted(recs) if recs[i].get("a") is not None]
    if ck["d_sig"] and any(recs[i].get("sig") is None for i in ids):
        ids = [i for i in ids if recs[i].get("sig") is not None]
    out = {}
    model = model.to(dev)
    for k in range(0, len(ids), bs):
        chunk = ids[k:k + bs]
        A = torch.tensor(np.stack([recs[i]["a"] for i in chunk]), device=dev)
        G = torch.tensor(np.stack([recs[i]["g"] for i in chunk]), device=dev)
        S = (torch.tensor(np.stack([recs[i]["sig"] for i in chunk]), device=dev)
             if ck["d_sig"] else torch.zeros(len(chunk), 1, 1, device=dev))
        with torch.no_grad():
            Z, _, _ = model(A, G, S)
        for j, i in enumerate(chunk):
            out[i] = Z[j].cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="a1", choices=["a1", "a1sig"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wrec", type=float, default=0.1)
    ap.add_argument("--wphys", type=float, default=1.0)
    # CONTROL: permute which episode each target block belongs to. If val
    # retrieval survives this, the physics supervision is not what is working
    # and the recurrence architecture alone explains the gain.
    ap.add_argument("--shuffle-y", action="store_true")
    # Motion weighting. Most trace steps of a Close/Drawer episode carry no
    # event at all - 1.6% of its articulation happens in the first six steps -
    # so a flat per-step loss spends most of its capacity on idle state, and
    # idle state looks alike in every class. Weight each step by how much is
    # actually happening in it, normalised WITHIN the episode so a large event
    # does not simply outvote a small one.
    ap.add_argument("--wmotion", type=float, default=0.0)
    # Drop target channels from the LOSS (they stay in vjphys on disk). Chosen
    # by leave-one-out on val over the GT-physics oracle, never on test: an
    # earlier test-side pass called grip_dist and speed nuisance, and val says
    # they are among the most valuable channels (-0.051, -0.048 to drop). Only
    # d_rel_x (+0.026) and open (+0.021) hurt there.
    ap.add_argument("--drop", default="",
                    help="comma-separated vjphys.KEYS names to exclude")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from tqdm import tqdm

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    sp = load_split()
    print("loading cached records (no forward passes)...", flush=True)
    D = gather(sp["train"] | sp["val"] | sp["test"])
    tr = [i for i in sp["train"] if i in D]
    va = [i for i in sp["val"] if i in D]
    print(f"  train {len(tr)} | val {len(va)} | arm {a.arm} | dim {a.dim}",
          flush=True)

    def stack(ids, key):
        return np.stack([D[i][key] for i in ids])
    if a.shuffle_y:
        rs = np.random.default_rng(a.seed)
        perm = rs.permutation(len(tr))
        ys = [D[i]["y"] for i in tr]
        for k, i in enumerate(tr):
            D[i] = dict(D[i], y=ys[perm[k]])
        print("  CONTROL: physics targets permuted across episodes", flush=True)
    drop = [k.strip() for k in a.drop.split(",") if k.strip()]
    bad = [k for k in drop if k not in vjphys.KEYS]
    if bad:
        raise SystemExit(f"unknown target channel(s): {bad}")
    keepj = np.array([j for j, k in enumerate(vjphys.KEYS) if k not in drop])
    if drop:
        print(f"  dropping targets {drop} -> {len(keepj)} channels", flush=True)
    Ytr = stack(tr, "y")[:, :, keepj]
    ymu, ysd = Ytr.reshape(-1, Ytr.shape[-1]).mean(0), \
        Ytr.reshape(-1, Ytr.shape[-1]).std(0) + 1e-6
    Atr = stack(tr, "a")
    amu, asd = Atr.reshape(-1, 1024).mean(0), Atr.reshape(-1, 1024).std(0) + 1e-6

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    use_sig = a.arm == "a1sig"
    d_sig = D[tr[0]]["sig"].shape[-1] if use_sig and D[tr[0]]["sig"] is not None else 0

    kmov = [vjphys.KEYS.index(k) for k in ("d_open", "speed") ]

    def step_weight(ids):
        Y = stack(ids, "y")
        m = np.abs(Y[:, :, kmov]).sum(-1)
        m = m / (m.mean(1, keepdims=True) + 1e-9)      # within-episode
        return (1.0 + a.wmotion * m).astype(np.float32)

    def tens(ids):
        A = torch.tensor(stack(ids, "a"), device=dev)
        G = torch.tensor(stack(ids, "g"), device=dev)
        Y = torch.tensor((stack(ids, "y")[:, :, keepj] - ymu) / ysd,
                         device=dev).float()
        S = (torch.tensor(stack(ids, "sig"), device=dev)
             if d_sig else torch.zeros(len(ids), 1, 1, device=dev))
        Ahat = torch.tensor((stack(ids, "a") - amu) / asd, device=dev)
        W = torch.tensor(step_weight(ids), device=dev).unsqueeze(-1)
        return A, G, S, Y, Ahat, W
    TR, VA = tens(tr), tens(va)

    model = build_model(torch, nn, 1024, d_sig, a.dim, a.arm,
                        n_phys=len(keepj)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  {npar/1e3:.0f}k params on {dev} | wrec {a.wrec}", flush=True)

    def descriptors(ids, T):
        model.eval()
        with torch.no_grad():
            Z, _, _ = model(T[0], T[1], T[2])
        return {i: Z[k].cpu().numpy() for k, i in enumerate(ids)}

    best, best_state, best_ep = -1.0, None, -1
    hist = []
    bs = 32
    bar = tqdm(range(a.epochs), unit="ep", desc=f"vjz:{a.arm}")
    for ep in bar:
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        tot = 0.0
        for k in range(0, len(tr), bs):
            s = perm[k:k + bs]
            Z, P, Rc = model(TR[0][s], TR[1][s], TR[2][s] if d_sig else TR[2])
            w = TR[5][s]
            loss = a.wphys * (w * (P - TR[3][s]) ** 2).mean()
            if a.wrec:
                loss = loss + a.wrec * nn.functional.mse_loss(Rc, TR[4][s])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(s)
        if (ep + 1) % a.every == 0 or ep == a.epochs - 1:
            dv = descriptors(va, VA)
            r = evaluate(dv, va, va)
            hist.append((ep, tot / len(tr), r["overall"]))
            if r["overall"] > best:
                best, best_ep = r["overall"], ep
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            bar.set_postfix(loss=f"{tot/len(tr):.3f}",
                            val=f"{r['overall']:.3f}", best=f"{best:.3f}")
    bar.close()

    model.load_state_dict(best_state)
    CKPT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.arm}_d{a.dim}_r{a.wrec}"
    torch.save(dict(state=model.state_dict(), arm=a.arm, dim=a.dim,
                    drop=drop,
                    d_sig=d_sig, wrec=a.wrec, ymu=ymu, ysd=ysd,
                    amu=amu, asd=asd, best_val=best, best_epoch=best_ep),
               CKPT / f"{tag}.pt")
    print(f"\nbest val retrieval {best:.3f} at epoch {best_ep} "
          f"-> {CKPT / (tag + '.pt')}")

    # honest read: val is what selected the checkpoint, so it is not a result.
    dv = descriptors(va, VA)
    report(f"[{tag}] val (SELECTION SET - not a result)", evaluate(dv, va, va))
    R.log("vjz", arm=a.arm, dim=a.dim, wrec=a.wrec, epochs=a.epochs,
          params=npar, best_val=round(best, 4), best_epoch=best_ep,
          n_train=len(tr), tag=tag)
    print("\n" + json.dumps(dict(tag=tag, best_val=round(best, 4),
                                 best_epoch=best_ep), indent=1))


if __name__ == "__main__":
    main()
