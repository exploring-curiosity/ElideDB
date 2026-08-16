"""Learned spatial pooling over the 256 tokens, trained jointly with f.

    w_{t,k} = softmax_k  q( tok_{t,k}, gate_{t,k} )      pooling, trainable
    u_t     = sum_k w_{t,k} tok_{t,k}                    256 tokens -> 1 vector
    z_t     = f(z_{t-1}, u_t, g_t, sig_t)                recurrence, trainable
    s       = pooled masked cosine between z sequences   matcher

v6 pooled with a FIXED gate - layer-6 observed change magnitude, median
subtracted and clipped - which was never trained and never compared against an
alternative. Here the pooling is a head that sees each token's own feature
alongside that gate value, so the gate becomes a hint rather than the whole
answer.

THE PRIMARY CHANNEL IS `b`. The barred residual pred(t+1)-act(t+1) = a - b is
formable by any linear layer receiving both as vectors, so at most one may be
one - and measurement says it should be `b`: obs_change outscores pred_change
frozen (0.442 vs 0.420 prec) and as the trained primary (val 0.752 vs 0.673,
test 0.763 vs 0.676). `a` enters only through g's two scalars, which cannot
reconstruct a 1024-d residual.

WHAT THIS IS ACTUALLY FOR. The measured gap is not capacity - train precision is
already 0.968 against val 0.752. It is generalisation. So the pooling head comes
with the regularisers that target that gap directly:

  temporal crop   train on a random contiguous sub-span of each trace. A query
                  IS a sub-span, so this matches the deployed condition rather
                  than merely perturbing the input.
  token dropout   drop spatial tokens before pooling, so the head cannot rely
                  on a few positions.
  channel dropout drop `sig` or `u` entirely for a step, so neither channel can
                  become load-bearing on its own.

    python -m relmo.vjrank2 --epochs 1500 --tag p1_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrel, vjz  # noqa: E402
from relmo.vjeval import group_key, parse  # noqa: E402
from relmo.vjmatch import ARC_DS, arc_resample  # noqa: E402
from relmo.vjrank import CKPT, PAIR_BUDGET, lambda_rank, pad  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402

REC7 = R.BASE / "vjrec7"


def load_corpus(datasets=("rcasa", "rcasa_eval"), layer=6, arc=ARC_DS,
                suffix=""):
    """{id: dict(tok (T,256,d), gate (T,256), g (T,2), sig (T,768))}."""
    out = {}
    for ds in datasets:
        t7 = REC7 / f"{ds}_L{layer}{suffix}"
        rec6, sig6 = vjz.dirs("vjrec6", ds, suffix=suffix)
        if not t7.exists():
            continue
        for p in sorted(t7.glob("*.npz")):
            if p.name.startswith("."):
                continue
            f6 = rec6 / p.name
            s6 = sig6 / p.name
            if not (f6.exists() and s6.exists()):
                continue
            z7 = np.load(p)
            z6 = np.load(f6)
            sig = np.load(s6)["sig"].astype(np.float32)
            tok = z7["b_tok"].astype(np.float32)
            gate = z7["gate"].astype(np.float32)
            # the FIXED gate-pooled b at full 1024-d, from the v6 records. The
            # token PCA keeps only 69.5% of the per-token variance at 96-d
            # (against 99.1% for pooled descriptors at 256-d - per-token
            # features are far higher rank), so a learned pooling over 96-d
            # tokens would be capped below the fixed pooling it is meant to
            # improve on. Feeding BOTH makes the learned head additive: it can
            # only add to what the fixed gate already found, never subtract.
            fix, g = vjz.channels(z6, primary="b")
            T, K, D = tok.shape
            if len(sig) != T or len(g) != T or len(fix) != T:
                continue
            # where_map is the per-step SPATIAL motion signature (16x16 gate
            # energy). It has always been loaded and then collapsed to a
            # scalar for arc-length weighting; the map itself has never been
            # used as a channel. It answers WHERE change happened, which is
            # orthogonal to fix (WHAT changed) and sig (what it LOOKS like).
            wmap = z6["where_map"].reshape(T, -1).astype(np.float32)
            if arc > 0:
                w = wmap.sum(1)
                flat, rest = arc_resample(tok.reshape(T, K * D), w, arc,
                                          aux=[gate, g, sig, fix, wmap],
                                          max_len=256)
                tok = flat.reshape(len(flat), K, D)
                gate, g, sig, fix, wmap = rest
            out[p.stem] = dict(tok=tok, gate=gate, g=g, sig=sig, fix=fix,
                               wmap=wmap)
    return out


def build(torch, nn, d=128, d_tok=96, d_s=768, d_fix=1024, p_drop=0.1):
    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            # pooling head: each token scored from its own feature + its gate
            self.nt = nn.LayerNorm(d_tok)
            self.q = nn.Sequential(nn.Linear(d_tok + 1, 64), nn.GELU(),
                                   nn.Linear(64, 1))
            self.pu = nn.Linear(d_tok, d)
            self.nf = nn.LayerNorm(d_fix)
            self.pf = nn.Linear(d_fix, d)
            self.ns = nn.LayerNorm(d_s)
            self.ps = nn.Linear(d_s, d)
            self.gate = nn.Linear(d + 3 * d + 2, d)
            self.cand = nn.Linear(d + 3 * d, d)
            self.drop = nn.Dropout(p_drop)
            self.w_cos = nn.Parameter(torch.tensor(4.0))
            self.d = d

        def pool(self, tok, gate, tok_drop=0.0):
            """(N,T,K,d_tok) + (N,T,K) -> (N,T,d_tok)."""
            x = self.nt(tok)
            logit = self.q(torch.cat([x, gate.unsqueeze(-1)], -1)).squeeze(-1)
            if tok_drop > 0 and self.training:
                keep = (torch.rand_like(logit) > tok_drop)
                logit = logit.masked_fill(~keep, -1e9)
            w = torch.softmax(logit, -1)
            return (w.unsqueeze(-1) * tok).sum(2)

        def encode(self, tok, gate, g, sig, fix, m, tok_drop=0.0,
                   chan_drop=0.0):
            u = self.pool(tok, gate, tok_drop)
            uu = self.drop(nn.functional.gelu(self.pu(u)))
            ff = self.drop(nn.functional.gelu(self.pf(self.nf(fix))))
            vv = nn.functional.gelu(self.ps(self.ns(sig)))
            if chan_drop > 0 and self.training:
                b = uu.shape[0]
                r = lambda: (torch.rand(b, 1, 1, device=uu.device)  # noqa: E731
                             > chan_drop).float()
                uu, ff, vv = uu * r(), ff * r(), vv * r()
            h = torch.cat([uu, ff, vv], -1)
            z = uu.new_zeros(uu.shape[0], self.d)
            Z = []
            for t in range(uu.shape[1]):
                ht = h[:, t]
                gt = torch.sigmoid(self.gate(torch.cat([z, ht, g[:, t]], -1)))
                ct = torch.tanh(self.cand(torch.cat([z, ht], -1)))
                z = torch.where(m[:, t:t + 1], (1 - gt) * z + gt * ct, z)
                Z.append(z)
            return torch.stack(Z, 1)

        def score(self, Zq, mq, Zc, mc):
            Q = nn.functional.normalize(Zq, dim=-1) * mq.unsqueeze(-1)
            C = nn.functional.normalize(Zc, dim=-1) * mc.unsqueeze(-1)
            M = torch.einsum("qtd,csd->qcts", Q, C)
            W = (mq[:, None, :, None] * mc[None, :, None, :]).to(M.dtype)
            return ((M * W).sum((2, 3))
                    / W.sum((2, 3)).clamp(min=1.0)) * self.w_cos
    return Ranker()


def load_ckpt(tag):
    import torch
    import torch.nn as nn
    ck = torch.load(CKPT / f"{tag}.pt", weights_only=False, map_location="cpu")
    m = build(torch, nn, ck["dim"], ck.get("d_tok", 96),
              d_fix=ck.get("d_fix", 1024))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def encode_all(model, D, ids=None, chunk=24):
    import torch
    ids = sorted(ids if ids is not None else D,
                 key=lambda i: (len(D[i]["tok"]), i))
    out = {}
    for c in range(0, len(ids), chunk):
        b = ids[c:c + chunk]
        T = max(len(D[i]["tok"]) for i in b)
        K, Dt = D[b[0]]["tok"].shape[1:]
        tok = np.zeros((len(b), T, K, Dt), np.float32)
        gate = np.zeros((len(b), T, K), np.float32)
        for k, i in enumerate(b):
            n = len(D[i]["tok"])
            tok[k, :n] = D[i]["tok"]
            gate[k, :n] = D[i]["gate"]
        g, m = pad([D[i]["g"] for i in b], "cpu", torch)
        s, _ = pad([D[i]["sig"] for i in b], "cpu", torch)
        f, _ = pad([D[i]["fix"] for i in b], "cpu", torch)
        with torch.no_grad():
            Z = model.encode(torch.tensor(tok), torch.tensor(gate), g, s, f,
                             m).numpy()
        for k, i in enumerate(b):
            out[i] = Z[k, :len(D[i]["tok"])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--tok-drop", type=float, default=0.2)
    ap.add_argument("--chan-drop", type=float, default=0.15)
    ap.add_argument("--crop", type=float, default=0.7,
                    help="min fraction of a trace kept by the temporal crop")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="p1_s0")
    ap.add_argument("--rec-suffix", default="")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from tqdm import tqdm

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    sp = load_split()
    D = load_corpus(suffix=a.rec_suffix)
    meta = vjrel.meta_table("rcasa")
    tr = [i for i in sorted(sp["train"]) if i in D]
    va = [i for i in sorted(sp["val"]) if i in D]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    K, Dt = D[tr[0]]["tok"].shape[1:]
    print(f"train {len(tr)} | val {len(va)} | {K} tokens x {Dt}d | dev {dev}",
          flush=True)

    rel_tr, val_tr = vjrel.relevance(tr, meta)
    rel_va, val_va = vjrel.relevance(va, meta)
    RT = torch.tensor(rel_tr, device=dev)
    VT = torch.tensor(val_tr, device=dev)

    def batch(ids, crop=False):
        T = max(len(D[i]["tok"]) for i in ids)
        n = len(ids)
        tok = np.zeros((n, T, K, Dt), np.float32)
        gate = np.zeros((n, T, K), np.float32)
        g = np.zeros((n, T, 2), np.float32)
        sig = np.zeros((n, T, D[ids[0]]["sig"].shape[1]), np.float32)
        fix = np.zeros((n, T, D[ids[0]]["fix"].shape[1]), np.float32)
        m = np.zeros((n, T), bool)
        for k, i in enumerate(ids):
            L = len(D[i]["tok"])
            lo, hi = 0, L
            if crop and L > 8:
                keep = max(8, int(L * rng.uniform(a.crop, 1.0)))
                lo = int(rng.integers(0, L - keep + 1))
                hi = lo + keep
            s = hi - lo
            tok[k, :s] = D[i]["tok"][lo:hi]
            gate[k, :s] = D[i]["gate"][lo:hi]
            g[k, :s] = D[i]["g"][lo:hi]
            sig[k, :s] = D[i]["sig"][lo:hi]
            fix[k, :s] = D[i]["fix"][lo:hi]
            m[k, :s] = True
        t = lambda x: torch.tensor(x, device=dev)  # noqa: E731
        return t(tok), t(gate), t(g), t(sig), t(fix), t(m)

    VA = batch(va)
    model = build(torch, nn, a.dim, Dt, d_fix=D[tr[0]]["fix"].shape[1],
                  p_drop=a.drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  {npar/1e3:.0f}k params | crop>={a.crop} tok_drop={a.tok_drop} "
          f"chan_drop={a.chan_drop} wd={a.wd}", flush=True)

    gv = np.array([group_key(parse(i)) for i in va])
    roll_v = np.array([meta[i]["rollout"] for i in va])

    def eval_val():
        model.eval()
        with torch.no_grad():
            Z = model.encode(*VA)
            S = model.score(Z, VA[5], Z, VA[5]).cpu().numpy()
        np.fill_diagonal(S, -1e9)
        nd = vjrel.ndcg(S, rel_va, val_va)
        hit = sup = 0
        for q in range(len(va)):
            keep = roll_v != roll_v[q]
            sv = gv[keep] == gv[q]
            k = int(sv.sum())
            if k < 5:
                continue
            hit += int(sv[np.argsort(-S[q][keep])[:k]].sum())
            sup += k
        return nd, (hit / sup if sup else float("nan"))

    best, best_state, best_ep = -1.0, None, -1
    bar = tqdm(range(a.epochs), unit="ep", desc="vjrank2")
    for ep in bar:
        model.train()
        idx = rng.choice(len(tr), min(a.bs, len(tr)), replace=False)
        ids = [tr[i] for i in idx]
        tok, gate, g, sig, fx, m = batch(ids, crop=True)
        tmax = int(m.sum(1).max())
        n_ok = max(4, min(len(ids), int(PAIR_BUDGET ** 0.5) // max(1, tmax)))
        if n_ok < len(ids):
            tok, gate, g, sig, fx, m = (x[:n_ok] for x in
                                        (tok, gate, g, sig, fx, m))
            idx = idx[:n_ok]
        Z = model.encode(tok, gate, g, sig, fx, m, a.tok_drop, a.chan_drop)
        S = model.score(Z, m, Z, m)
        I = torch.tensor(idx, device=dev)
        loss = lambda_rank(S, RT[I][:, I], VT[I][:, I], torch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (ep + 1) % a.every == 0 or ep == a.epochs - 1:
            nd, pr = eval_val()
            if nd > best:
                best, best_ep = nd, ep
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            bar.set_postfix(loss=f"{float(loss):.3f}", ndcg=f"{nd:.3f}",
                            prec=f"{pr:.3f}", best=f"{best:.3f}")
    bar.close()
    model.load_state_dict(best_state)
    nd, pr = eval_val()
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), dim=a.dim, d_tok=Dt,
                    d_fix=D[tr[0]]["fix"].shape[1],
                    params=npar, primary="b", pooled="learned"),
               CKPT / f"{a.tag}.pt")
    print(f"\nbest val NDCG {nd:.4f}  prec@support {pr:.4f}  (epoch {best_ep})")
    R.log("vjrank2", tag=a.tag, params=npar, seed=a.seed,
          val_ndcg=round(nd, 4), val_prec=round(pr, 4))
    print(json.dumps(dict(tag=a.tag, val_ndcg=round(nd, 4),
                          val_prec=round(pr, 4), params=npar), indent=1))


if __name__ == "__main__":
    main()
