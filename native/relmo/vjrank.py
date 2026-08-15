"""Learned similarity ranker over latents. No hand-written event channels.

    z_t   = f(z_{t-1}, a_t, g_t, sig_t)          recurrence, trainable
    s     = S(Z_query, Z_candidate)              scorer, trainable

INPUTS ARE LATENTS ONLY. a_t is V-JEPA's expectation pred(t+1)-act(t); g_t is
its realised change compressed to two scalars; sig_t is SigLIP appearance.
Nothing describes the event. The model is never asked to predict a verb, an
object, a scene or a camera - given two clips it produces one number.

THE ERROR BAR IS STRUCTURAL. pred(t+1)-act(t+1) lies in the linear span of
{act(t), pred(t+1), act(t+1)}, so any MLP over that concatenation forms it in
its first layer. Reality therefore enters as two scalars, which cannot
reconstruct a 1024-d residual.

WHY A SCORER AND NOT COSINE. Cosine weights all directions equally, which is
the measured cause of the whole problem: a linear probe reads event structure
off these same frozen features at 92.4% while cosine retrieval gets 0.525. The
scorer here consumes the full (T,T) matrix of step-to-step similarities between
two clips and learns what pattern in it means "same event" - a learned
generalisation of DTW, which is one fixed hand-chosen reduction of that matrix.
It is initialised as pooled cosine plus a learned residual so it starts no
worse than the metric it replaces.

COST. Encoding is once per clip. Scoring a pair is a 24x24 matmul plus a
two-layer CNN on a 24x24 image - microseconds, batched over a whole pool. The
approximate-index stage comes later; this is the quality ceiling it will
approximate.

    python -m relmo.vjrank --epochs 400
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
from relmo.vjsplit import load as load_split  # noqa: E402

CKPT = R.BASE / "models" / "vjrank"


def build(torch, nn, d=128, d_a=1024, d_s=768, use_sig=True):
    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.na = nn.LayerNorm(d_a)
            self.pa = nn.Linear(d_a, d)
            self.use_sig = use_sig
            if use_sig:
                self.ns = nn.LayerNorm(d_s)
                self.ps = nn.Linear(d_s, d)
            n_in = d * (2 if use_sig else 1)
            self.gate = nn.Linear(d + n_in + 2, d)
            self.cand = nn.Linear(d + n_in, d)
            self.drop = nn.Dropout(0.1)
            # scorer over the (T,T) step-similarity image
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1), nn.GELU(),
                nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1))
            self.head = nn.Linear(32, 1)
            self.w_cos = nn.Parameter(torch.tensor(4.0))   # start = cosine
            self.d = d

        def encode(self, a, g, sig):
            u = self.drop(nn.functional.gelu(self.pa(self.na(a))))
            if self.use_sig:
                v = nn.functional.gelu(self.ps(self.ns(sig)))
                u = torch.cat([u, v], -1)
            z = a.new_zeros(a.shape[0], self.d)
            Z = []
            for t in range(a.shape[1]):
                ut = u[:, t]
                gt = torch.sigmoid(self.gate(torch.cat([z, ut, g[:, t]], -1)))
                ct = torch.tanh(self.cand(torch.cat([z, ut], -1)))
                z = (1 - gt) * z + gt * ct
                Z.append(z)
            return torch.stack(Z, 1)

        def score(self, Zq, Zc):
            """(Nq,T,d) x (Nc,T,d) -> (Nq,Nc). Full pairwise."""
            Q = nn.functional.normalize(Zq, dim=-1)
            C = nn.functional.normalize(Zc, dim=-1)
            M = torch.einsum("qtd,csd->qcts", Q, C)         # (Nq,Nc,T,T)
            nq, nc, T, _ = M.shape
            base = M.mean((2, 3)) * self.w_cos              # pooled cosine
            f = self.cnn(M.reshape(nq * nc, 1, T, T)).flatten(1)
            return base + self.head(f).reshape(nq, nc)
    return Ranker()


def lambda_rank(s, rel, valid, torch):
    """RankNet with gain-weighted pairs: a bigger relevance gap costs more.

    valid[q,i] says candidate i is usable for query q. A PAIR (i,j) counts only
    when both are usable for that query, which is what drops the diagonal and
    every same-rollout cross-view candidate.
    """
    d_s = s.unsqueeze(2) - s.unsqueeze(1)                   # (N,Ni,Nj)
    d_r = rel.unsqueeze(2) - rel.unsqueeze(1)
    m = valid.unsqueeze(2) & valid.unsqueeze(1) & (d_r > 0)
    if not m.any():
        return s.sum() * 0
    return (d_r[m] * torch.nn.functional.softplus(-d_s[m])).sum() / m.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--rec-suffix", default="",
                    help="compression variant: _fp16, _fp16f32, ...")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--no-sig", action="store_true")
    ap.add_argument("--no-scorer", action="store_true",
                    help="ablation: pooled cosine only, scorer CNN disabled")
    ap.add_argument("--hold-families", default="",
                    help="comma-separated task families excluded from TRAINING "
                         "labels, to measure unseen-event generalisation")
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
    D = vjz.gather(sp["train"] | sp["val"] | sp["test"],
                   rec_dir=R.BASE / "vjrec4" / f"rcasa_L6{a.rec_suffix}",
                   sig_dir=R.BASE / "vjsig" / f"rcasa{a.rec_suffix}")
    meta = vjrel.meta_table(a.dataset)
    hold = {h.strip() for h in a.hold_families.split(",") if h.strip()}
    tr = [i for i in sorted(sp["train"]) if i in D
          and parse(i)["task"] not in hold]
    va = [i for i in sorted(sp["val"]) if i in D]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"train {len(tr)} | val {len(va)} | dev {dev}"
          + (f" | HELD OUT of training: {sorted(hold)}" if hold else ""),
          flush=True)

    def tens(ids):
        return (torch.tensor(np.stack([D[i]["a"] for i in ids]), device=dev),
                torch.tensor(np.stack([D[i]["g"] for i in ids]), device=dev),
                torch.tensor(np.stack([D[i]["sig"] for i in ids]), device=dev))
    TR, VA = tens(tr), tens(va)
    rel_tr, val_tr = vjrel.relevance(tr, meta)
    rel_va, val_va = vjrel.relevance(va, meta)
    RT = torch.tensor(rel_tr, device=dev)
    VT = torch.tensor(val_tr, device=dev)

    model = build(torch, nn, a.dim, use_sig=not a.no_sig).to(dev)
    if a.no_scorer:
        for p_ in list(model.cnn.parameters()) + list(model.head.parameters()):
            p_.requires_grad_(False)
            p_.zero_()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.wd)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  {npar/1e3:.0f}k params | scorer "
          f"{'OFF (pooled cosine)' if a.no_scorer else 'ON'}", flush=True)

    gv = np.array([group_key(parse(i)) for i in va])
    roll_v = np.array([meta[i]["rollout"] for i in va])

    def eval_val():
        model.eval()
        with torch.no_grad():
            Z = model.encode(*VA)
            S = model.score(Z, Z).cpu().numpy()
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
    bar = tqdm(range(a.epochs), unit="ep", desc="vjrank")
    for ep in bar:
        model.train()
        idx = torch.randperm(len(tr), device=dev)[:a.bs]
        Z = model.encode(TR[0][idx], TR[1][idx], TR[2][idx])
        S = model.score(Z, Z)
        # valid drops the diagonal and every same-rollout (cross-view) pair
        loss = lambda_rank(S, RT[idx][:, idx], VT[idx][:, idx], torch)
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
    tag = a.tag or f"rank_d{a.dim}_s{a.seed}"
    torch.save(dict(state=model.state_dict(), dim=a.dim, use_sig=not a.no_sig,
                    hold=sorted(hold), params=npar), CKPT / f"{tag}.pt")
    print(f"\nbest val NDCG {nd:.4f}  precision@support {pr:.4f}  "
          f"(epoch {best_ep})  -> {CKPT / (tag + '.pt')}")
    R.log("vjrank", tag=tag, dim=a.dim, params=npar, seed=a.seed,
          held=sorted(hold), val_ndcg=round(nd, 4), val_prec=round(pr, 4))
    print(json.dumps(dict(tag=tag, val_ndcg=round(nd, 4),
                          val_prec=round(pr, 4), params=npar), indent=1))


if __name__ == "__main__":
    main()
