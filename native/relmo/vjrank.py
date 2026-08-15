"""Learned similarity ranker over latents. No hand-written event channels.

    z_t   = f(z_{t-1}, a_t, g_t, sig_t)          recurrence, trainable
    s     = S(Z_query, Z_candidate)              scorer, trainable

t IS ABSOLUTE STREAM TIME (relmo/vjrec6), one step every 0.25 s, running the
length of the RECORDING. It is not a position inside an encoded clip. Under v4
it was, and the consequence was that `z` was re-initialised to zero at every
clip boundary - so the recurrence above had no history to carry and was, in the
owner's words, "just an variable length encoder of smaller vs bigger window".

Two things follow, and both are load-bearing here:

  * z is initialised ONCE PER RECORDING. Windows tile the stream, their
    descriptor blocks concatenate into one trace, and the recurrence crosses
    those boundaries without noticing them.
  * T therefore VARIES with duration. Batches are right-padded and every
    operation is masked: z must not advance past the end of a recording, and a
    padded step must not contribute to a similarity. On this corpus the median
    recording spans 4 windows, so the boundary crossing is exercised by most
    of the training set rather than by a tail of long clips.

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
# The step-similarity tensor is (Nq,Nc,T,S) - QUADRATIC in trace length, which
# under stream time varies from 16 to 216 steps. A 48x48 batch of the longest
# recordings would allocate 430 MB for that tensor alone (and ~1.7 GB more for
# the CNN's first feature map), so both scoring and training tile against a
# fixed element budget instead of a fixed item count.
PAIR_BUDGET = 24_000_000


def build(torch, nn, d=128, d_a=1024, d_s=768, use_sig=True, use_scorer=True):
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
            self.use_scorer = use_scorer

        def encode(self, a, g, sig, m):
            """(N,T,*) padded + (N,T) bool valid -> (N,T,d).

            z is zeroed once, at t=0 of the RECORDING, and then carried. Past
            the end of a recording it is frozen rather than updated, so padding
            cannot leak into the state.
            """
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
                z = torch.where(m[:, t:t + 1], (1 - gt) * z + gt * ct, z)
                Z.append(z)
            return torch.stack(Z, 1)

        def score(self, Zq, mq, Zc, mc):
            """(Nq,T,d),(Nq,T) x (Nc,S,d),(Nc,S) -> (Nq,Nc). Full pairwise."""
            Q = nn.functional.normalize(Zq, dim=-1) * mq.unsqueeze(-1)
            C = nn.functional.normalize(Zc, dim=-1) * mc.unsqueeze(-1)
            M = torch.einsum("qtd,csd->qcts", Q, C)         # (Nq,Nc,T,S)
            nq, nc, T, S = M.shape
            # a pair (t,s) counts only if both steps are real
            W = (mq[:, None, :, None] * mc[None, :, None, :]).to(M.dtype)
            MW = M * W
            base = MW.sum((2, 3)) / W.sum((2, 3)).clamp(min=1.0)
            if not self.use_scorer:
                # the CNN is the dominant cost at T=216; when it is ablated
                # off, do not pay for it
                return base * self.w_cos
            f = self.cnn(MW.reshape(nq * nc, 1, T, S)).flatten(1)
            return base * self.w_cos + self.head(f).reshape(nq, nc)
    return Ranker()


def tile_score(model, Zq, mq, Zc, mc, torch, budget=PAIR_BUDGET):
    """model.score over arbitrarily many items, tiled to a memory budget."""
    per = max(1, int((budget / max(1, Zq.shape[1] * Zc.shape[1])) ** 0.5))
    rows = []
    for i in range(0, len(Zq), per):
        cols = [model.score(Zq[i:i + per], mq[i:i + per],
                            Zc[j:j + per], mc[j:j + per])
                for j in range(0, len(Zc), per)]
        rows.append(torch.cat(cols, 1))
    return torch.cat(rows, 0)


def load_ckpt(tag):
    """-> (model, ck). Lives here so evaluators can share it without importing
    each other."""
    import torch
    import torch.nn as nn
    ck = torch.load(CKPT / f"{tag}.pt", weights_only=False,
                    map_location="cpu")
    m = build(torch, nn, ck["dim"], use_sig=ck["use_sig"],
              use_scorer=ck.get("use_scorer", True))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def encode_all(model, recs, ids=None, chunk=64):
    """{id: (T_i, d)} trained z sequences, trimmed back to their real lengths."""
    import torch
    ids = sorted(ids if ids is not None else recs,
                 key=lambda i: (len(recs[i]["a"]), i))
    out = {}
    for c in range(0, len(ids), chunk):
        b = ids[c:c + chunk]
        a, m = pad([recs[i]["a"] for i in b], "cpu", torch)
        g, _ = pad([recs[i]["g"] for i in b], "cpu", torch)
        s, _ = pad([recs[i]["sig"] for i in b], "cpu", torch)
        with torch.no_grad():
            Z = model.encode(a, g, s, m).numpy()
        for k, i in enumerate(b):
            out[i] = Z[k, :len(recs[i]["a"])]
    return out


def pad(seqs, dev, torch):
    """[(T_i,D)] -> padded (N,Tmax,D) tensor and (N,Tmax) bool mask."""
    tmax = max(len(s) for s in seqs)
    x = np.zeros((len(seqs), tmax, seqs[0].shape[-1]), np.float32)
    m = np.zeros((len(seqs), tmax), bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        m[i, :len(s)] = True
    return (torch.tensor(x, device=dev),
            torch.tensor(m, device=dev))


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
    ap.add_argument("--rec-root", default="vjrec6",
                    help="vjrec6 = stream time (default), vjrec4 = clip time")
    ap.add_argument("--phase", default="rcasa_train",
                    help="vjphase model to subtract; '' disables it")
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
    rec_dir, sig_dir = vjz.dirs(a.rec_root, a.dataset, suffix=a.rec_suffix)
    ph = None
    if a.phase:
        from relmo.vjphase import load as load_phase
        ph = load_phase(a.phase)
    D = vjz.gather(sp["train"] | sp["val"] | sp["test"],
                   rec_dir=rec_dir, sig_dir=sig_dir, phase=ph)
    D = {i: v for i, v in D.items() if v["sig"] is not None
         and len(v["sig"]) == len(v["a"])}
    meta = vjrel.meta_table(a.dataset)
    hold = {h.strip() for h in a.hold_families.split(",") if h.strip()}
    tr = [i for i in sorted(sp["train"]) if i in D
          and parse(i)["task"] not in hold]
    va = [i for i in sorted(sp["val"]) if i in D]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    lens = np.array([len(D[i]["a"]) for i in tr])
    print(f"train {len(tr)} | val {len(va)} | dev {dev} | records {rec_dir.name}"
          f" | steps/recording min {lens.min()} med {int(np.median(lens))} "
          f"max {lens.max()}"
          + (f" | HELD OUT of training: {sorted(hold)}" if hold else ""),
          flush=True)

    def tens(ids):
        a_, m_ = pad([D[i]["a"] for i in ids], dev, torch)
        g_, _ = pad([D[i]["g"] for i in ids], dev, torch)
        s_, _ = pad([D[i]["sig"] for i in ids], dev, torch)
        return a_, g_, s_, m_
    TR, VA = tens(tr), tens(va)
    rel_tr, val_tr = vjrel.relevance(tr, meta)
    rel_va, val_va = vjrel.relevance(va, meta)
    RT = torch.tensor(rel_tr, device=dev)
    VT = torch.tensor(val_tr, device=dev)

    model = build(torch, nn, a.dim, use_sig=not a.no_sig,
                  use_scorer=not a.no_scorer).to(dev)
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
            S = tile_score(model, Z, VA[3], Z, VA[3], torch).cpu().numpy()
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

    best, best_state, best_ep, n_shrunk = -1.0, None, -1, 0
    bar = tqdm(range(a.epochs), unit="ep", desc="vjrank")
    for ep in bar:
        model.train()
        idx = torch.randperm(len(tr), device=dev)[:a.bs]
        # a batch of long recordings must shrink: the pair tensor is bs^2*T^2
        tmax = int(TR[3][idx].sum(1).max())
        n_ok = max(4, min(a.bs, int(PAIR_BUDGET ** 0.5) // max(1, tmax)))
        if n_ok < len(idx):
            idx = idx[:n_ok]
            n_shrunk += 1
        # TR is padded to the GLOBAL max (216 steps); running the recurrence
        # that far for a batch whose longest recording is 32 is pure waste
        mb = TR[3][idx][:, :tmax]
        Z = model.encode(TR[0][idx][:, :tmax], TR[1][idx][:, :tmax],
                         TR[2][idx][:, :tmax], mb)
        S = model.score(Z, mb, Z, mb)
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
    if n_shrunk:
        print(f"  {n_shrunk}/{a.epochs} batches shrunk below --bs {a.bs} to "
              f"stay inside the pair budget (long recordings)")

    model.load_state_dict(best_state)
    nd, pr = eval_val()
    CKPT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"rank_d{a.dim}_s{a.seed}"
    torch.save(dict(state=model.state_dict(), dim=a.dim, use_sig=not a.no_sig,
                    use_scorer=not a.no_scorer, rec_root=a.rec_root,
                    phase=a.phase,
                    hold=sorted(hold), params=npar), CKPT / f"{tag}.pt")
    print(f"\nbest val NDCG {nd:.4f}  precision@support {pr:.4f}  "
          f"(epoch {best_ep})  -> {CKPT / (tag + '.pt')}")
    R.log("vjrank", tag=tag, dim=a.dim, params=npar, seed=a.seed,
          held=sorted(hold), val_ndcg=round(nd, 4), val_prec=round(pr, 4))
    print(json.dumps(dict(tag=tag, val_ndcg=round(nd, 4),
                          val_prec=round(pr, 4), params=npar), indent=1))


if __name__ == "__main__":
    main()
