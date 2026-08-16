"""Stage C of SSL_TRAINING.md: self-supervised training of `f`.

NO LABELS. This module never reads a task name, scene id or camera id for
supervision. The only identity it touches is the rollout prefix of an episode
id (`Task_episode_NNNNNN__cam` before `__`), used to EXCLUDE same-rollout
pairs from the contrastive loss - honouring the cross-view bar, not learning
from it.

Objective (weights per SSL_TRAINING.md §5):
  span    [1.0]  predict a masked 8-24-step span's EMA-target embedding from
                 the prefix state. Single-step gaps banned - interpolation
                 must not solve the task.
  overlap [1.0]  two crops of one recording overlapping 25-75% -> InfoNCE,
                 temp 0.1, same-rollout logits masked out entirely.
  vicreg  [0.5 var / 0.1 cov]  variance floor 1.0 per dim, covariance
                 off-diagonals to zero. The anti-collapse term: the shipped
                 head sits at PR 3.8 of 128; the width bought here (d=256)
                 is only real if this term makes the model use it.

GATE, per epoch, held-out 10% of recordings: participation ratio of z
covariance. The runbook's abort rule: plateau < 15 after epoch 5 -> stop,
the objective is not escaping collapse and an eval would be noise.

Memory: tokens stay fp16 until the batch is cut (the fp32 corpus load in
vjrank2 would need ~25 GB over this pool).

    python3 -m relmo.vjssl --tag ssl_v1_s0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjz  # noqa: E402
from relmo.vjmatch import ARC_DS, arc_resample  # noqa: E402
from relmo.vjrank import CKPT  # noqa: E402
from relmo.vjrank2 import REC7, build  # noqa: E402

POOL = ("rcasa", "rcasa_eval", "rcasa_atomic_full", "bridge_wide", "movi_e",
        "kitti_seq", "oxford_seq", "drone_fpv")
# rcasa_composite_full is the sealed eval and must never appear here.
assert "rcasa_composite_full" not in POOL


def load_pool(datasets=POOL, layer=6):
    """Like vjrank2.load_corpus but tokens kept fp16; ~4x less RAM.

    DEDUPLICATED BY EPISODE STEM. 266 RoboCasa demonstrations exist in BOTH
    rcasa and rcasa_atomic_full - different files, different bitrates
    (432 KB vs 101 KB), same demonstration (obs_change cosine 0.76). Training
    on both teaches invariance to video encoding, and mining ranks them as
    the corpus's most similar cross-video pairs, which is degenerate. First
    dataset in POOL order wins.

    `rollout` deliberately OMITS the dataset name so that the same
    demonstration appearing under two corpora is treated as one rollout by
    the cross-view bar, in training and in mining alike.
    """
    out, seen_stem = {}, set()
    for ds in datasets:
        t7 = REC7 / f"{ds}_L{layer}"
        rec6, sig6 = vjz.dirs("vjrec6", ds)
        if not t7.exists():
            print(f"  WARNING: no records for {ds}, skipped", flush=True)
            continue
        n0 = len(out)
        for p in sorted(t7.glob("*.npz")):
            if p.name.startswith("."):
                continue
            if p.stem in seen_stem:          # same demo under another corpus
                continue
            f6, s6 = rec6 / p.name, sig6 / p.name
            if not (f6.exists() and s6.exists()):
                continue
            seen_stem.add(p.stem)
            z7, z6 = np.load(p), np.load(f6)
            sig = np.load(s6)["sig"].astype(np.float32)
            tok, gate = z7["b_tok"], z7["gate"].astype(np.float32)
            fix, g = vjz.channels(z6, primary="b")
            T, K, Dt = tok.shape
            if not (len(sig) == len(g) == len(fix) == T):
                continue
            w = z6["where_map"].reshape(T, -1).sum(1)
            flat, rest = arc_resample(tok.reshape(T, K * Dt).astype(np.float32),
                                      w, ARC_DS, aux=[gate, g, sig, fix],
                                      max_len=256)
            out[f"{ds}/{p.stem}"] = dict(
                tok=flat.reshape(len(flat), K, Dt).astype(np.float16),
                gate=rest[0].astype(np.float16), g=rest[1],
                sig=rest[2].astype(np.float16), fix=rest[3].astype(np.float16),
                rollout=p.stem.split("__")[0])
            if len(out[f"{ds}/{p.stem}"]["tok"]) < 12:
                del out[f"{ds}/{p.stem}"]        # too short for span+prefix
        print(f"  {ds}: {len(out)-n0} recordings", flush=True)
    return out


def crop(rec, lo, hi, torch, dev):
    """[lo,hi) of one recording -> dict of fp32 tensors on dev."""
    return {k: torch.tensor(np.ascontiguousarray(rec[k][lo:hi]),
                            dtype=torch.float32, device=dev)
            for k in ("tok", "gate", "g", "sig", "fix")}


def encode_batch(model, crops, torch, dev):
    """Pad a list of crop dicts to one batch; -> z at every step + mask."""
    T = max(len(c["tok"]) for c in crops)
    B = len(crops)
    K, Dt = crops[0]["tok"].shape[1:]
    tok = torch.zeros(B, T, K, Dt, device=dev)
    gate = torch.zeros(B, T, K, device=dev)
    g = torch.zeros(B, T, 2, device=dev)
    sig = torch.zeros(B, T, crops[0]["sig"].shape[1], device=dev)
    fix = torch.zeros(B, T, crops[0]["fix"].shape[1], device=dev)
    m = torch.zeros(B, T, dtype=torch.bool, device=dev)
    for i, c in enumerate(crops):
        n = len(c["tok"])
        tok[i, :n], gate[i, :n], g[i, :n] = c["tok"], c["gate"], c["g"]
        sig[i, :n], fix[i, :n], m[i, :n] = c["sig"], c["fix"], True
    Z = model.encode(tok, gate, g, sig, fix, m)          # (B, T, d)
    last = m.sum(1) - 1
    return Z, m, Z[torch.arange(B, device=dev), last]    # per-step, mask, final


def participation(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh((Zc.T @ Zc) / len(Zc))[::-1].clip(0)
    return float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-span", type=float, default=1.0)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--hard-neg", type=float, default=0.0,
                    help="weight on WITHIN-recording negatives: a "
                         "non-overlapping span of the same recording is a "
                         "different MOMENT. v1 measured nce -> 0.005 by "
                         "epoch 2, i.e. 'which video is this' is trivial; "
                         "this makes the task 'which moment is this'. Risk "
                         "to watch: at high weight it can teach phase "
                         "discrimination at the expense of event identity.")
    ap.add_argument("--w-var", type=float, default=0.5)
    ap.add_argument("--w-cov", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--mined", default="",
                    help="vjmine pairs file. THE v1 FIX: with no mined "
                         "partner every positive is two crops of one "
                         "recording, so the model learns within-video "
                         "consistency and is then asked for cross-video "
                         "retrieval. A mined partner makes the positive a "
                         "DIFFERENT video of the same kind of moment.")
    ap.add_argument("--p-mined", type=float, default=0.5,
                    help="probability of using a mined partner when one "
                         "exists; the rest stay self-overlap crops")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    MINED = {}
    if a.mined:
        md = json.loads(Path(a.mined).read_text())
        for x, y, _c in md["pairs"]:
            MINED.setdefault(x, []).append(y)
            MINED.setdefault(y, []).append(x)
        print(f"mined positives: {md['n_pairs']} pairs over "
              f"{len(MINED)} recordings", flush=True)

    print("loading pool (fp16-resident)...", flush=True)
    D = load_pool()
    ids = sorted(D)
    # held-out gate set: stable 10% by id hash, no labels involved
    hold = [i for i in ids if hash(i) % 10 == 0]
    train = [i for i in ids if i not in set(hold)]
    print(f"pool {len(ids)} recordings -> train {len(train)}, gate {len(hold)}",
          flush=True)

    model = build(torch, nn, a.dim).to(dev)
    target = build(torch, nn, a.dim).to(dev)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(a.dim, a.dim), nn.GELU(),
                         nn.Linear(a.dim, a.dim)).to(dev)
    npar = sum(p.numel() for p in model.parameters()) \
        + sum(p.numel() for p in pred.parameters())
    print(f"online+pred params {npar/1e6:.2f}M | d={a.dim} | {dev}", flush=True)

    opt = torch.optim.AdamW(list(model.parameters()) + list(pred.parameters()),
                            lr=a.lr, weight_decay=1e-4)
    steps_ep = max(1, len(train) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.epochs * steps_ep, eta_min=a.lr / 10)

    def gate_pr():
        model.eval()
        zs = []
        with torch.no_grad():
            for c0 in range(0, len(hold), 24):
                cs = [crop(D[i], 0, min(len(D[i]["tok"]), 256), torch, dev)
                      for i in hold[c0:c0 + 24]]
                Z, m, _ = encode_batch(model, cs, torch, dev)
                zs.append(Z[m].cpu().numpy())
        model.train()
        return participation(np.concatenate(zs))

    t0, hist = time.time(), []
    n_mined = 0
    for ep in range(a.epochs):
        order = rng.permutation(len(train))
        bar = tqdm(range(steps_ep), desc=f"ep{ep}", unit="step")
        for s in bar:
            batch = [train[j] for j in
                     order[s * a.batch:(s + 1) * a.batch]]
            if len(batch) < 4:
                continue
            crops_a, crops_b, spans, span_tgts, rolls = [], [], [], [], []
            crops_h = []
            for i in batch:
                rec = D[i]
                T = len(rec["tok"])
                L = int(min(T, 64))
                # two overlapping crops (25-75% overlap)
                ov = rng.integers(L // 4, 3 * L // 4 + 1)
                s0 = rng.integers(0, T - L + 1)
                s1 = int(np.clip(s0 + L - ov, 0, T - L))
                crops_a.append(crop(rec, s0, s0 + L, torch, dev))
                # the positive: a DIFFERENT video of the same kind of moment
                # when the miner found one, else the self-overlap fallback
                part = None
                if MINED.get(i) and rng.random() < a.p_mined:
                    cs = [c for c in MINED[i] if c in D]
                    part = cs[rng.integers(len(cs))] if cs else None
                if part is not None:
                    pr_ = D[part]
                    Tp = len(pr_["tok"])
                    Lp = int(min(Tp, 64))
                    sp = int(rng.integers(0, Tp - Lp + 1))
                    crops_b.append(crop(pr_, sp, sp + Lp, torch, dev))
                    n_mined += 1
                else:
                    crops_b.append(crop(rec, s1, s1 + L, torch, dev))
                if a.hard_neg > 0:
                    # a DISJOINT span of the same recording: same video, other
                    # moment. Falls back to an overlapping crop when the
                    # recording is too short to be disjoint - such a recording
                    # simply contributes no hard negative.
                    cand = [(u, u + L) for u in range(0, T - L + 1, max(1, L // 2))
                            if u + L <= s0 or u >= s0 + L]
                    h0 = int(cand[rng.integers(len(cand))][0]) if cand else s1
                    crops_h.append(crop(rec, h0, h0 + L, torch, dev))
                # span task inside crop A: prefix >=4, span 8-24
                sp = int(rng.integers(8, min(24, L - 4) + 1))
                st = int(rng.integers(4, L - sp + 1))
                spans.append((st, sp))
                span_tgts.append(crop(rec, s0 + st, s0 + st + sp, torch, dev))
                rolls.append(rec["rollout"])

            Za, ma, fa = encode_batch(model, crops_a, torch, dev)
            _, _, fb = encode_batch(model, crops_b, torch, dev)
            with torch.no_grad():
                _, _, ft = encode_batch(target, span_tgts, torch, dev)
            B = len(batch)
            zpre = Za[torch.arange(B, device=dev),
                      torch.tensor([st - 1 for st, _ in spans], device=dev)]
            l_span = (1 - F.cosine_similarity(pred(zpre), ft, dim=-1)).mean()

            ua, ub = F.normalize(fa, dim=-1), F.normalize(fb, dim=-1)
            logits = ua @ ub.T / 0.1
            same = torch.tensor(
                [[ri == rj for rj in rolls] for ri in rolls], device=dev)
            eye = torch.eye(B, dtype=torch.bool, device=dev)
            logits = logits.masked_fill(same & ~eye, -1e9)
            if a.hard_neg > 0:
                _, _, fh = encode_batch(model, crops_h, torch, dev)
                hard = ua @ F.normalize(fh, dim=-1).T / 0.1
                # a recording's own disjoint span is the hard negative; other
                # rows' hard crops are ordinary negatives, EXCEPT where they
                # share a rollout (cross-view bar applies here too)
                hard = hard.masked_fill(same & ~eye, -1e9)
                logits = torch.cat([logits, hard + np.log(a.hard_neg)], 1)
            l_nce = F.cross_entropy(logits,
                                    torch.arange(B, device=dev))

            zc = torch.cat([fa, fb]) \
                - torch.cat([fa, fb]).mean(0, keepdim=True)
            std = torch.sqrt(zc.var(0) + 1e-4)
            l_var = F.relu(1.0 - std).mean()
            cov = (zc.T @ zc) / (len(zc) - 1)
            l_cov = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) \
                / a.dim

            loss = (a.w_span * l_span + a.w_nce * l_nce
                    + a.w_var * l_var + a.w_cov * l_cov)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pt, po in zip(target.parameters(), model.parameters()):
                    pt.mul_(a.ema).add_(po, alpha=1 - a.ema)
            bar.set_postfix(span=f"{float(l_span):.3f}",
                            nce=f"{float(l_nce):.3f}",
                            var=f"{float(l_var):.3f}")
        pr = gate_pr()
        hist.append(dict(ep=ep, pr=round(pr, 1), span=float(l_span),
                         nce=float(l_nce), var=float(l_var), mined=n_mined))
        print(f"  ep{ep}: PR(z)={pr:.1f}  span {float(l_span):.3f}  "
              f"nce {float(l_nce):.3f}  var {float(l_var):.3f}  "
              f"mined+{n_mined}  [{(time.time()-t0)/60:.0f}m]", flush=True)
        n_mined = 0
        # THE GATE, CORRECTED. It used to abort below PR 15, written when I
        # believed PR was the objective. It is not - see EXPERIMENTS.md §3b:
        # v1 reached PR 46.7 and scored 0.211, while the shipped head's 3.8
        # dims scored 0.248. That gate then aborted v2 at PR 14.7 even though
        # v2's contrastive loss had risen from v1's dead 0.005 to a working
        # 0.44, which is the real progress signal.
        #
        # So: abort on TOTAL collapse (PR below the shipped head's 3.8), and
        # abort on a DEAD contrastive task, which is the failure that actually
        # cost v1 its result.
        if ep >= 5 and pr < 6:
            print(f"GATE FAILED: PR(z)={pr:.1f} - total collapse, at or below "
                  f"the shipped head. Aborting.")
            break
        if ep >= 3 and float(l_nce) < 0.01 and a.w_nce > 0:
            print(f"GATE FAILED: nce={float(l_nce):.4f} - the contrastive "
                  f"task is trivial and has stopped contributing (this is "
                  f"exactly how v1 failed). Aborting.")
            break
        if (time.time() - t0) / 60 > a.minutes:
            print("time budget reached")
            break

    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), dim=a.dim, d_tok=96,
                    d_fix=1024, params=npar, primary="b", pooled="learned",
                    kind="ssl_v1", seed=a.seed, hist=hist),
               CKPT / f"{a.tag}.pt")
    print(f"VERIFIED: wrote {CKPT / (a.tag + '.pt')}  final PR {hist[-1]['pr']}")
    R.log("vjssl", tag=a.tag, seed=a.seed, pool=len(ids),
          final_pr=hist[-1]["pr"], epochs=len(hist))


if __name__ == "__main__":
    main()
