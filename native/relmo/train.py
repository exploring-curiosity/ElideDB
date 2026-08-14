"""RelMo trainer — consumes the versioned track dataset, writes
versioned checkpoints, logs everything to the append-only ledger.

Two supervised objectives, both from privileged simulator state
(training only, never at eval):
  grouping : same-body / different-body for point pairs
  contact  : per-frame contact between body pairs

Resumable: on start it loads the newest checkpoint of the run and
continues. Safe to kill at any time.

    python -m relmo.train --steps 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.device import DEVICE  # noqa: E402
from relmo.model import NT, RelMo, RelMo2  # noqa: E402

DEV = DEVICE
PMAX = 320               # points per sample


class Shards:
    """Streaming loader over the track cache. Re-scans the directory
    every epoch so episodes generated WHILE training is running are
    picked up automatically - that is the whole point of the loop."""

    def __init__(self, name, holdout=64):
        self.dir = R.TRACKS / name
        self.holdout_max = holdout
        self.files = []
        self.rescan()

    @property
    def holdout(self):
        # adaptive: a fixed 64-file holdout starved the training split
        # when the dataset was still small and the trainer slept
        # forever waiting for data that was already there
        return max(0, min(self.holdout_max, len(self.files) // 5))

    def rescan(self):
        f = sorted(self.dir.glob("ep*.npz"))
        self.files = f
        return len(f)

    def split(self):
        return self.files[self.holdout:], self.files[:self.holdout]

    @staticmethod
    def load(f, rng):
        z = np.load(f)
        xy = z["xy"].astype(np.float32)
        vis = z["vis"]
        ident = z["ident"].astype(np.int64)
        con = z["contact"].astype(np.float32)
        pdep = (z["pdepth"].astype(np.float32)
                if "pdepth" in z.files else None)
        P = xy.shape[1]
        # keep every point that sits on a body, fill the rest with
        # background points - the model must learn to separate them
        on = np.where(ident > 0)[0]
        off = np.where(ident == 0)[0]
        if len(on) < 6:
            return None
        keep_off = min(len(off), max(PMAX - len(on), 0))
        sel = np.concatenate([
            on if len(on) <= PMAX else rng.choice(on, PMAX, False),
            rng.choice(off, keep_off, False) if keep_off else
            np.zeros(0, int)]).astype(int)
        rng.shuffle(sel)
        return (xy[:, sel], vis[:, sel], ident[sel], con,
                None if pdep is None else pdep[:, sel])


def batch(files, rng, bs=4):
    out = []
    for f in rng.choice(files, bs, replace=False):
        s = Shards.load(f, rng)
        if s is not None:
            out.append(s)
    if not out:
        return None
    T = min(s[0].shape[0] for s in out)
    P = max(s[0].shape[1] for s in out)
    B = len(out)
    xy = np.zeros((B, T, P, 2), np.float32)
    vis = np.zeros((B, T, P), bool)
    ident = np.zeros((B, P), np.int64)
    cons, mask = [], np.zeros((B, P), bool)
    dep = np.full((B, T, P), np.nan, np.float32)
    for b, s in enumerate(out):
        x, v, i, c = s[0], s[1], s[2], s[3]
        d_ = s[4] if len(s) > 4 else None
        p = x.shape[1]
        xy[b, :, :p], vis[b, :, :p], ident[b, :p] = x[:T], v[:T], i
        mask[b, :p] = True
        cons.append(c[:T])
        if d_ is not None:
            dep[b, :, :p] = d_[:T]
    return (torch.tensor(xy), torch.tensor(vis), torch.tensor(ident),
            torch.tensor(mask), cons, torch.tensor(dep))


def losses(net, bt):
    parts = [x.to(DEV) if torch.is_tensor(x) else x for x in bt]
    xy, vis, ident, mask, cons = parts[:5]
    dep = parts[5] if len(parts) > 5 else None
    tok = net.tokens(xy, vis)
    z = net.point_emb(tok)
    B, P = ident.shape
    lg, lc, accs = [], [], []
    for b in range(B):
        on = torch.where((ident[b] > 0) & mask[b])[0]
        if len(on) < 6:
            continue
        zz = z[b, on]
        lab = ident[b, on]
        same = (lab[:, None] == lab[None, :]).float()
        sim = zz @ zz.T / 0.1
        iu = torch.triu_indices(len(on), len(on), 1, device=DEV)
        s, y = sim[iu[0], iu[1]], same[iu[0], iu[1]]
        # balance: same-body pairs are the minority
        w = torch.where(y > 0, (y.numel() - y.sum()) / y.sum().clamp(min=1),
                        torch.ones_like(y))
        lg.append(F.binary_cross_entropy_with_logits(
            s, y, weight=w.clamp(max=20.0)))
        accs.append((((s > 0).float() == y).float().mean()).item())
        # contact: use GT bodies to form groups (teacher forcing)
        bodies = torch.unique(lab)
        if len(bodies) >= 2:
            assign = torch.zeros(P, len(bodies), device=DEV)
            for gi, bd in enumerate(bodies):
                idx = on[lab == bd]
                assign[idx, gi] = 1.0
            gtok = net.group_pool(tok[b:b + 1], assign[None])
            con = torch.tensor(cons[b], device=DEV)
            ti = torch.linspace(0, con.shape[0] - 1, NT).long()
            for a_ in range(len(bodies)):
                for b_ in range(a_ + 1, len(bodies)):
                    tgt = (con[ti][:, bodies[a_], bodies[b_]] > 0).float()
                    lg_ = net.contact(gtok,
                                      torch.tensor([a_], device=DEV),
                                      torch.tensor([b_], device=DEV))[0]
                    pw = torch.where(tgt > 0, 3.0, 1.0)
                    lc.append(F.binary_cross_entropy_with_logits(
                        lg_, tgt, weight=pw))
    if not lg:
        return None
    L = torch.stack(lg).mean()
    Lc = torch.stack(lc).mean() if lc else torch.zeros((), device=DEV)
    Ld = torch.zeros((), device=DEV)
    if dep is not None and hasattr(net, "depth_pred"):
        idx = torch.linspace(0, dep.shape[1] - 1, NT,
                             device=DEV).long()
        tgt = dep[:, idx].permute(0, 2, 1)          # (B,P,NT)
        ok = torch.isfinite(tgt) & mask[..., None]
        if ok.any():
            # scale-free: z-score per clip so the model learns
            # relative arrangement, not this camera's metres
            m_ = tgt.clone()
            m_[~ok] = float("nan")
            mu = torch.nanmean(m_.reshape(len(tgt), -1), 1)
            sd = torch.sqrt(torch.nanmean(
                (m_.reshape(len(tgt), -1) - mu[:, None]) ** 2, 1))
            z = (tgt - mu[:, None, None]) / sd[:, None, None].clamp(min=1e-3)
            pr = net.depth_pred(net.tokens(xy, vis))
            Ld = F.smooth_l1_loss(pr[ok], z[ok])
    return L, Lc, float(np.mean(accs)), Ld


def train(dataset="physgen_v1", steps=20000, run_id=None, bs=4,
          lr=3e-4, log_every=200, ckpt_every=2000, arch="v1"):
    cfg = dict(dataset=dataset, bs=bs, lr=lr, d=128, layers=4,
               zdim=64, pmax=PMAX, nt=NT,
               model=f"RelMo-{arch}")
    run_id = run_id or f"relmo_{R.cfg_hash(cfg)}"
    out = R.model_dir(run_id)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    net = (RelMo2() if arch == "v2" else RelMo()).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    step0 = 0
    cks = sorted(out.glob("ckpt_*.pt"))
    if cks:
        sd = torch.load(cks[-1], map_location=DEV)
        net.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step0 = sd["step"]
    sh = Shards(dataset)
    rng = np.random.default_rng(0)
    man = R.read_manifest(dataset)
    R.log("train_start", run=run_id, dataset=dataset, step=step0,
          n_tracks=len(sh.files), data_fp=man.get("fingerprint"),
          config=cfg)
    t0 = time.time()
    hist = []
    for step in range(step0 + 1, steps + 1):
        if step % 500 == 0:
            sh.rescan()                     # pick up new episodes
        tr, _ = sh.split()
        if len(tr) < bs:
            R.log("train_wait", run=run_id, have=len(sh.files))
            time.sleep(20)
            sh.rescan()
            continue
        bt = batch(tr, rng, bs)
        if bt is None:
            continue
        o = losses(net, bt)
        if o is None:
            continue
        L, Lc, acc = o[0], o[1], o[2]
        Ld = o[3] if len(o) > 3 else torch.zeros((), device=DEV)
        loss = L + 0.5 * Lc + 0.3 * Ld
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        hist.append((float(L), float(Lc), acc, float(Ld)))
        if step % log_every == 0:
            h = np.array(hist[-log_every:])
            rec = dict(run=run_id, step=step, group_loss=round(float(h[:, 0].mean()), 4),
                       contact_loss=round(float(h[:, 1].mean()), 4),
                       depth_loss=round(float(h[:, 3].mean()), 4),
                       group_acc=round(float(h[:, 2].mean()), 4),
                       n_tracks=len(sh.files),
                       min_per_1k=round((time.time() - t0) / max(step - step0, 1) * 1000 / 60, 2))
            R.log("train", **rec)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        if step % ckpt_every == 0:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(),
                            step=step, cfg=cfg,
                            data_fp=R.read_manifest(dataset).get("fingerprint")),
                       out / f"ckpt_{step:07d}.pt")
            R.log("ckpt", run=run_id, step=step)
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="physgen_v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--run", default=None)
    ap.add_argument("--arch", default="v1")
    a = ap.parse_args()
    print(train(a.dataset, a.steps, a.run, arch=a.arch))
