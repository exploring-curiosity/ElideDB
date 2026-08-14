"""L5.1 — does any of this move the PRODUCT metric?

Everything measured in L1.x is a probe AUC. A probe only needs a
separating direction; retrieval needs a METRIC - distances that are
comparable across scenes, cameras and object instances. Those are
different requirements, and a feature set can satisfy the first and fail
the second. This closes the loop before any learned head is built on
features whose retrieval value is assumed.

WHAT THE BARS ACTUALLY ARE - and this reframes the whole comparison.
Reading relsep.traj_feats and rel_feats: every existing reference point
is PRIVILEGED.

  chance 0.108   task prior
  WM     0.191   trunk features of a model fed gxy + gdist (GT 3D)
  TRAJ   0.235   reads xpos AND target_bodies - it is told WHICH BODY is
                 the manipulation target and given its exact 3D path
  REL    0.314   reads contact_pairs and qpos - sim contact and joint state

So 0.235 is not a "hand-written baseline" a video system should trivially
clear; it is an oracle that already knows the answer to the hardest part
of the problem (which object matters). The descriptor here reads only
CoTracker xy and vis. Landing anywhere near 0.235 is therefore a much
stronger result than the number alone suggests, and landing below it is
not by itself evidence that the features are inadequate.

THE SCENE CONFOUND. RoboCasa renders each demo in one of many kitchen
layouts/styles, and a task correlates with the kitchens it was captured
in. A descriptor that encodes "this room" would post a good task-mAP
while retrieving nothing about the event. mAP alone cannot show this, so
two controls run beside it:

  kitchen-mAP     relevance = same (layout_id, style_id) instead of same
                  task. High means the descriptor encodes the room.
  cross-kitchen   task-mAP with same-kitchen neighbours ALSO excluded, so
                  every retrieved neighbour must come from a different
                  room. This is the retrieval analogue of the `flat`
                  control - it removes the shortcut rather than measuring
                  around it.

layout_id/style_id are read from the SOURCE demo's ep_meta.json. They are
scoring metadata only, exactly as task labels are; nothing reads them to
build a descriptor.

    python -m relmo.retr2 --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import splits as SP  # noqa: E402
from relmo.relg0 import pixel_feats  # noqa: E402
from relmo.relg3 import F2, F2C, comotion  # noqa: E402
from relmo.relsep import rel_feats, traj_feats  # noqa: E402

W = 24
WIN_STRIDE = W // 2     # 12, matching the stride rcreplay emitted


def window_starts_of(z, stride=WIN_STRIDE):
    """Window starts for an episode, WITHOUT reading any sim field.

    Prefers the cached `window_starts` when present, so every number this
    module has already produced is bit-identical to before this fix.
    Otherwise it derives windows from the track length alone — a window
    start is a frame index, not an observation, and `xy.shape[0]`
    determines it.

    THE TWO ARE NOT THE SAME THING, and the difference is honest rather
    than hidden. rcreplay.py:358 chose its starts with a 60-frame warmup
    skip and a per-window motion/visibility gate — it knows which frames
    contain the demo's real motion because it has the simulator. At
    serve there is no warmup and no motion oracle, so the fallback
    tiles the WHOLE clip at stride W/2. On this corpus tracks2.py:239
    measured the two as equivalent in content (uniform windows median
    max-displacement 0.0750 m, 9% under 1 cm; gated windows 0.0722 m,
    10% under 1 cm), so the fallback is not expected to shift results —
    but it will produce MORE windows per episode, which is what a serve
    system genuinely has to score."""
    if "window_starts" in z.files:
        return np.asarray(z["window_starts"], dtype=np.int32)
    if "xy" not in z.files:
        return None
    T = int(z["xy"].shape[0])
    if T < W:
        return None
    return np.arange(0, T - W + 1, stride, dtype=np.int32)


def kitchen_of(dataset):
    """episode id -> (layout_id, style_id) from the SOURCE demo.

    Scoring metadata for the confound control, never a descriptor input."""
    from relmo.rcreplay import episodes
    out, cache = {}, {}
    man = R.read_manifest(dataset)
    for e in man["episodes"]:
        task = e["id"].split("_episode_")[0]
        m = re.search(r"_episode_(\d+)", e["id"])
        if not m:
            continue
        demo = "episode_" + m.group(1)
        if task not in cache:
            eps = {}
            for sp in ("pretrain", "target"):
                for d in episodes(task, sp):
                    eps.setdefault(d.name, d)
            cache[task] = eps
        d = cache[task].get(demo)
        if d is None:
            continue
        p = d / "ep_meta.json"
        if not p.exists():
            continue
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        out[e["id"]] = (int(j.get("layout_id", -1)), int(j.get("style_id", -1)))
    return out


def prep(M, fit=None):
    """Heterogeneous scalar features -> a vector a cosine can compare.

    This is the step that separates a metric from a probe. A logistic
    probe is invariant to per-feature scaling - it just rescales its own
    coefficient - so scaling never mattered for the AUCs. A cosine is
    NOT: one heavy-tailed ratio with a large variance silently becomes
    the distance. These features include unbounded ratios spanning orders
    of magnitude beside 0-1 fractions, so:

      signed log1p   tames the ratios without discarding sign
      winsorise      clips to the fitting set's 2/98 percentiles so a
                     single outlier cannot define the axis
      z-score        equalises each feature's contribution
      L2             makes the inner product a cosine

    `fit` carries the constants so they can be estimated on TRAIN
    episodes and APPLIED to held-out ones - fitting them on everything
    is transduction and would leak the evaluation set into the metric."""
    M = np.asarray(M, np.float64)
    M = np.log1p(np.abs(M)) * np.sign(M)
    if fit is None:
        lo, hi = np.quantile(M, 0.02, axis=0), np.quantile(M, 0.98, axis=0)
        Mc = np.clip(M, lo, hi)
        mu, sd = Mc.mean(0), np.maximum(Mc.std(0), 1e-6)
        fit = dict(lo=lo, hi=hi, mu=mu, sd=sd)
    Z = (np.clip(M, fit["lo"], fit["hi"]) - fit["mu"]) / fit["sd"]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return Z, fit


def score(M, labels, eps, k=5, extra_block=None):
    """mAP and recall@k. Same-episode neighbours always excluded.

    extra_block optionally excludes another grouping too (the kitchen),
    which is how the scene shortcut is removed rather than measured."""
    S = M @ M.T
    n = len(M)
    labels = np.asarray(labels)
    ap, rec = [], []
    for i in range(n):
        mask = np.array([eps[j] != eps[i] for j in range(n)])
        if extra_block is not None:
            mask &= np.array([extra_block[j] != extra_block[i]
                              for j in range(n)])
        if mask.sum() < 2:
            continue
        idx = np.argsort(-S[i][mask])
        rel = (labels[mask][idx] == labels[i]).astype(float)
        if rel.sum() == 0:
            continue
        hits = np.cumsum(rel)
        prec = hits / np.arange(1, len(rel) + 1)
        ap.append(float((prec * rel).sum() / rel.sum()))
        rec.append(float(rel[:k].max()))
    return float(np.mean(ap)), float(np.mean(rec)), len(ap)


def boot_map(M, labels, eps, ids, n=500, seed=0, extra_block=None):
    """Bootstrap mAP by EPISODE. 300 windows come from 100 episodes and
    windows within one episode share a scene and a label, so a
    window-level interval would be far too narrow."""
    rng = np.random.default_rng(seed)
    uid = np.unique(ids)
    byep = [np.where(ids == u)[0] for u in uid]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uid), len(uid))
        m = np.concatenate([byep[i] for i in pick])
        # de-duplicate: a repeated episode would otherwise retrieve itself
        m = np.unique(m)
        if len(m) < 20:
            continue
        a, _, _ = score(M[m], labels[m], [eps[i] for i in m],
                        extra_block=None if extra_block is None
                        else [extra_block[i] for i in m])
        out.append(a)
    o = np.array(out)
    return float(o.mean()), float(np.quantile(o, .025)), \
        float(np.quantile(o, .975))


def collect(dataset, per_ep):
    man = R.read_manifest(dataset)
    by = {e["id"]: e for e in man["episodes"]}
    kit = kitchen_of(dataset)
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    part = SP.partition(files, dataset)
    tr_ids = {f.stem for f in part[SP.TRAIN]}
    rows = []
    for f in files:
        if f.stem not in by:
            continue
        z = np.load(f)
        # SERVE-PATH FIX (2026-08-13). This previously read
        #     if "window_starts" not in z.files or
        #        "contact_pairs" not in z.files: continue
        # which gated CORPUS MEMBERSHIP on a SIM field. contact_pairs is
        # MuJoCo's contact list; it does not exist on real video, so at
        # serve this dropped 100% of episodes silently — the retrieval
        # index would simply have been empty. It was also vestigial:
        # nothing below this line reads contact_pairs (the descriptor is
        # px/cm/po/tr/re, all from xy+vis). The gate came from
        # backfill.FIELDS, which copies window_starts across in the SAME
        # payload as the sim state, so the two got treated as one thing.
        # window_starts is NOT privileged — it is just "every Wth frame"
        # — so derive it when absent instead of dropping the episode.
        starts = window_starts_of(z)
        if starts is None:
            continue
        for t0 in starts[:per_ep]:
            t0 = int(t0)
            px = pixel_feats(z, t0, W)
            cm = comotion(z, t0, W)
            po = pose_feats(z, t0)
            if px is None or cm is None or po is None:
                continue
            rows.append(dict(
                id=f.stem, t0=t0, ep=f.stem.split("__")[0],
                task=f.stem.split("_episode_")[0],
                kitchen=str(kit.get(f.stem, ("?", "?"))),
                train=f.stem in tr_ids,
                px=px, cm=cm, po=po,
                tr=traj_feats(z, t0, W), re=rel_feats(z, t0, W)))
    return rows


# POSE-DEPENDENT quantities, deliberately absent from the probe features.
# Measured: they are the single biggest lever on retrieval (mAP 0.172 ->
# 0.202, r@5 0.574 -> 0.646). A CLASSIFIER wants invariance - it should
# call an opening a drawer an opening whichever way the camera faces, so
# every probe feature was built as a ratio or an angle-free statistic. A
# retrieval METRIC wants the opposite: task identity is largely carried
# by WHICH WAY and HOW FAR something moved, and the invariances that make
# a good probe feature throw exactly that away. This is the concrete form
# of "a probe needs a direction, a metric needs distances".
POSE = ("p_dir_x", "p_dir_y", "p_logmag", "p_cx", "p_cy", "p_logsize")

TRk = ("dir_x", "dir_y", "dir_z", "logmag", "straightness")
REk = ("grip_frac", "third_frac", "n_partners", "contact_changes",
       "articulation")


def pose_feats(z, t0):
    """Image-plane direction, magnitude, position and scale of the mover."""
    from relmo.relg0 import MOVE_Q
    a, b = t0, min(t0 + W, len(z["xy"]))
    X = z["xy"][a:b].astype(np.float64)
    ok = z["vis"][a:b].astype(bool).all(0)
    if ok.sum() < 12:
        return None
    Xo = X[:, ok]
    d = np.linalg.norm(Xo[-1] - Xo[0], axis=-1)
    mv = d >= max(np.quantile(d, MOVE_Q), 1e-6)
    if mv.sum() < 4:
        return None
    M = Xo[:, mv]
    cen = M.mean(1)
    v = cen[-1] - cen[0]
    n = float(np.linalg.norm(v)) + 1e-9
    Wd = float(z["width"]) if "width" in z.files else 320.0
    Hd = float(z["height"]) if "height" in z.files else 240.0
    sz = float(np.linalg.norm(M[0] - M[0].mean(0), axis=-1).mean())
    return dict(p_dir_x=v[0] / n, p_dir_y=v[1] / n,
                p_logmag=float(np.log10(n + 1e-6)),
                p_cx=float(cen[:, 0].mean() / Wd),
                p_cy=float(cen[:, 1].mean() / Hd),
                p_logsize=float(np.log10(sz + 1e-6)))


def build(rows, which):
    """Assemble one descriptor bank from named feature families."""
    def vec(r):
        v = []
        if "motion" in which:
            v += [r["px"][k] for k in F2]
        if "comotion" in which:
            v += [r["cm"][k] for k in F2C]
        if "pose" in which:
            v += [r["po"][k] for k in POSE]
        if "traj" in which:
            v += [r["tr"][k] for k in TRk]
        if "rel" in which:
            v += [r["re"][k] for k in REk]
        return v
    return np.array([vec(r) for r in rows], np.float64)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dataset", default="rcasa")
    ap_.add_argument("--per-ep", type=int, default=3)
    a = ap_.parse_args()
    rows = collect(a.dataset, a.per_ep)
    tasks = np.array([r["task"] for r in rows])
    eps = [r["ep"] for r in rows]
    ids = np.array([r["id"] for r in rows])
    kits = [r["kitchen"] for r in rows]
    tr = np.array([r["train"] for r in rows])
    nk = len(set(kits))
    print(f"{a.dataset}: {len(rows)} windows, {len(set(eps))} episodes, "
          f"{len(set(tasks))} tasks, {nk} kitchens "
          f"({sum(1 for k in kits if k.startswith('(?'))} unresolved)")
    chance = float(np.mean([(tasks == q).mean() for q in tasks]))
    kchance = float(np.mean([(np.array(kits) == q).mean() for q in kits]))
    print(f"chance task precision {chance:.3f} | chance kitchen precision "
          f"{kchance:.3f}\n")

    rng = np.random.default_rng(0)
    banks = {
        "RANDOM": ("floor", rng.normal(size=(len(rows), 16))),
        "TRAJ (privileged xpos+target)": ("priv", build(rows, {"traj"})),
        "REL oracle (privileged sim)": ("priv", build(rows, {"rel"})),
        "2D motion": ("video", build(rows, {"motion"})),
        "2D co-motion": ("video", build(rows, {"comotion"})),
        "2D motion + co-motion": ("video", build(rows, {"motion",
                                                        "comotion"})),
        "2D pose (dir/mag/pos)": ("video", build(rows, {"pose"})),
        "FULL video descriptor": ("video", build(rows, {"motion", "comotion",
                                                        "pose"})),
    }
    print(f"{'descriptor':32s} {'kind':6s} {'mAP':>6s} {'r@5':>6s} "
          f"{'mAP[95% CI]':>18s} {'kitchen-mAP':>12s} {'xkitchen-mAP':>13s}")
    print("-" * 100)
    res = {}
    for nm, (kind, M0) in banks.items():
        # FIT ON TRAIN EPISODES, APPLY TO ALL. The historical numbers were
        # produced by z-scoring over every window at once; doing that here
        # would let the evaluation set define the metric.
        _, fit = prep(M0[tr])
        Z, _ = prep(M0, fit)
        m, r, n = score(Z, tasks, eps)
        bm, lo, hi = boot_map(Z, tasks, eps, ids)
        km, _, _ = score(Z, np.array(kits), eps)
        xm, xr, xn = score(Z, tasks, eps, extra_block=kits)
        res[nm] = dict(kind=kind, mAP=round(m, 4), recall_at_5=round(r, 4),
                       ci=[round(lo, 4), round(hi, 4)],
                       kitchen_mAP=round(km, 4), xkitchen_mAP=round(xm, 4),
                       xkitchen_r5=round(xr, 4), n=n, n_xkitchen=xn)
        print(f"{nm:32s} {kind:6s} {m:6.3f} {r:6.3f} "
              f"  [{lo:+.3f},{hi:+.3f}] {km:12.3f} {xm:13.3f}")
    print(f"\nchance: task {chance:.3f} | kitchen {kchance:.3f}")
    print("kitchen-mAP near chance = the descriptor is NOT encoding the "
          "room.\nxkitchen-mAP is task retrieval with same-kitchen "
          "neighbours removed entirely.")
    print("\nhistorical reference (transductive norm, same protocol): "
          "chance 0.108 | WM 0.191 | TRAJ 0.235 | REL 0.314")
    R.log("retrieval_v2", dataset=a.dataset, windows=len(rows),
          episodes=len(set(eps)), kitchens=nk, chance=round(chance, 4),
          kitchen_chance=round(kchance, 4), results=res)
    (R.BASE / "retrieval_v2.json").write_text(json.dumps(res, indent=1))
