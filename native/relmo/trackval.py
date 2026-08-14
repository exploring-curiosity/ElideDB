"""Tracker validation - the frozen tracker scored against pixel-exact
sim GT, on the SAME cached arrays the trainer consumed.

Owner directive (2026-08-11): the tracker used at eval must be
validated inside the training pipeline - "if cotracker itself fails
during eval then what?". This harness is that gate. Every track cache
is scored against pixelgt before a dataset version is accepted, and
the report is versioned in the ledger next to the data fingerprint.

Scored per point (query pixel = xy[0], all queries start at frame 0):
  EPE        position error vs GT where both claim visible
  VEL ERR    per-frame step error - THE quantity the world model must
             predict; datacheck showed it at SNR 1.8x, this says who
             is responsible and by how much
  JUMPS      frames with EPE > 4px (track lost / re-attached)
  VIS AGREE  tracker visibility vs GT depth-test visibility

    python -m relmo.trackval --dataset physgen_v2 --n 40
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import splits as SP  # noqa: E402
from relmo.pixelgt import body_speed, open_gt  # noqa: E402


def score_episode(ep_dir: Path, cache: Path):
    z = np.load(cache)
    xy = z["xy"].astype(np.float32)
    vs = z["vis"]
    gt = open_gt(ep_dir)
    T = min(len(xy), gt.T)
    # speed indexed by SEGMENT VALUE, uniform across both sources
    sp = body_speed(ep_dir, T)
    out = defaultdict(list)
    lifted = 0
    for p in range(xy.shape[1]):
        u, v = float(xy[0, p, 0]), float(xy[0, p, 1])
        lt = gt.lift(0, u, v)
        if lt is None:
            continue
        lifted += 1
        b, pb = lt
        gpx, gvis, _ = gt.track(b, pb)
        both = gvis[:T] & (vs[:T, p] > 0.5)
        if both.sum() < 8:
            continue
        epe = np.linalg.norm(xy[:T, p] - gpx[:T], axis=-1)
        # seg value: 0 = background for both sources (KubricGT
        # reports background as -1, MuJoCo as body 0)
        sv = 0 if b < 0 else (b if gt.driver != "kubric" else b + 1)
        moving = (sp[:T, sv] > 0.02 if sv > 0 and sv < sp.shape[1]
                  else np.zeros(T, bool))
        # per-frame velocity error - consecutive frames both valid
        cv = both[1:] & both[:-1]
        verr = np.linalg.norm(np.diff(xy[:T, p], axis=0)
                              - np.diff(gpx[:T], axis=0), axis=-1)
        gstep = np.linalg.norm(np.diff(gpx[:T], axis=0), axis=-1)
        cls = "moving" if moving.any() else ("body" if sv > 0 else "bg")
        out["epe_" + cls].extend(epe[both].tolist())
        out["verr_" + cls].extend(verr[cv].tolist())
        if cls == "moving":
            mv = cv & moving[1:]
            out["gstep_moving"].extend(gstep[mv].tolist())
            out["verr_at_motion"].extend(verr[mv].tolist())
        out["jump_" + cls].append(float((epe[both] > 4.0).mean()))
        out["visagree"].append(float(((vs[:T, p] > 0.5) == gvis[:T]).mean()))
    return out, lifted, xy.shape[1], gt.driver


def stats(v):
    if not len(v):
        return None
    a = np.asarray(v)
    return dict(med=round(float(np.median(a)), 3),
                p90=round(float(np.percentile(a, 90)), 3),
                mean=round(float(a.mean()), 3))


def run(dataset="physgen_v2", n=40, seed=0, tracker="cotracker3_offline",
        split="all"):
    """Score a tracker against sim GT on ONE split.

    Split becomes load-bearing the moment a tracker is FINETUNED:
    sampling the whole corpus would put ~80% training episodes in
    the report and inflate it. It is harmless for a stock checkpoint
    (nothing is held out from a model that never trained), which is
    why the existing frozen baselines were run with split='all' -
    but any finetuned checkpoint must be scored on 'test', which no
    trainer and no selection step has ever read. The split is
    recorded in both the report and the filename so two numbers can
    never be silently compared across different populations."""
    from tqdm import tqdm
    man = R.read_manifest(dataset)
    rng = np.random.default_rng(seed)
    if split == "all":
        pool = man["episodes"]
    else:
        keep = {f.stem for f in SP.partition(
            sorted((R.TRACKS / dataset).glob("*.npz")), dataset)[split]}
        pool = [e for e in man["episodes"] if e["id"] in keep]
    if not pool:
        raise SystemExit(f"no episodes in split {split!r} of {dataset}")
    # Score only what has a cached track. Without this, sampling n at
    # random from the manifest silently scores far fewer episodes than
    # asked whenever tracking is partial - the caches that do not exist
    # are skipped in the loop below and the report still says n. Scoring
    # a partially tracked corpus is the normal case while a tracking run
    # is still in flight, and it is exactly when an early read is worth
    # the most, so make the population explicit instead of accidental.
    have = {p.stem for p in (R.TRACKS / dataset).glob("*.npz")}
    pool = [e for e in pool if e["id"] in have] or pool
    n = min(n, len(pool))
    eps = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
    agg = defaultdict(list)
    lifted = total = 0
    by_driver = defaultdict(lambda: defaultdict(list))
    for e in tqdm(eps, unit="ep", desc="trackval"):
        cache = R.TRACKS / dataset / f"{e['id']}.npz"
        ep_dir = R.dataset_dir(dataset) / e["shard"] / e["id"]
        if not cache.exists():
            continue
        out, lf, tot, driver = score_episode(ep_dir, cache)
        lifted += lf
        total += tot
        for k, v in out.items():
            agg[k].extend(v)
            by_driver[driver][k].extend(v)
    sig = stats(agg["gstep_moving"])
    noise = stats(agg["verr_at_motion"])
    rep = dict(
        dataset=dataset, tracker=tracker, split=split,
        episodes=len(eps), pool=len(pool),
        lift_rate=round(lifted / max(total, 1), 3),
        epe_px=dict(moving=stats(agg["epe_moving"]),
                    resting_body=stats(agg["epe_body"]),
                    background=stats(agg["epe_bg"])),
        vel_err_px=dict(moving=stats(agg["verr_moving"]),
                        resting_body=stats(agg["verr_body"]),
                        background=stats(agg["verr_bg"])),
        vel_snr_at_motion=round(sig["med"] / max(noise["med"], 1e-6), 2)
        if sig and noise else None,
        jump_frac=dict(moving=stats(agg["jump_moving"]),
                       resting_body=stats(agg["jump_body"])),
        vis_agreement=stats(agg["visagree"]),
        per_driver={d: dict(epe_moving=stats(v["epe_moving"]),
                            verr_moving=stats(v["verr_moving"]))
                    for d, v in sorted(by_driver.items())})
    R.log("trackval", **rep)
    suffix = "" if split == "all" else f"_{split}"
    (R.TRACKS / dataset / f"trackval_{tracker}{suffix}.json").write_text(
        json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="physgen_v2")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="all",
                    choices=["all", "train", "val", "test"],
                    help="use 'test' for any FINETUNED tracker")
    ap.add_argument("--tracker", default="cotracker3_offline")
    a = ap.parse_args()
    print(json.dumps(run(a.dataset, a.n, tracker=a.tracker,
                         split=a.split), indent=1))
