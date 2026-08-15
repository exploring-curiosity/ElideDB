"""Duration invariance, isolated. The metric this work is actually optimising.

WHY NOT precision@support. On rcasa, duration is itself a group CUE - group
trace lengths are 8-16 steps for Close/sliding and 112-160 for Stack/hinged,
and a ranker using ONLY duration and no pixels scores 0.315 against a chance of
0.215. So a change that makes the system duration-INVARIANT removes a shortcut
the aggregate metric is currently rewarding, and the aggregate can fall while
the system gets more correct. Judged by precision@support alone, the right
change looks like a regression.

THE TEST, which needs no labels. Take one recording and replay its own footage
at a different speed, then ask whether it still retrieves the ORIGINAL out of
the whole corpus. The right answer is "itself" and is known without annotation.
Nothing else about the recording changes - same scene, same camera, same
object, same event, same lighting. Only duration.

    factor 0.5   twice as fast
    factor 1.0   control - must be rank 1, and is the check that the harness
                 itself is sound
    factor 2.0   twice as slow
    factor 3.0   three times as slow

Reported as rank-1 accuracy and mean reciprocal rank of the source recording.
A system with duration invariance holds MRR near 1.0 across factors; one
without it degrades as the factor moves away from 1.

This differs from relmo/vjtime, which warps the PROFILE inside a clip
(ease_in/ease_out/sigmoid) while holding total duration fixed. Under v4 global
duration was free by construction - sample_clip spread a fixed frame count over
whatever the episode's length was - so there was nothing to measure. Under
stream time there is.

    python -m relmo.vjwarp --ingest --episodes 16
    python -m relmo.vjwarp --arms time,arc
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
from relmo import vjrel  # noqa: E402
from relmo.vjeval import REC, l2  # noqa: E402
from relmo.vjmatch import arc_resample, dtw  # noqa: E402
from relmo.vjrec6 import OUT6, episode_paths, record_stream  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402
from relmo.vjzeval import PAD_COST, _pad  # noqa: E402

OUTW = R.BASE / "vjwarp"
# Factors must NOT sit on the rate ladder. With rates 8 / 4 / 8-3 fps the
# achievable exact ratios are {1/3, 1/2, 2/3, 1, 3/2, 2, 3}, and a factor drawn
# from that set makes the test DEGENERATE: a 2x-slow clip encoded at 4 fps
# resamples the identical source frames as the original at 8 fps, so multi-rate
# scores rank-1 1.000 by numerical identity rather than by invariance. That is
# what the first run measured. These factors fall between rungs, so a rate pair
# can only ever get close, never exact.
# 1.0 is the harness control. 1.25 and 1.75 sit INSIDE vjrel.R_CUT = 2.0 and
# must still retrieve the source - same moment. 2.5 sits BEYOND it and must NOT:
# under the owner's definition a 2.5x-slower replay is a different moment, so
# rank-1 there is a FAILURE, not a success. The metric is two-sided.
FACTORS = (1.0, 1.25, 1.75, 2.5)
# (stream_fps, hop_s, suffix). Window spans 32/fps seconds, so the coarse rates
# only exist for recordings long enough to hold one window - which is exactly
# when a coarse rate is what a short query needs to match against.
RATES = ((8.0, 2.0, ""), (4.0, 4.0, "_f4"), (8.0 / 3.0, 6.0, "_f3"))


def warp_index(n_frames, factor):
    """Frame indices that replay n_frames over factor*n_frames of wall time."""
    m = max(2, int(round(n_frames * factor)))
    return np.linspace(0, n_frames - 1, m).round().astype(int)


def gate_of(z):
    return z["where_map"].reshape(len(z["step"]), -1).sum(1)


def ingest(args):
    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model
    from relmo.vjs import MODEL, probe_dims, read_frames
    from relmo.vjsig import RES

    sp = load_split()
    eps = [e for e in episode_paths(args.dataset) if e[0] in sp["val"]]
    # short recordings only: a 3x warp of a 40 s clip is 120 s, and the DTW
    # cost is quadratic in trace length. Selection is on DURATION, which is
    # visible without any label.
    lens = {}
    for eid, _, _ in eps:
        f = OUT6 / f"{args.dataset}_L6" / f"{eid}.npz"
        if f.exists():
            lens[eid] = len(np.load(f)["step"])
    eps = [e for e in eps if lens.get(e[0], 999) <= args.max_steps]
    eps = eps[:args.episodes]
    if not eps:
        raise SystemExit("no short-enough val recordings found")

    z = np.load(REC / args.dataset / "_calib.npz")
    cal = (float(z["alpha"]), z["b"].astype(np.float32))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt_t = torch.float16
    print(f"loading {MODEL} onto {dev} in fp16...", flush=True)
    vj = VJEPA2Model.from_pretrained(MODEL, dtype=dt_t).to(dev).eval()
    from transformers import AutoModel
    from relmo.vjsig import MODEL as SIGM
    sig = AutoModel.from_pretrained(SIGM, dtype=torch.float16).to(dev).eval()
    print(f"  loaded both | {len(eps)} recordings x {len(FACTORS)} factors",
          flush=True)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    OUTW.mkdir(parents=True, exist_ok=True)
    todo = [(e, f, r) for e in eps for f in FACTORS for r in RATES
            if not (OUTW / f"{e[0]}__f{f:g}{r[2]}.npz").exists()]
    t0, done = time.time(), 0
    cache = {}
    for (eid, mp4, src_fps), f, (fps_r, hop_r, sfx) in tqdm(
            todo, unit="clip", desc="warp"):
        if eid not in cache:
            cache.clear()
            w_, h_ = probe_dims(mp4)
            cache[eid] = read_frames(mp4, w_, h_)
        Fw = cache[eid][warp_index(len(cache[eid]), f)]
        rec = record_stream(vj, torch, dev, Fw, src_fps, cal, 6, dt_t,
                            stream_fps=fps_r, hop_s=hop_r)
        if rec is None:
            continue
        sel = Fw[np.clip(rec["frame0"].astype(int), 0, len(Fw) - 1)]
        V = []
        for s in range(0, len(sel), 64):
            x = torch.tensor(sel[s:s + 64]).permute(0, 3, 1, 2).float().div_(255.)
            x = torch.nn.functional.interpolate(x, size=(RES, RES),
                                                mode="bilinear",
                                                align_corners=False)
            with torch.no_grad():
                V.append(sig.get_image_features(
                    pixel_values=((x - mean) / std).to(dev, torch.float16)
                ).float().cpu().numpy())
        rec["sig"] = np.concatenate(V).astype(np.float32)
        rec["factor"] = np.float32(f)
        rec["source"] = eid
        rec["rate"] = np.float32(fps_r)
        np.savez_compressed(OUTW / f"{eid}__f{f:g}{sfx}.npz", **rec)
        done += 1
    print(f"\nVERIFIED on disk: {len(list(OUTW.glob('*.npz')))} warped clips "
          f"({done} written this run, {(time.time()-t0)/60:.1f} min)")
    R.log("vjwarp_ingest", dataset=args.dataset, episodes=len(eps),
          factors=list(FACTORS), written=done)


def evaluate(args):
    from relmo import vjz
    from relmo.vjood import recs_for

    sp = load_split()
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    want_rates = [r for r in RATES if r[2] in ("",) or "multi" in arms]

    # ---- pool, one dict per rate. A recording absent at a rate is None ----
    pool_rate, raw_rate = {}, {}
    for _, _, sfx in RATES:
        rec = OUT6 / f"{args.dataset}_L6{sfx}"
        if not rec.exists():
            continue
        ids = sorted(p.stem for p in rec.glob("*.npz")
                     if not p.name.startswith("."))
        d = vjz.gather(ids, args.dataset, rec_dir=rec,
                       sig_dir=R.BASE / "vjsig6" / f"{args.dataset}{sfx}",
                       arc=args.ds if args.tag else 0.0)
        pool_rate[sfx] = d
        raw_rate[sfx] = {i: np.load(rec / f"{i}.npz") for i in d}
    pool_ids = sorted(pool_rate[""])
    warped = sorted(OUTW.glob("*.npz"))
    if not warped:
        raise SystemExit("no warped clips - run --ingest first")

    ds = args.ds
    if ds <= 0:
        tr = sorted(sp["train"] & set(pool_rate[""]))
        ds = float(np.median(np.concatenate([gate_of(raw_rate[""][i])
                                             for i in tr])))
        print(f"ds not given; using the train median gate energy {ds:.2f}")

    model = None
    if args.tag:
        from relmo import vjrank
        model, _ = vjrank.load_ckpt(args.tag)
        print(f"trained arm: {args.tag} - records are gathered pre-arced and "
              f"encoded to z, then matched by the same anchored DTW")

    def prep(a, gate, arm):
        if "arc" not in arm or model is not None:
            return a                       # gather() already arced for a tag
        return arc_resample(a, gate, ds, max_len=256)[0]

    def to_z(rec):
        """dict(a,g,sig) -> the trained z sequence."""
        import torch
        a_, m_ = vjrank.pad([rec["a"]], "cpu", torch)
        g_, _ = vjrank.pad([rec["g"]], "cpu", torch)
        s_, _ = vjrank.pad([rec["sig"]], "cpu", torch)
        with torch.no_grad():
            return model.encode(a_, g_, s_, m_)[0].numpy()

    # ---- queries, grouped by (source, factor) across rates ----
    Q = {}
    for w in warped:
        z = np.load(w, allow_pickle=True)
        # clips written before multi-rate carry no `rate`; they were all 8 fps
        rate = round(float(z["rate"]), 4) if "rate" in z.files else 8.0
        sfx = {8.0: "", 4.0: "_f4"}.get(rate, "_f3")
        a, g = vjz.channels(z)
        gate = gate_of(z)
        sg = z["sig"] if "sig" in z.files else None
        Q.setdefault((str(z["source"]), float(z["factor"])), {})[sfx] = \
            (a, gate, g, sg)

    print(f"pool {len(pool_ids)} recordings | rates on disk "
          f"{[k or '_f8' for k in sorted(pool_rate)]} | "
          f"{len(Q)} warped queries | chance MRR "
          f"{np.mean([1/(k+1) for k in range(len(pool_ids))]):.4f}")

    for arm in arms:
        rates = list(pool_rate) if "multi" in arm else [""]
        # cache the padded pool per rate
        PP = {}
        for sfx in rates:
            d = pool_rate[sfx]
            present = np.array([i in d for i in pool_ids])
            dim = model.d if model is not None else 1024
            desc = [(to_z(d[i]) if model is not None
                     else prep(d[i]["a"], gate_of(raw_rate[sfx][i]), arm))
                    if i in d else np.zeros((2, dim), np.float32)
                    for i in pool_ids]
            P, ok = _pad([l2(x) for x in desc])
            PP[sfx] = (P, ok, np.array([len(x) for x in desc]), present)

        rows = {}
        for (src, fac), per_rate in Q.items():
            if src not in pool_ids:
                continue
            best = np.full(len(pool_ids), np.inf)
            mism = np.full(len(pool_ids), np.inf)
            for qsfx, (qa, qg, qgg, qsg) in per_rate.items():
                if "multi" not in arm and qsfx != "":
                    continue
                if model is not None:
                    from relmo.vjmatch import arc_resample as _ar
                    aa, rest = _ar(qa, qg, ds, aux=[qgg, qsg], max_len=256)
                    q = l2(to_z(dict(a=aa, g=rest[0], sig=rest[1])))
                else:
                    q = l2(prep(qa, qg, arm))
                if len(q) < 2:
                    continue
                for sfx in rates:
                    P, ok, L, present = PP[sfx]
                    if not present.any():
                        continue
                    C = 1.0 - np.einsum("sd,nkd->nsk", q, P[present])
                    C = np.where(ok[present][:, None, :], C, PAD_COST)
                    d = dtw(C, False, L[present])
                    idx = np.where(present)[0]
                    if args.mr_mode == "min":
                        best[idx] = np.minimum(best[idx], d)
                    else:
                        mm = np.abs(np.log(len(q) / np.maximum(L[present], 1)))
                        t = mm < mism[idx]
                        best[idx[t]] = d[t]
                        mism[idx[t]] = mm[t]
            order = np.argsort(best)
            rank = 1 + int(np.where(np.array(pool_ids)[order] == src)[0][0])
            rows.setdefault(fac, []).append(rank)

        print(f"\n  arm = {arm}")
        print(f"  {'factor':>7s} {'n':>4s} {'rank-1':>7s} {'top-5':>7s} "
              f"{'MRR':>7s} {'med rank':>9s}")
        for fac in sorted(rows):
            r = np.array(rows[fac], float)
            print(f"  {fac:7.1f} {len(r):4d} {float((r==1).mean()):7.3f} "
                  f"{float((r<=5).mean()):7.3f} {float((1/r).mean()):7.3f} "
                  f"{int(np.median(r)):9d}")
        from relmo.vjrel import R_CUT
        inb = [np.array(v, float) for k, v in rows.items()
               if k != 1.0 and max(k, 1 / k) < R_CUT]
        out = [np.array(v, float) for k, v in rows.items()
               if max(k, 1 / k) >= R_CUT]
        if inb:
            a_ = np.concatenate(inb)
            print(f"  {'IN-BAND':>7s} {len(a_):4d} {float((a_==1).mean()):7.3f} "
                  f"{float((a_<=5).mean()):7.3f} {float((1/a_).mean()):7.3f}"
                  f"   <- same moment, WANT rank 1")
        if out:
            b_ = np.concatenate(out)
            print(f"  {'BEYOND':>7s} {len(b_):4d} {float((b_==1).mean()):7.3f} "
                  f"{float((b_<=5).mean()):7.3f} {float((1/b_).mean()):7.3f}"
                  f"   <- different moment, want NOT rank 1 "
                  f"(reject {float((b_>1).mean()):.3f})")
        if inb and out:
            a_, b_ = np.concatenate(inb), np.concatenate(out)
            sep = float((a_ == 1).mean()) - float((b_ == 1).mean())
            print(f"  {'MARGIN':>7s}      {sep:+7.3f}"
                  f"                       <- in-band rank-1 minus beyond rank-1;"
                  f" this is the number")
            R.log("vjwarp", arm=arm, ds=round(ds, 3),
                  inband_rank1=round(float((a_ == 1).mean()), 4),
                  beyond_rank1=round(float((b_ == 1).mean()), 4),
                  margin=round(sep, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--factors", default="")
    ap.add_argument("--mr-mode", default="min", choices=["min","match"])
    ap.add_argument("--tag", default="", help="ranker checkpoint; "
                    "encodes to trained z instead of raw a_t")
    ap.add_argument("--arms", default="time,arc")
    ap.add_argument("--ds", type=float, default=-1.0)
    a = ap.parse_args()
    if a.factors:
        global FACTORS
        FACTORS = tuple(float(x) for x in a.factors.split(','))
    if a.ingest:
        ingest(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
