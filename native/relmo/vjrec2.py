"""Experience record v2: EARLY layer says WHERE, the predictor says WHAT KIND.

Measured (relmo/vjchange.py + the layer sweep): localisation of change in
V-JEPA 2 dies monotonically with encoder depth.

    layer  0   conc 0.785   moving/static ~1e10 (static patch: exactly 0 change)
    layer  3   conc 0.641   moving/static 6.64
    layer  6   conc 0.596   moving/static 2.62
    layer 12   conc 0.511   moving/static 1.40
    layer 24   conc 0.478   moving/static 1.03   <- v1 used only this

24 layers of full self-attention over 8192 tokens make each token a global
mixture, so anything that moves perturbs every token. v1 asked the layer-24
residual both WHERE the action was and WHAT it was, and layer 24 cannot answer
the first question at all. Hence: geometry arms at chance on the event control,
and a "still kitchen is 93% as surprising as the action" result that was an
artefact of depth, not a property of the world.

v2 splits the two questions:

  WHERE   || h_L(t,i) - h_L(now,i) ||  at an EARLY layer. This is expected
          change in a space that is still local, and unlike raw pixel
          differencing it is already abstracted a little away from shadow,
          texture and compression noise. Owner's correction applies here: this
          is a CHANGE, not an error, so it does not vanish as the model
          improves - it converges to the true change.

  WHAT    the calibrated layer-24 predictor residual, pooled with WHERE as the
          weight instead of its own magnitude. The residual stops choosing its
          own location and only has to say what kind of thing happened.

  SHAPE   geometry of the WHERE map - centroid relative to the clip's own mean
          (translation invariant, so a cabinet at top-right and a microwave at
          top-centre agree), centroid velocity (the sweep, which is what
          separates opening from closing), spread, elongation, orientation.

One forward per episode produces all of it: encoder hidden states and the
predictor pair come from the same call.

    python -m relmo.vjrec2 --episodes 120 --layer 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo.vjs import (GRID, MODEL, TUBELET, probe_dims, read_frames,  # noqa: E402
                       sample_clip, to_tensor)
from relmo.vjeval import REC, l2, parse  # noqa: E402
from relmo.vjspace import moments  # noqa: E402

OUT2 = R.BASE / "vjrec2"


def build_record(model, torch, dev, clip, n_frames, cal, layer):
    n_t, n_sp = n_frames // TUBELET, GRID * GRID
    half = n_t // 2
    S = n_t - half
    px = to_tensor(clip, torch, dev, torch.float32)
    ctx = torch.arange(0, half * n_sp, device=dev).unsqueeze(0)
    tgt = torch.arange(half * n_sp, n_t * n_sp, device=dev).unsqueeze(0)
    with torch.no_grad():
        out = model(pixel_values_videos=px, context_mask=[ctx],
                    target_mask=[tgt], output_hidden_states=True)
    hl = out.hidden_states[layer].float()[0].cpu().numpy()
    now = hl[(half - 1) * n_sp: half * n_sp]
    where = np.linalg.norm(hl[half * n_sp:].reshape(S, n_sp, -1) - now[None],
                           axis=-1)
    # temporal centring, same reason as v1: remove this clip's steady state so
    # a permanently-hard-to-embed patch does not read as a permanent event
    wc = np.clip(where - np.median(where, 0, keepdims=True), 0, None)
    po = out.predictor_output
    alpha, b = cal
    res = (alpha * po.last_hidden_state.float()[0].cpu().numpy() + b
           - po.target_hidden_state.float()[0].cpu().numpy()).reshape(S, n_sp, -1)
    what = (res * wc[..., None]).sum(1) / (wc.sum(1, keepdims=True) + 1e-9)
    return wc.reshape(S, GRID, GRID).astype(np.float32), what.astype(np.float32)


def geom_from(maps):
    mo = [moments(m) for m in maps]

    def stk(*ks):
        return np.stack([np.stack([m[k] for k in ks], -1) for m in mo])
    pos = stk("cx", "cy")
    rel = pos - pos.mean(1, keepdims=True)
    flow = np.diff(pos, axis=1, prepend=pos[:, :1])
    shape = stk("spread", "elong", "ori_c", "ori_s")

    def nz(x):
        return (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-9)
    return np.concatenate([nz(rel), nz(flow), nz(shape)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--save", action="store_true",
                    help="cache each record so the viewer and warp test reuse it")
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import VJEPA2Model

    d = REC / a.dataset
    files = sorted(p for p in d.glob("*.npz") if not p.name.startswith("_"))
    z = np.load(d / "_calib.npz")
    cal = (float(z["alpha"]), z["b"])
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(files), min(a.episodes, len(files)),
                             replace=False))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} onto {dev}...", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    print(f"  loaded | layer {a.layer} for WHERE, layer 24 predictor for WHAT "
          f"| {len(pick)} episodes", flush=True)

    cache = OUT2 / f"{a.dataset}_L{a.layer}"
    if a.save:
        cache.mkdir(parents=True, exist_ok=True)
    keep, MAPS, WHAT, OLD, FF = [], [], [], [], []
    for i in tqdm(pick, unit="ep", desc=f"rec2/L{a.layer}"):
        ep = R.dataset_dir(a.dataset) / "shard_0000" / files[i].stem / "frames.mp4"
        if not ep.exists():
            continue
        cf = cache / f"{files[i].stem}.npz"
        if a.save and cf.exists():
            zc = np.load(cf)
            m, w = zc["where_map"], zc["what_seq"]
        else:
            w_, h_ = probe_dims(ep)
            F = read_frames(ep, w_, h_)
            if len(F) < a.frames:
                continue
            m, w = build_record(model, torch, dev, sample_clip(F, a.frames),
                                a.frames, cal, a.layer)
            if a.save:
                tmp = cache / f".w_{files[i].stem}.npz"
                np.savez_compressed(tmp, where_map=m, what_seq=w)
                tmp.rename(cf)
        MAPS.append(m)
        WHAT.append(w)
        zz = np.load(files[i])
        OLD.append(zz["res_seq"])
        FF.append(zz["f_first"])
        keep.append(i)

    meta = [parse(files[i].stem) for i in keep]
    obj = np.array([m["obj"] for m in meta])
    verb = np.array([m["verb"] for m in meta])
    epid = np.array([f"{m['task']}#{m['epnum']}" for m in meta])
    GEO = l2(geom_from(MAPS))
    W = l2(np.stack(WHAT))
    O = l2(np.stack(OLD))
    FFn = l2(np.stack(FF))
    arms = {"v1 self-weighted": O, "v2 what (early-gated)": W,
            "v2 geom (shape)": GEO,
            "v2 what+geom": l2(np.concatenate([W, GEO], -1))}
    rows = {k: [] for k in list(arms) + ["scene-only"]}
    ctrl = {k: [] for k in arms}
    base, rnd_c = [], []
    for i in range(len(keep)):
        cand = np.where((obj != obj[i]) & (epid != epid[i]))[0]
        if len(cand) < a.k:
            continue
        y = (verb[cand] == verb[i])
        if y.sum() == 0:
            continue
        base.append(y.mean())
        full = np.where(epid != epid[i])[0]
        rnd_c.append((((obj[full] == obj[i]) & (verb[full] != verb[i])).mean(),
                      ((obj[full] != obj[i]) & (verb[full] == verb[i])).mean()))
        for k, M in arms.items():
            s = np.einsum("sd,nsd->n", M[i], M[cand]) / M.shape[1]
            rows[k].append(y[np.argsort(-s)[:a.k]].mean())
            sf = np.einsum("sd,nsd->n", M[i], M[full]) / M.shape[1]
            top = full[np.argsort(-sf)[:a.k]]
            ctrl[k].append((((obj[top] == obj[i]) & (verb[top] != verb[i])).mean(),
                            ((obj[top] != obj[i]) & (verb[top] == verb[i])).mean()))
        rows["scene-only"].append(
            y[np.argsort(-(FFn[cand] @ FFn[i]))[:a.k]].mean())

    b = float(np.mean(base))
    rc = np.array(rnd_c).mean(0)
    print(f"\n{len(base)} queries | {len(keep)}-episode subset | layer {a.layer}")
    print(f"{'arm':22s} {'P@k':>8s} {'lift':>7s}")
    print("-" * 40)
    for k in ["scene-only"] + list(arms):
        v = float(np.mean(rows[k]))
        print(f"{k:22s} {v:8.3f} {v/b:7.2f}")
    print(f"{'random':22s} {b:8.3f} {1.0:7.2f}")
    print(f"\nCONTROL (full pool). RANDOM: object {rc[0]:.3f} | event {rc[1]:.3f}")
    print(f"{'arm':22s} {'obj lift':>9s} {'event lift':>11s}")
    for k, v in ctrl.items():
        c = np.array(v).mean(0)
        print(f"{k:22s} {c[0]/rc[0]:9.2f} {c[1]/rc[1]:11.2f}"
              f"   {'EVENT>object' if c[1]/rc[1] > c[0]/rc[0] else 'object>EVENT'}")
    R.log("vjrec2", dataset=a.dataset, layer=a.layer, episodes=len(keep),
          queries=len(base), base_rate=round(b, 4),
          **{k.replace(" ", "_").replace("+", "_"): round(float(np.mean(v)), 4)
             for k, v in rows.items()})


if __name__ == "__main__":
    main()
