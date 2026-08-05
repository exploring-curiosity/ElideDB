"""STEP 4, principled: surprise from the SHIPPED predictor.

Everything earlier in this build approximated "prediction" with my own
finite differences in embedding space (linear extrapolation, block
means), each carrying a scale and a threshold I chose - configuration
that turned out to be domain-dependent (sim wanted 2 s extrapolation,
driving wanted 0.5 s direction change; neither transferred).

V-JEPA 2 ships a self-supervised PREDICTOR (22 M params, trained on
generic video, no labels, no downstream head, nothing from this
project's data). Asking it directly - "predict the latents of the next
temporal group from all preceding groups" - gives a calibrated
residual whose time resolution is set by the model's own tubelet grid,
not by me. There is no scale parameter and no threshold to pick.

    python native/predsurprise.py --grade
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

MID = "facebook/vjepa2-vitl-fpc64-256"
NFRAME = 16          # frames per clip fed to the model
CLIP_STRIDE = 4      # frames between successive clips
_M = {}


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def _model():
    if "m" not in _M:
        import torch
        from transformers import AutoModel, AutoVideoProcessor
        _M["proc"] = AutoVideoProcessor.from_pretrained(MID)
        _M["m"] = AutoModel.from_pretrained(MID, dtype=torch.float16) \
            .to("mps").eval()
        _M["torch"] = torch
    return _M["m"], _M["proc"], _M["torch"]


def clip_surprise(clip):
    """(NFRAME,H,W,3) -> per-temporal-group predictor residual.

    For each group k: context = groups [0,k), target = group k.
    Residual = 1 - cos(predicted, actual) averaged over the group's
    spatial tokens. Returns (K-1,) for k = 1..K-1.
    """
    model, proc, torch = _model()
    inp = proc([list(clip)], return_tensors="pt")
    pv = inp["pixel_values_videos"].to("mps", torch.float16)
    with torch.no_grad():
        enc = model(pixel_values_videos=pv,
                    skip_predictor=True).last_hidden_state
    n = enc.shape[1]
    ps = model.config.patch_size
    sp = (pv.shape[-1] // ps) ** 2
    K = n // sp
    out = []
    for k in range(1, K):
        ctx = torch.arange(0, k * sp, device=pv.device)[None]
        tgt = torch.arange(k * sp, (k + 1) * sp,
                           device=pv.device)[None]
        with torch.no_grad():
            o = model(pixel_values_videos=pv, context_mask=[ctx],
                      target_mask=[tgt])
        pred = o.predictor_output.last_hidden_state[0].float()
        act = enc[0, k * sp:(k + 1) * sp].float()
        if pred.shape[0] != act.shape[0]:
            pred = pred[-act.shape[0]:]
        pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        an = act / (act.norm(dim=1, keepdim=True) + 1e-8)
        out.append(float(1.0 - (pn * an).sum(1).mean()))
    return np.array(out)


def media_surprise(F, dur, fps):
    """Slide clips across the media; average overlapping estimates.
    Returns (times, signal)."""
    n = len(F)
    acc, cnt = {}, {}
    starts = list(range(0, max(n - NFRAME, 0) + 1, CLIP_STRIDE))
    if not starts:
        starts = [0]
    for s in starts:
        idx = np.linspace(s, min(s + NFRAME - 1, n - 1),
                          NFRAME).round().astype(int)
        r = clip_surprise([F[i] for i in idx])
        K = len(r) + 1
        for k, v in enumerate(r, start=1):
            # group k covers frames [s + k*NFRAME/K, ...)
            t = (s + k * NFRAME / K) / fps
            key = round(t, 2)
            acc[key] = acc.get(key, 0.0) + v
            cnt[key] = cnt.get(key, 0) + 1
    times = np.array(sorted(acc))
    sig = np.array([acc[t] / cnt[t] for t in times])
    return times, sig


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return float((r[y == 1].sum() - len(pos) * (len(pos) - 1) / 2)
                 / max(len(pos) * len(neg), 1))


def grade():
    import encode as E
    import pyarrow.parquet as pq
    import oxford as OX
    TOL = arg("--tol", 1.0, float)
    NEPS = arg("--eps", 10, int)
    FPS = 4.0

    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet") \
        .to_pydict()
    bnd = {}
    for e, a, b in zip(t["episode"], t["t0"], t["t1"]):
        bnd.setdefault(int(e), set()).update(
            [round(float(a), 2), round(float(b), 2)])

    jobs = []
    for d in sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                    if p.is_dir() and p.name.startswith("ep"))[:NEPS]:
        jobs.append(("sim", sorted(d.glob("cam*.mp4"))[0],
                     sorted(bnd.get(int(d.name[2:]), []))))
    bs_ox, rel, sp_, yr = OX.truth_boundaries()
    jobs.append(("oxford", OX.MP4, bs_ox))

    per = {}
    for dom, path, bs in jobs:
        dur = E.probe_duration(path)
        F = E.decode(path, fps=FPS)
        times, sig = media_surprise(F, dur, FPS)
        near = np.zeros(len(times), bool)
        far = np.ones(len(times), bool)
        for b in bs:
            near |= np.abs(times - b) <= TOL
            far &= np.abs(times - b) > 2 * TOL
        if near.sum() < 2 or far.sum() < 2:
            continue
        per.setdefault(dom, []).append(auc(sig[near], sig[far]))
        print(f"  {dom:<7} {path.parent.name if dom=='sim' else 'drive':<8}"
              f" AUC {per[dom][-1]:.3f}", flush=True)
    print()
    for dom, v in per.items():
        print(f"STEP 4 (shipped predictor) {dom}: per-media AUC "
              f"{np.mean(v):.3f} +- {np.std(v):.2f}  (n={len(v)})")


if __name__ == "__main__":
    if "--grade" in sys.argv:
        grade()
