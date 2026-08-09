"""QbE over latent trajectories. The world-model read path.

WHAT IS STORED per clip: the projected-grid latent trajectory (the raw
spatial record) plus the predictor's state trajectory - nothing is
summarised at write time. WHAT SCORES a window: ORDERED SPATIAL CHANGE
- the window's grid-delta thirds, concat of l2(cp[t2]-cp[t1]) over
three consecutive segments. Measured on the 882-event sim truthset
against every alternative (9-way sweep, BENCHMARKS 2026-08-09): 0.565
P@10 vs 0.416 for pooled appearance and 0.36 for every predictor-state
pool - what separates experiences is the SEQUENCE OF CHANGES the world
undergoes, the delta-appearance result at the event scale. The
predictor's states still ride along (surprise, future reps); the
scoring representation is whichever wins the sweep.

    python native/vwm_qbe.py --media sim/ep0003 --t0 4 --t1 12
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

CACHE = ROOT / "data" / "cache" / "vwm"
FPS = 10.0


def _l2(V):
    V = np.asarray(V, np.float32)
    if V.ndim == 1:
        return V / max(float(np.linalg.norm(V)), 1e-8)
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-8)


_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        import vwm
        _MODEL = vwm.load()
    return _MODEL


class WMIndex:
    """State trajectories for one corpus root (default: the sim eval
    corpus), built lazily from the vwm_encode cache + wm_v1."""

    def __init__(self, root="data/sim_chains", corpus="sim"):
        self.corpus = corpus
        self.traj = {}
        # states cache is keyed by the MODEL: a retrained predictor
        # must not read trajectories a previous one wrote
        import hashlib
        import vwm
        tag = hashlib.sha1(vwm.OUT.read_bytes()).hexdigest()[:10]
        sdir = ROOT / "data" / "cache" / f"vwm_states_{tag}"
        sdir.mkdir(parents=True, exist_ok=True)
        pref = "_".join((ROOT / root).resolve()
                        .relative_to(ROOT / "data").parts)
        for npz in sorted(CACHE.glob(f"{pref}_ep*.npz")):
            ep = npz.stem.split("_")[-1]
            mid = f"{corpus}/{ep}"
            sp = sdir / f"{pref}_{ep}.npz"
            if sp.exists():
                z2 = np.load(sp)
                if "cp" in z2:
                    self.traj[mid] = (z2["h"].astype(np.float32),
                                      z2["cp"].astype(np.float32))
                    continue
            import vwm
            z = np.load(npz)
            c = z["c"].astype(np.float32)
            cp = c @ vwm.rproj()
            cp /= np.maximum(
                np.linalg.norm(cp, axis=-1, keepdims=True), 1e-8)
            gi = vwm.build_input(z["g"].astype(np.float32), c)
            h, _ = model().states(gi)
            np.savez(sp, h=h.astype(np.float16),
                     cp=cp.astype(np.float16))
            self.traj[mid] = (h, cp)

    @staticmethod
    def d3rds(cp, s, e):
        """Ordered spatial change of window [s,e): grid-delta thirds.
        Deltas cancel the shared scene, so no centering is needed."""
        t1 = s + (e - s) // 3
        t2 = s + 2 * (e - s) // 3
        return np.concatenate([_l2(cp[t1] - cp[s]),
                               _l2(cp[t2] - cp[t1]),
                               _l2(cp[e - 1] - cp[t2])])

    def search(self, hq, k=10, exclude=None, query_fps=None):
        """hq: (Tq, D) query states. Returns [(mid, t0, t1, score)].
        query_fps: rate hq was sampled at (vsrc decodes at 4fps while
        the store's trajectories run at the recording's 10) - the
        window is matched in SECONDS, not frames."""
        qf = float(query_fps or FPS)
        secs = max(0.4, len(hq) / qf)
        Lq = max(6, int(round(secs * FPS)))
        v = _l2(self.d3rds(np.asarray(hq, np.float32), 0, len(hq)))
        stride = max(2, Lq // 4)
        cands = []
        for mid, (h, cp) in self.traj.items():
            T = len(cp)
            if T < Lq:
                continue
            for s in range(0, T - Lq + 1, stride):
                sc = float(v @ _l2(self.d3rds(cp, s, s + Lq)))
                a, b = s / FPS, (s + Lq) / FPS
                if exclude and mid == exclude[0] \
                        and not (b <= exclude[1] or a >= exclude[2]):
                    continue
                cands.append((mid, a, b, sc))
        cands.sort(key=lambda r: -r[3])
        out = []
        for mid, a, b, sc in cands:      # greedy non-overlap collapse
            if any(m2 == mid and not (b <= a2 or a >= b2)
                   for m2, a2, b2, _ in out):
                continue
            out.append((mid, a, b, sc))
            if len(out) >= k:
                break
        return out


_INDEX = {}


def index(root="data/sim_chains", corpus="sim"):
    key = (root, corpus)
    if key not in _INDEX:
        _INDEX[key] = WMIndex(root, corpus)
    return _INDEX[key]


_VE = {}


def _vits():
    """The WM query encoder, SELF-CONTAINED. The Desk process serves
    the c2 stores with their own encoder (vitl16); reusing vcore's
    global singleton would force one encoder on both paths and break
    whichever the environment did not name. fp32: the DINOv3 ViT fp16
    NaN trap is recorded, not rediscovered."""
    if not _VE:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _VE["proc"] = AutoImageProcessor.from_pretrained(name)
        _VE["m"] = AutoModel.from_pretrained(
            name, dtype=torch.float32,
            low_cpu_mem_usage=True).to(dev).eval()
        _VE["dev"] = dev
        _VE["torch"] = torch
    return _VE


GRID = 20


def _encode_grid(F):
    """(T,H,W,3) -> (T,GRID,GRID,d) with the vits16@320 operator the
    caches were built with (mirrors vtrans.encode)."""
    e = _vits()
    torch = e["torch"]
    out = []
    B = 16
    for i in range(0, len(F), B):
        chunk = [np.ascontiguousarray(x[..., :3]) for x in F[i:i + B]]
        px = e["proc"](images=chunk, return_tensors="pt",
                       size={"height": 320, "width": 320})["pixel_values"]
        with torch.no_grad():
            r = e["m"](pixel_values=px.to(e["dev"]))
        h = r.last_hidden_state
        nreg = int(getattr(e["m"].config, "num_register_tokens", 0) or 0)
        P = h[:, 1 + nreg:].float().cpu().numpy()
        sq = int(round(P.shape[1] ** 0.5))
        P = P[:, :sq * sq].reshape(len(chunk), sq, sq, -1)
        k = max(sq // GRID, 1)
        P = P[:, :k * GRID, :k * GRID]
        P = P.reshape(len(chunk), GRID, k, GRID, k, -1).mean((2, 4))
        out.append(P)
    G = np.concatenate(out)
    # per-cell l2, exactly as vtrans.encode built the caches
    return G / np.maximum(np.linalg.norm(G, axis=-1, keepdims=True), 1e-8)


def query_states(F):
    """Frames -> states, the identical write-path operator."""
    import vwm
    G = _encode_grid(F)
    T = len(G)
    g = _l2(G.reshape(T, -1, G.shape[-1]).mean(1))
    k = G.shape[1] // 5
    c = G.reshape(T, 5, k, 5, k, -1).mean((2, 4)).reshape(T, -1)
    h, sur = model().states(vwm.build_input(g, c))
    return h, sur


def query_grid(F):
    """Frames -> projected-grid trajectory, the write-path operator."""
    import vwm
    G = _encode_grid(F)
    T = len(G)
    k = G.shape[1] // 5
    c = G.reshape(T, 5, k, 5, k, -1).mean((2, 4)).reshape(T, -1)
    cp = c @ vwm.rproj()
    return cp / np.maximum(
        np.linalg.norm(cp, axis=-1, keepdims=True), 1e-8)


def search_frames(F, k=10, exclude=None, query_fps=None):
    """query_fps: the rate F was decoded at (vsrc.FPS for Desk cuts)."""
    if query_fps is None:
        import vsrc
        query_fps = vsrc.FPS
    cp = query_grid(F)
    return index().search(cp, k=k, exclude=exclude, query_fps=query_fps)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    import vsrc
    srcs = {s.id: s for s in vsrc.sources(a.media.split("/")[0])}
    F = srcs[a.media].cut(a.t0, a.t1)
    rows = search_frames(F, k=a.k,
                         exclude=(a.media, a.t0, a.t1))
    for mid, t0, t1, sc in rows:
        print(f"  {mid}  {t0:6.1f}-{t1:6.1f}  {sc:+.4f}")


if __name__ == "__main__":
    main()
