"""QbE over latent trajectories. The world-model read path.

WHAT IS STORED per clip: the 20-row HEIGHT-PROFILE trajectory of
EVERY recorded camera (image rows ~ physical height; the second view
is raw data, and single-view retrieval measured a +0.53 same-camera
bias). WHAT SCORES a window: ordered height-profile change at three
temporal scales (3+5+8 segment deltas), max over the store's views,
with a CSLS hubness correction (generic moments attract everything -
the leak-to-majority mechanism, measured in the confusion matrix).
The ladder, all on the 882-event truthset: pooled appearance 0.416 ->
grid deltas 0.565 -> height-profile 0.612 -> both views 0.676 ->
+CSLS 0.695 -> three scales 0.723 P@10 (yield 0.587 prec 0.391).
Controls at every step in BENCHMARKS. The predictor's states still
ride along; the scorer is whichever wins the sweep.

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
        self.traj = {}          # mid -> list of per-view r20 series
        self.kin = {}           # mid -> list of per-view (cuts, proj)
        pref = "_".join((ROOT / root).resolve()
                        .relative_to(ROOT / "data").parts)
        import re as _re
        import vwm_kin
        self.P = None
        for npz in sorted(CACHE.glob(f"{pref}_ep*_cam*_v3.npz")):
            m = _re.search(r"(ep\d+)_cam", npz.stem)
            mid = f"{corpus}/{m.group(1)}"
            r = np.load(npz)["r20"].astype(np.float32)
            self.traj.setdefault(mid, []).append(r)
            # WRITE-TIME segmentation: each episode is cut ONCE into
            # kinematic phases and a query window inherits the cuts
            # inside it. Measured equal to per-span cutting, and O(1)
            # per candidate window instead of a re-segmentation.
            if self.P is None:
                self.P = vwm_kin.rproj(r.shape[1])
            rp = vwm_kin._l2(r @ self.P)
            self.kin.setdefault(mid, []).append(
                (vwm_kin.dp_cuts(rp, max_units=64), rp))

    @staticmethod
    def dmulti(r, s, e):
        """Ordered height-profile change of window [s,e): segment
        deltas at three temporal scales (3+5+8). Deltas cancel the
        standing scene; the row marginal cancels WHERE on the table."""
        parts = []
        for n in (3, 5, 8):
            cuts = [s + (e - s) * i // n for i in range(n + 1)]
            cuts[-1] = e - 1
            parts += [_l2(r[cuts[i + 1]] - r[cuts[i]])
                      for i in range(n)]
        return np.concatenate(parts)

    def search(self, hq, k=10, exclude=None, query_fps=None):
        """hq: (Tq, D) query states. Returns [(mid, t0, t1, score)].
        query_fps: rate hq was sampled at (vsrc decodes at 4fps while
        the store's trajectories run at the recording's 10) - the
        window is matched in SECONDS, not frames."""
        qf = float(query_fps or FPS)
        secs = max(0.4, len(hq) / qf)
        Lq = max(6, int(round(secs * FPS)))
        import vwm_kin
        v = _l2(self.dmulti(np.asarray(hq, np.float32), 0, len(hq)))
        # query phases, same operator as the store's
        qp = vwm_kin._l2(np.asarray(hq, np.float32) @ self.P)
        qc = vwm_kin.dp_cuts(qp)
        QU = np.stack([vwm_kin.unit_rep(qp, qc[i], qc[i + 1])
                       for i in range(len(qc) - 1)
                       if qc[i + 1] - qc[i] >= vwm_kin.MIN_UNIT]
                      or [vwm_kin.unit_rep(qp, 0, len(qp))])
        QU = _l2(QU)
        stride = max(2, Lq // 4)
        cands, reps = [], []
        for mid, views in self.traj.items():
            T = min(len(r) for r in views)
            if T < Lq:
                continue
            kv = self.kin.get(mid, [])
            for s in range(0, T - Lq + 1, stride):
                a, b = s / FPS, (s + Lq) / FPS
                if exclude and mid == exclude[0] \
                        and not (b <= exclude[1] or a >= exclude[2]):
                    continue
                # max over the store's views of this moment
                rs = [_l2(self.dmulti(r, s, s + Lq)) for r in views]
                sc = max(float(v @ x) for x in rs)
                # ORDERED PHASE ALIGNMENT: the window inherits the
                # episode's cuts; DTW keeps descend->lift from matching
                # lift->descend. Fused 50/50 with the span score - the
                # configuration that strictly dominated span-only on
                # yield, precision AND AP under this exact geometry.
                dt = -1.0
                for cuts, rp in kv:
                    inner = [c for c in cuts
                             if s + vwm_kin.MIN_UNIT <= c
                             <= s + Lq - vwm_kin.MIN_UNIT]
                    seg = [s] + inner + [s + Lq]
                    U = [vwm_kin.unit_rep(rp, seg[i], seg[i + 1])
                         for i in range(len(seg) - 1)
                         if seg[i + 1] - seg[i] >= vwm_kin.MIN_UNIT]
                    if not U:
                        continue
                    dt = max(dt, _dtw(QU, _l2(np.stack(U))))
                if dt > -1.0:
                    sc = 0.5 * sc + 0.5 * dt
                cands.append([mid, a, b, sc])
                reps.append(rs[0])
        # CSLS: a window close to EVERYTHING is close to nothing in
        # particular - the measured leak-to-majority mechanism. Hub =
        # mean top-50 similarity to a 256-window probe sample.
        if len(reps) > 300:
            R = np.stack(reps)
            rs2 = np.random.RandomState(0)
            probe = R[rs2.choice(len(R), 256, replace=False)]
            hub = np.sort(R @ probe.T, 1)[:, -50:].mean(1)
            for c, hb in zip(cands, hub):
                c[3] = 2 * c[3] - float(hb)
        cands = [tuple(c) for c in cands]
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


def _dtw(Q, C):
    """Path-length-normalized DTW similarity between two unit
    sequences (tiny: <=8 x <=8)."""
    M = Q @ C.T
    u, m = M.shape
    D = np.empty((u, m), np.float32)
    D[0, 0] = M[0, 0]
    for j in range(1, m):
        D[0, j] = M[0, j] + D[0, j - 1]
    for i in range(1, u):
        D[i, 0] = M[i, 0] + D[i - 1, 0]
        for j in range(1, m):
            D[i, j] = M[i, j] + max(D[i - 1, j], D[i - 1, j - 1],
                                    D[i, j - 1])
    return float(D[u - 1, m - 1] / (u + m - 1))


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
    """Frames -> 20-row height-profile trajectory, the identical
    write-path operator (v3 cache stores r20 = grid.mean(cols))."""
    G = _encode_grid(F)
    return G.mean(2).reshape(len(G), -1)     # (T, 20*384)


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
