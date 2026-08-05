"""STEP 6 - ENCODE UNITS. The measured bottleneck.

native/sensitivity.py established that boundaries are NOT the
constraint: oracle spans + real encoders retrieve at yield 0.267, while
oracle spans + true LABELS retrieve at 0.992. That whole gap is what a
unit's vector fails to say about what happened inside it. This step is
where it lives.

The ledger's metric was "unit-vector stability across views", which is
necessary but not sufficient - a constant function is perfectly stable
and says nothing. So the grade here is a PAIR:

  discriminability  can the vector tell one kind of event from
                    another? AUC over unit pairs, positive = same
                    label, negative = different. Measured separately
                    for the ACTION (prim: pick/place/stack) and the
                    OBJECT (color), because an encoder can easily have
                    one and not the other - and the retrieval failures
                    in this project have consistently been the action.
  stability         same event, different frame sampling -> cosine.
                    Guards against buying discriminability with noise.

Labels are EVAL-ONLY. Nothing here fits on them; they score encoders
that were chosen and run without them.

Held-out discipline (learned in step 5, where the dev set flattered by
0.084): tune nothing on the 12 dev episodes, confirm on 60 unseen.

    python native/unitenc.py --limit 12            # dev,  ~4 min
    python native/unitenc.py --skip 12 --limit 60  # held out
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg                                       # noqa: E402

FPS = 4.0
NF = 8                      # frames per unit handed to an encoder


def _norm(V):
    V = np.asarray(V, np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)


def frames_of(F, t0, t1, nf=NF, off=0.0):
    """Uniform sample of a span. `off` shifts the sampling phase, which
    is how the stability check makes a genuinely different VIEW of the
    same event rather than re-running identical arithmetic."""
    a, b = t0 + off, t1 + off
    ia = max(0, min(int(round(a * FPS)), len(F) - 1))
    ib = max(ia + 1, min(int(round(b * FPS)), len(F) - 1))
    idx = np.linspace(ia, ib, nf).round().astype(int)
    return [F[min(int(j), len(F) - 1)] for j in idx]


# ------------------------------------------------------------ encoders

def enc_vjepa(F, spans, off=0.0):
    """The CURRENT step-3 encoder: V-JEPA2 ViT-L, tokens mean-pooled."""
    import encode as E
    model, proc, torch = E._model()
    out = []
    for i in range(0, len(spans), 4):
        clips = [frames_of(F, a, b, off=off)
                 for a, b in spans[i:i + 4]]
        pv = proc(clips, return_tensors="pt")["pixel_values_videos"]
        with torch.no_grad():
            o = model(pixel_values_videos=pv.to("mps", torch.float16))
        out.append(o.last_hidden_state.mean(1).float().cpu().numpy())
    return _norm(np.concatenate(out))


def _frame_enc(name):
    from gebdback import BACKBONES, KIND
    kind, mid = BACKBONES[name]
    return KIND[kind], mid


def enc_frames(F, spans, off=0.0, back="dinov3_ct", mode="mean",
               nf=NF):
    """Per-frame encoder pooled over the unit.

    mode=mean   : the plain average - what the unit LOOKS like
    mode=delta  : [mean ; normalised(last - first)] - what CHANGED
                  across the unit. A mean is order-blind: reversing the
                  clip leaves it identical, so it cannot represent a
                  pick as different from a place. The delta term is the
                  cheapest possible fix for that and costs one extra
                  concatenation, no extra model.
    """
    fn, mid = _frame_enc(back)
    flat, idx = [], []
    for a, b in spans:
        fr = frames_of(F, a, b, nf=nf, off=off)
        idx.append((len(flat), len(flat) + len(fr)))
        flat += fr
    V = _norm(fn(mid, flat))
    out = []
    for i0, i1 in idx:
        seg = V[i0:i1]
        m = seg.mean(0)
        if mode == "mean":
            out.append(m)
        elif mode == "delta":
            d = seg[-1] - seg[0]
            d = d / max(np.linalg.norm(d), 1e-8)
            out.append(np.concatenate([m, d]))
        elif mode == "meanstd":
            out.append(np.concatenate([m, seg.std(0)]))
        elif mode == "rank":
            # Approximate rank pooling (Bilen et al., dynamic images;
            # Fernando et al., rank pooling). The rank-1 solution to
            # "find w whose projection increases with t" has the closed
            # form sum_t alpha_t v_t with alpha_t = 2t - T - 1. Fully
            # parameter-free, one pass, and order-aware BY
            # CONSTRUCTION: reverse the clip and it flips sign, which
            # is exactly the property a mean lacks.
            T = len(seg)
            al = (2 * np.arange(1, T + 1) - T - 1).astype(np.float32)
            r = (al[:, None] * seg).sum(0)
            r = r / max(np.linalg.norm(r), 1e-8)
            out.append(np.concatenate([m, r]))
        elif mode == "rankonly":
            # Direction of change ONLY - the mean is dropped. If pick
            # and place differ by the STATE they leave behind rather
            # than by the motion, then the scene-appearance term is
            # dilution, not signal.
            T = len(seg)
            al = (2 * np.arange(1, T + 1) - T - 1).astype(np.float32)
            r = (al[:, None] * seg).sum(0)
            out.append(r / max(np.linalg.norm(r), 1e-8))
        elif mode == "endpoints":
            # Explicit before/after state, no averaging at all.
            out.append(np.concatenate([seg[0], seg[-1]]))
        elif mode == "halves":
            # Crudest possible order preservation: what it looked like
            # first, then what it looked like after. If this alone
            # lifts action AUC, the problem really is just pooling.
            h = max(len(seg) // 2, 1)
            out.append(np.concatenate([seg[:h].mean(0),
                                       seg[h:].mean(0)]))
        else:
            raise ValueError(mode)
    return _norm(np.stack(out))


def enc_vjepa_tgroup(F, spans, off=0.0, groups=4):
    """V-JEPA2 keeping the TEMPORAL axis instead of averaging it away.

    encode.py mean-pools every token, spatial and temporal together,
    which throws away the one thing a video encoder knows that an image
    encoder does not. Here the tokens are split into `groups` temporal
    blocks, averaged within each, and concatenated - order survives.
    """
    import encode as E
    model, proc, torch = E._model()
    out = []
    for i in range(0, len(spans), 4):
        clips = [frames_of(F, a, b, off=off)
                 for a, b in spans[i:i + 4]]
        pv = proc(clips, return_tensors="pt")["pixel_values_videos"]
        with torch.no_grad():
            o = model(pixel_values_videos=pv.to("mps", torch.float16))
        H = o.last_hidden_state.float().cpu().numpy()
        n = H.shape[1]
        sp = max(n // groups, 1)
        for k in range(H.shape[0]):
            segs = [H[k, g * sp:(g + 1) * sp].mean(0)
                    for g in range(groups)]
            out.append(np.concatenate(segs))
    return _norm(np.stack(out))


def enc_ssv2(F, spans, off=0.0, sqrt=True):
    """Meta's SSv2 attentive probe on V-JEPA2 - 174-way ACTION evidence.

    Every encoder tried so far is a generic appearance/feature model
    asked to reveal an action as a side effect, and all of them land
    near chance (best 0.691). This one was TRAINED to name actions:
    Something-Something-v2's 174 classes are motion primitives
    ("Picking something up", "Putting something into something"), and
    the probe is Meta's released checkpoint, not anything fitted here.

    The output is a DISTRIBUTION over action classes, which is also the
    discrete-symbol representation the label oracle showed is worth
    0.942/0.890 against 0.600/0.400 for continuous vectors on the same
    spans. sqrt turns the simplex into Hellinger space so that a cosine
    between two distributions behaves like a proper distribution
    distance rather than being dominated by the top class.
    """
    from elidedb import action_probe as AP
    out = []
    for a, b in spans:
        fr = frames_of(F, a, b, nf=16, off=off)
        pr = AP.clip_action_probs(fr)
        v = np.sqrt(pr) if sqrt else pr
        out.append(np.asarray(v, np.float32))
    return _norm(np.stack(out))


def enc_ssv2_siglip(F, spans, off=0.0):
    """Action evidence AND appearance - they answer different questions
    (what happened vs what it looked like), so concatenating tests
    whether they are complementary rather than redundant."""
    a = enc_ssv2(F, spans, off)
    b = enc_frames(F, spans, off, "siglip2", "rank")
    return _norm(np.concatenate([a, b], axis=1))


def enc_flow(F, spans, off=0.0, nf=16, grid=4):
    """Motion descriptor per unit - the flow stream of a two-stream net.

    Every encoder tried so far reads APPEARANCE and is asked to reveal
    the action as a side effect; the best manages action AUC 0.691.
    Simonyan & Zisserman's two-stream result is that the action lives in
    MOTION, and appearance mostly identifies the scene. Here that is
    dense Farneback flow, pooled on a grid x grid grid into (dx, dy)
    plus magnitude, then rank-pooled over time so that "arm descends
    then rises" is a different vector from "arm rises then descends" -
    which is precisely pick versus place.

    No model, no weights, no training. Just pixels.
    """
    import cv2
    out = []
    for a, b in spans:
        fr = frames_of(F, a, b, nf=nf, off=off)
        g = [cv2.cvtColor(np.ascontiguousarray(x[..., :3]),
                          cv2.COLOR_RGB2GRAY) for x in fr]
        prof = []
        for i in range(len(g) - 1):
            fl = cv2.calcOpticalFlowFarneback(
                g[i], g[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = fl.shape[:2]
            ch, cw = h // grid, w // grid
            cells = []
            for r in range(grid):
                for c in range(grid):
                    blk = fl[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                    dx, dy = blk[..., 0].mean(), blk[..., 1].mean()
                    mg = np.hypot(blk[..., 0], blk[..., 1]).mean()
                    cells += [dx, dy, mg]
            prof.append(cells)
        P = np.asarray(prof, np.float32)
        if len(P) < 2:
            out.append(np.zeros(grid * grid * 3 * 2, np.float32))
            continue
        m = P.mean(0)
        T = len(P)
        al = (2 * np.arange(1, T + 1) - T - 1).astype(np.float32)
        r = (al[:, None] * P).sum(0)
        r = r / max(np.linalg.norm(r), 1e-8)
        out.append(np.concatenate([m / max(np.linalg.norm(m), 1e-8), r]))
    return _norm(np.stack(out))


def enc_flow_siglip(F, spans, off=0.0):
    """Two-stream: motion AND appearance."""
    return _norm(np.concatenate(
        [enc_flow(F, spans, off),
         enc_frames(F, spans, off, "siglip2", "rank")], axis=1))


def enc_flow_r50(F, spans, off=0.0):
    return _norm(np.concatenate(
        [enc_flow(F, spans, off),
         enc_frames(F, spans, off, "resnet50", "rank")], axis=1))


ENCODERS = {
    "sig_rankonly":  lambda F, s, o: enc_frames(F, s, o, "siglip2",
                                                "rankonly"),
    "sig_endpoints": lambda F, s, o: enc_frames(F, s, o, "siglip2",
                                                "endpoints"),
    "sig_rankonly16": lambda F, s, o: enc_frames(F, s, o, "siglip2",
                                                 "rankonly", nf=16),
    "flow":          lambda F, s, o: enc_flow(F, s, o),
    "flow_siglip":   lambda F, s, o: enc_flow_siglip(F, s, o),
    "flow_r50":      lambda F, s, o: enc_flow_r50(F, s, o),
    "ssv2":          lambda F, s, o: enc_ssv2(F, s, o),
    "ssv2_raw":      lambda F, s, o: enc_ssv2(F, s, o, sqrt=False),
    "ssv2_siglip":   lambda F, s, o: enc_ssv2_siglip(F, s, o),
    "vjepa2_tgroup": lambda F, s, o: enc_vjepa_tgroup(F, s, o),
    "dino_rank":     lambda F, s, o: enc_frames(F, s, o, "dinov3_ct",
                                                "rank"),
    "dino_halves":   lambda F, s, o: enc_frames(F, s, o, "dinov3_ct",
                                                "halves"),
    "siglip2_rank":  lambda F, s, o: enc_frames(F, s, o, "siglip2",
                                                "rank"),
    "siglip2_halves": lambda F, s, o: enc_frames(F, s, o, "siglip2",
                                                 "halves"),
    "r50_rank":      lambda F, s, o: enc_frames(F, s, o, "resnet50",
                                                "rank"),
    # Temporal RESOLUTION of the unit. Rank pooling reads the direction
    # of change across a span from nf samples; at nf=8 over a 3 s window
    # that is 2.7 Hz, which may simply be too coarse to see a grasp
    # begin. Costs nothing but frames - no new model.
    "r50_rank16":    lambda F, s, o: enc_frames(F, s, o, "resnet50",
                                                "rank", nf=16),
    "r50_rank24":    lambda F, s, o: enc_frames(F, s, o, "resnet50",
                                                "rank", nf=24),
    "dino_rank16":   lambda F, s, o: enc_frames(F, s, o, "dinov3_ct",
                                                "rank", nf=16),
    "siglip2_rank16": lambda F, s, o: enc_frames(F, s, o, "siglip2",
                                                 "rank", nf=16),
    "vjepa2":        lambda F, s, o: enc_vjepa(F, s, o),
    "dino_mean":     lambda F, s, o: enc_frames(F, s, o, "dinov3_ct",
                                                "mean"),
    "dino_delta":    lambda F, s, o: enc_frames(F, s, o, "dinov3_ct",
                                                "delta"),
    "dino_meanstd":  lambda F, s, o: enc_frames(F, s, o, "dinov3_ct",
                                                "meanstd"),
    "siglip2_mean":  lambda F, s, o: enc_frames(F, s, o, "siglip2",
                                                "mean"),
    "siglip2_delta": lambda F, s, o: enc_frames(F, s, o, "siglip2",
                                                "delta"),
    "r50_delta":     lambda F, s, o: enc_frames(F, s, o, "resnet50",
                                                "delta"),
}


# --------------------------------------------------------------- grade

def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return float((r[y == 1].sum() - len(pos) * (len(pos) - 1) / 2)
                 / (len(pos) * len(neg)))


def pair_auc(V, labels, cross_episode):
    """AUC of same-label vs different-label cosine, pairs taken only
    ACROSS episodes so the score cannot be won by within-episode
    background similarity (same table, same lighting, same camera)."""
    S = V @ V.T
    pos, neg = [], []
    n = len(V)
    for i in range(n):
        for j in range(i + 1, n):
            if cross_episode[i] == cross_episode[j]:
                continue
            (pos if labels[i] == labels[j] else neg).append(S[i, j])
    return auc(np.array(pos), np.array(neg))


def main():
    import encode as E
    import pyarrow.parquet as pq
    from tqdm import tqdm
    skip = arg("--skip", 0, int)
    limit = arg("--limit", 12, int)
    names = arg("--enc", ",".join(ENCODERS)).split(",")

    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    ev = {}
    for e, pr, co, a, b in zip(t["episode"], t["prim"], t["color"],
                               t["t0"], t["t1"]):
        ev.setdefault(int(e), []).append(
            (float(a), float(b), str(pr), str(co)))

    dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                  if p.is_dir() and p.name.startswith("ep"))
    dirs = dirs[skip:skip + limit]
    print(f"STEP 6 unit encoding — {len(dirs)} episodes "
          f"(skip {skip}), {len(names)} encoders", flush=True)

    media = {}
    for d in tqdm(dirs, desc="decode", unit="ep"):
        ei = int(d.name[2:])
        rows = sorted(ev.get(ei, []))
        if not rows:
            continue
        cam = sorted(d.glob("cam*.mp4"))[0]
        media[ei] = (E.decode(cam, fps=FPS, w=256), rows)
    nun = sum(len(r) for _, r in media.values())
    print(f"{len(media)} episodes, {nun} truth units\n", flush=True)

    print(f"{'encoder':<15}{'prim AUC':<11}{'color AUC':<12}"
          f"{'stability':<11}{'dim'}")
    res = []
    for nm in names:
        if nm not in ENCODERS:
            continue
        fn = ENCODERS[nm]
        Vs, prim, col, epi, stab = [], [], [], [], []
        for ei, (F, rows) in media.items():
            spans = [(a, b) for a, b, _, _ in rows]
            V = fn(F, spans, 0.0)
            # a genuinely different view: sampling phase shifted by
            # half a frame interval
            V2 = fn(F, spans, 0.5 / FPS)
            stab += list((V * V2).sum(1))
            Vs.append(V)
            prim += [p for _, _, p, _ in rows]
            col += [c for _, _, _, c in rows]
            epi += [ei] * len(rows)
        V = np.concatenate(Vs)
        pa = pair_auc(V, prim, epi)
        ca = pair_auc(V, col, epi)
        st = float(np.mean(stab))
        res.append((nm, pa, ca, st, V.shape[1]))
        print(f"{nm:<15}{pa:<11.3f}{ca:<12.3f}{st:<11.3f}"
              f"{V.shape[1]}", flush=True)

    b = max(res, key=lambda r: r[1])
    print(f"\nbest by action (prim) AUC: {b[0]}  {b[1]:.3f}")
    print("chance = 0.500; a mean-pooled encoder is ORDER-BLIND, so "
          "prim AUC near chance is the expected failure.")


if __name__ == "__main__":
    main()
