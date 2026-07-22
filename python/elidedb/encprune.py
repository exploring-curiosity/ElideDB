"""FDNN cellular turnover applied to the SigLIP ENCODER — where the time is.

WHY HERE AND NOT THE LITTLE TOWER
---------------------------------
Ingest cost is measured, and it is not the database: byte-range decode runs at
2.7 ms/frame while the SigLIP vision tower runs at 27.7-90.3 ms/frame. So 97%
of ingest is one frozen 428M-parameter ViT. Pruning the 80k-parameter temporal
tower — which is what was done first — optimises 0.001% of the bill. The
encoder is the neuron population that matters.

THE NEURON, HERE
----------------
Each encoder layer is `fc1 (1152 -> 4304) -> gelu -> fc2 (4304 -> 1152)`.
Hidden unit *c* is a neuron in exactly FDNN's sense: it takes the residual
stream, applies its own nonlinearity, and writes back through its own column
of `fc2`. There are 27 x 4304 = 116,208 of them and they are 65% of the
tower's parameters.

Silencing one is `fc1.weight[c] = 0, fc1.bias[c] = 0` — gelu(0) = 0, so the
channel contributes nothing through fc2. Removing one for real is slicing
`fc1.weight[keep]`, `fc1.bias[keep]`, `fc2.weight[:, keep]`, which is a
genuine FLOP reduction, not a multiply by zero.

UTILIZATION IS DEFINED BY THE DATABASE'S WORKLOAD
-------------------------------------------------
FDNN measures utilization as the increase in validation loss when a neuron is
silenced. The equivalent here is NOT ImageNet accuracy — this encoder exists
to produce vectors that a retrieval index ranks with. So utilization is the
loss of EMBEDDING FIDELITY on frames from the actual corpus:

    fidelity = mean cosine( pruned_embedding, unpruned_embedding )

i.e. the unpruned encoder is its own teacher and the calibration set is the
user's own data. A channel that matters for photographs of dogs but never
fires on a robot arm in a toy kitchen is, for this database, dead weight.

Ablating 116,208 channels one at a time is not affordable (one forward pass
each). FDNN's own PPO feature vector already contains the cheap surrogates —
activation magnitude and downstream weight norm — and their product is the
standard structured-pruning saliency:

    saliency(c) = E_frames |act_c| * || fc2[:, c] ||

which is exactly "how much signal this neuron actually injects". Ablation is
still used, but per LAYER (27 measurements, affordable), to calibrate how much
each layer's saliency scale is worth. PPO then chooses keep-probabilities from
the same five features FDNN used, and the reward is the same shape:

    reward = -(fidelity loss ratio) - lambda * kept_fraction
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np


# ===========================================================================
# 0. Reaching the layers
# ===========================================================================
def vision_layers(model):
    """The encoder layer list of a SigLIP vision tower."""
    vm = model.vision_model
    vm = getattr(vm, "vision_model", vm)
    return list(vm.encoder.layers)


def mlp_width(layer):
    return np.array(layer.mlp.fc1.bias).shape[0]


class _Tap:
    """Wraps an nn.Linear so the mean |output| per channel is recorded.

    MLX calls `self.fc1(x)` by attribute, so swapping the attribute is enough
    to observe it — no hooks, no forked forward pass that could drift from the
    real one.
    """

    def __init__(self, inner):
        self.inner = inner
        self.acc = None
        self.n = 0

    def __call__(self, x):
        out = self.inner(x)
        a = mx.mean(mx.abs(out).reshape(-1, out.shape[-1]), axis=0)
        self.acc = a if self.acc is None else self.acc + a
        self.n += 1
        return out


# ===========================================================================
# 1. Saliency: activation magnitude x downstream weight norm
# ===========================================================================
def channel_saliency(model, images, model_id, batch=16):
    """Per-layer array of per-channel saliency, measured on REAL frames."""
    from .embeddings import _embed_images
    layers = vision_layers(model)
    taps = []
    for lyr in layers:
        t = _Tap(lyr.mlp.fc1)
        lyr.mlp.fc1 = t
        taps.append(t)
    try:
        for i in range(0, len(images), batch):
            _embed_images(images[i:i + batch], model_id)
    finally:
        for lyr, t in zip(layers, taps):
            lyr.mlp.fc1 = t.inner

    out = []
    for lyr, t in zip(layers, taps):
        act = np.array(t.acc) / max(t.n, 1)
        down = np.linalg.norm(np.array(lyr.mlp.fc2.weight), axis=0)
        out.append({"act": act.astype(np.float32),
                    "down": down.astype(np.float32),
                    "saliency": (act * down).astype(np.float32)})
    return out


# ===========================================================================
# 2. Masking and compaction
# ===========================================================================
def _snapshot(layers):
    return [(np.array(l.mlp.fc1.weight), np.array(l.mlp.fc1.bias))
            for l in layers]


def _restore(layers, snap):
    for l, (w, b) in zip(layers, snap):
        l.mlp.fc1.weight = mx.array(w)
        l.mlp.fc1.bias = mx.array(b)


def apply_masks(layers, snap, masks):
    """Silence channels: zeroing the fc1 row makes gelu(0)=0 downstream."""
    for l, (w, b), m in zip(layers, snap, masks):
        keep = m.astype(np.float32)[:, None]
        l.mlp.fc1.weight = mx.array(w * keep)
        l.mlp.fc1.bias = mx.array(b * keep[:, 0])


def compact_mlps(model, masks):
    """Physically delete the dead channels. This is where speed comes from."""
    layers = vision_layers(model)
    removed = 0
    for l, m in zip(layers, masks):
        keep = np.where(m > 0.5)[0]
        if len(keep) == mlp_width(l):
            continue
        removed += mlp_width(l) - len(keep)
        l.mlp.fc1.weight = mx.array(np.array(l.mlp.fc1.weight)[keep])
        l.mlp.fc1.bias = mx.array(np.array(l.mlp.fc1.bias)[keep])
        l.mlp.fc2.weight = mx.array(np.array(l.mlp.fc2.weight)[:, keep])
    mx.eval(model.parameters())
    return removed


# ===========================================================================
# 3. Fidelity against the unpruned teacher, on the user's own frames
# ===========================================================================
def fidelity(model_id, images, teacher, batch=16):
    from .embeddings import _embed_images
    out = []
    for i in range(0, len(images), batch):
        out.append(_embed_images(images[i:i + batch], model_id))
    v = np.concatenate(out, axis=0)
    return float((v * teacher).sum(axis=1).mean())


# ===========================================================================
# 4. Reverse attention (unchanged in spirit from FDNN)
# ===========================================================================
def reverse_attention(importance, temperature=1.0):
    imp = np.asarray(importance, dtype=np.float64)
    imp = (imp - imp.mean()) / (imp.std() + 1e-8)
    logits = -imp / max(temperature, 1e-6)
    logits -= logits.max()
    e = np.exp(logits)
    return (e / (e.sum() + 1e-12)).astype(np.float32)


# ===========================================================================
# 5. The cycle
# ===========================================================================
def prune_encoder(model_id, images, keep=0.5, layer_probe=True, iters=12,
                  lam=0.35, batch=16, verbose=True, seed=0):
    """Prune MLP channels of the vision tower against corpus fidelity.

    `keep` is the global target fraction of MLP channels to retain. The
    per-layer budget is not uniform: layers whose ablation barely moves
    fidelity give up more channels than layers that matter, which is the whole
    point of measuring instead of assuming.

    Returns (model, report). The returned model is COMPACTED — smaller
    matmuls, not masked ones.
    """
    from .embeddings import _embed_images, _load_model
    rng = np.random.default_rng(seed)
    model, _ = _load_model(model_id)
    layers = vision_layers(model)
    widths = [mlp_width(l) for l in layers]
    snap = _snapshot(layers)

    teacher = np.concatenate(
        [_embed_images(images[i:i + batch], model_id)
         for i in range(0, len(images), batch)], axis=0)

    t0 = time.time()
    sal = channel_saliency(model, images, model_id, batch=batch)
    if verbose:
        print(f"  saliency over {len(images)} corpus frames "
              f"({time.time() - t0:.1f}s)", flush=True)

    # ---- per-layer ablation: how much does this layer matter at all? -------
    layer_cost = np.ones(len(layers), dtype=np.float64)
    if layer_probe:
        for li in range(len(layers)):
            masks = [np.ones(w, np.float32) for w in widths]
            masks[li][:] = 0.0
            apply_masks(layers, snap, masks)
            layer_cost[li] = max(1.0 - fidelity(model_id, images[:batch],
                                                teacher[:batch], batch), 1e-6)
        _restore(layers, snap)
        if verbose:
            order = np.argsort(layer_cost)
            print(f"  layer ablation: cheapest {order[:4].tolist()} "
                  f"costliest {order[-4:].tolist()}", flush=True)

    # ---- allocate the budget across layers by measured importance ---------
    # A layer that costs little when removed entirely can afford to lose more
    # of its channels. Normalised so the global kept fraction hits `keep`.
    w = layer_cost / layer_cost.sum()
    share = w / w.mean()                       # 1.0 == average importance
    per_layer_keep = np.clip(keep * share, 0.05, 1.0)
    total = sum(widths)
    scale = (keep * total) / sum(k * n for k, n in zip(per_layer_keep, widths))
    per_layer_keep = np.clip(per_layer_keep * scale, 0.05, 1.0)

    masks = []
    for li, (s, wdt) in enumerate(zip(sal, widths)):
        n_keep = max(int(round(per_layer_keep[li] * wdt)), 1)
        idx = np.argsort(-s["saliency"])[:n_keep]
        m = np.zeros(wdt, np.float32)
        m[idx] = 1.0
        masks.append(m)

    apply_masks(layers, snap, masks)
    fid = fidelity(model_id, images, teacher, batch)
    kept = sum(m.sum() for m in masks) / total
    if verbose:
        print(f"  saliency prune: kept {kept:.1%}, fidelity {fid:.4f}",
              flush=True)

    # ---- PPO refinement over the per-layer budget --------------------------
    # The candidate set is the 27 per-layer keep fractions rather than 116,208
    # independent channels: one forward pass per sampled mask makes per-channel
    # sampling unaffordable, and the per-layer budget is where the leverage
    # actually is (saliency already orders channels within a layer).
    best = (fid, [m.copy() for m in masks], kept)
    ra = reverse_attention(layer_cost)
    logit = np.zeros(len(layers))
    for it in range(iters):
        cand = per_layer_keep * (1.0 + 0.25 * np.tanh(logit)
                                 + 0.15 * rng.standard_normal(len(layers)))
        cand = np.clip(cand, 0.05, 1.0)
        cand *= (keep * total) / sum(c * n for c, n in zip(cand, widths))
        cand = np.clip(cand, 0.05, 1.0)
        trial = []
        for li, (s, wdt) in enumerate(zip(sal, widths)):
            n_keep = max(int(round(cand[li] * wdt)), 1)
            idx = np.argsort(-s["saliency"])[:n_keep]
            m = np.zeros(wdt, np.float32)
            m[idx] = 1.0
            trial.append(m)
        apply_masks(layers, snap, trial)
        f = fidelity(model_id, images[:batch * 2], teacher[:batch * 2], batch)
        k = sum(m.sum() for m in trial) / total
        reward = -(1.0 - f) - lam * k
        best_reward = -(1.0 - best[0]) - lam * best[2]
        if reward > best_reward:
            # Direction of the accepted move, computed BEFORE the budget is
            # updated — comparing `cand` against itself would make every sign
            # zero and the search a pure random walk.
            step = np.sign(cand - per_layer_keep)
            best = (f, [m.copy() for m in trial], k)
            per_layer_keep = cand
            logit += 0.5 * ra * step
        if verbose and (it % 4 == 0 or it == iters - 1):
            print(f"  PPO {it:2d} | fidelity {f:.4f} | kept {k:.1%} | "
                  f"reward {reward:+.4f}", flush=True)

    fid, masks, kept = best
    _restore(layers, snap)
    removed = compact_mlps(model, masks)
    final = fidelity(model_id, images, teacher, batch)
    report = {"model": model_id, "kept_fraction": float(kept),
              "channels_removed": int(removed),
              "channels_total": int(total),
              "fidelity": float(final),
              "layer_keep": [float(x) for x in per_layer_keep]}
    if verbose:
        print(f"  compacted: removed {removed:,}/{total:,} MLP channels, "
              f"fidelity {final:.4f}", flush=True)
    return model, report
