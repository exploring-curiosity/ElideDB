"""FDNN students shaped like their teachers.

A student is not a generic head bolted onto whatever vectors happen to
be lying around. That was tried and it failed for a reason worth
keeping: an attention-pool + ReLU MLP over FDNN-V frame vectors reached
0.9168 cosine to the PE teacher while a CONSTANT prediction of the
corpus mean scored 0.8629, and nearest-neighbour agreement was 0.005.
The head had learned the mean. A least-squares oracle on the same input
reached nn_top1 0.0155, so no head over those vectors could have worked
- the information was not in them.

So the student takes the TEACHER'S OWN ARCHITECTURE as its skeleton,
reads pixels like the teacher does, and shrinks. What changes is the
neuron, per the three FDNN rules:

  RULE 1  each neuron is a sub-network. Every transformer block's MLP -
          the 2/3 of a ViT's parameters that is elementwise GELU -
          becomes a KAN-style sum over k heterogeneous bases of the
          same pre-activation: FINER (variable-period oscillator),
          Gabor (wavelet, fires on a burst), sine, poly-phase (chirp,
          fires on acceleration). omega bands partition the spectrum so
          different channels answer to different rates of change.
          A GELU unit can only answer "how much"; these answer "how
          much, how fast, and is it accelerating" - which is what a
          video model's MLP is being asked for in the first place.

  RULE 2  neurogenesis and apoptosis after training, each followed by a
          re-settle fine-tune. `mask` is the aliveness vector the cycle
          writes; it is frozen against the optimizer, because AdamW
          weight-decays an unfrozen mask off 1.0 and every channel
          quietly shrinks (measured on the context tower).

  RULE 3  PPO + reverse attention decide who lives: utilization is the
          drop in held-out fidelity when a channel is silenced.

WHY THE SKELETON MATTERS. V-JEPA2 is 24 blocks of width 1024 over 8,192
spatiotemporal tokens (256px, 64 frames, patch 16, tubelet 2). Most of
that cost is token count, not depth. A student that keeps the tubelet
embedding and the attentive pooler - the parts that decide WHAT is
compared - while cutting resolution, frames, width and depth, is doing
the teacher's computation at the teacher's shape. A student with a
different shape is a different model that happens to be trained on the
teacher's outputs, which is what failed.
"""
from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn

# Basis identifiers, matching FDNNTemporalCell so the two implementations
# cannot drift apart: 0 FINER, 1 Gabor, 2 sine, 3 poly-phase.
FINER, GABOR, SINE, POLY = 0, 1, 2, 3


class FDNNFeedForward(nn.Module):
    """Rule 1, applied to a transformer block's MLP.

    Standard: y = W2 @ gelu(W1 @ x). Here each of C output channels is a
    KAN sum over k sub-functions of the SAME pre-activation, so one
    channel can be periodic, another can fire only on a burst, another
    only on acceleration - heterogeneous by construction rather than by
    hoping a homogeneous nonlinearity specialises.

    omega bands are spatial-temporal here, not purely temporal: tokens
    are tubelets, so a band is a rate of change across the token grid.
    """

    def __init__(self, dim, channels, k_width=4,
                 omega_bands=(0.8, 2.5, 8.0),
                 band_fractions=(0.34, 0.33, 0.33), seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        C, k = channels, k_width
        self.C, self.k = C, k

        om = []
        for b, fr in zip(omega_bands, band_fractions):
            om.extend([b] * int(round(fr * C)))
        om = (om + [omega_bands[-1]] * C)[:C]
        self.omegas_per_neuron = np.array(om, np.float32)
        self.omegas = mx.array(np.repeat(self.omegas_per_neuron, k))

        half, quarter = max(k // 2, 1), max(k // 4, 1)
        per = np.array([FINER] * half + [GABOR] * quarter
                       + [POLY] * max(k - half - quarter, 0), np.int32)[:k]
        self.basis_types = mx.array(np.tile(per, C).astype(np.int32))
        # categorical, not a weight: unfrozen the optimizer drifts it off
        # its exact values and silently reroutes every Gabor to poly
        self.freeze(keys=["basis_types"], recurse=False)

        mean_om = float(self.omegas_per_neuron.mean())
        lim = float(np.sqrt(6.0 / dim) / mean_om)
        self.W1 = mx.array(rng.uniform(-lim, lim, (dim, C * k)).astype(np.float32))
        self.b1 = mx.array(rng.uniform(-2.0, 2.0, (C * k,)).astype(np.float32))
        self.phases = mx.array(rng.uniform(0, 2 * np.pi, (C * k,)).astype(np.float32))
        self.gabor_s = mx.array(rng.uniform(0.3, 1.5, (C * k,)).astype(np.float32))
        log_om = np.log(np.clip(np.repeat(self.omegas_per_neuron, k), 1e-3, None))
        self.log_alpha = mx.array(rng.uniform(np.minimum(0.0, log_om),
                                              np.maximum(0.0, log_om) + 1e-6
                                              ).astype(np.float32))
        w2s = float(np.sqrt(6.0 / (C * k)))
        self.w2 = mx.array(rng.uniform(-w2s, w2s, (C, k)).astype(np.float32))
        pl = float(np.sqrt(6.0 / C))
        self.Wp = mx.array(rng.uniform(-pl, pl, (C, dim)).astype(np.float32))

        self.mask = mx.array(np.ones((C,), np.float32))     # rule 2
        self.freeze(keys=["mask"], recurse=False)

    def channels_out(self, x):
        """Per-channel activation before the mask - rule 3's signal."""
        pre = x @ self.W1 + self.b1
        om_h = self.omegas * pre
        sq = pre * pre
        alpha = mx.exp(self.log_alpha)
        finer = mx.sin(self.omegas * (mx.abs(pre) + 1.0) * pre + self.phases)
        gab = mx.exp(-(self.gabor_s ** 2) * sq) * mx.sin(om_h + self.phases)
        sine = mx.sin(om_h + self.phases)
        poly = mx.sin(alpha * sq + om_h + self.phases)
        acts = mx.where(self.basis_types == FINER, finer,
                        mx.where(self.basis_types == GABOR, gab,
                                 mx.where(self.basis_types == SINE, sine, poly)))
        acts = acts.reshape(*pre.shape[:-1], self.C, self.k)
        return mx.sum(acts * self.w2, axis=-1)

    def __call__(self, x):
        return (self.channels_out(x) * self.mask) @ self.Wp

    def set_active_mask(self, m):
        self.mask = mx.array(np.asarray(m, np.float32))


class Block(nn.Module):
    """A ViT block with the teacher's shape and an FDNN neuron inside."""

    def __init__(self, dim, heads, channels, k_width=4, seed=0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.ff = FDNNFeedForward(dim, channels, k_width, seed=seed)

    def __call__(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h)
        return x + self.ff(self.n2(x))


class TubeletEmbed(nn.Module):
    """The teacher's input stage: non-overlapping spatiotemporal patches.

    Kept because it is what decides what a token IS. Resolution, frame
    count, width and depth are all shrunk; the tokenisation is not
    changed, so the student is comparing the same kind of thing.
    """

    def __init__(self, dim, patch=16, tubelet=2, in_ch=3):
        super().__init__()
        self.patch, self.tubelet = patch, tubelet
        self.proj = nn.Linear(in_ch * patch * patch * tubelet, dim)

    def __call__(self, v):                      # (B, T, H, W, 3) in [-1,1]
        B, T, H, W, C = v.shape
        p, t = self.patch, self.tubelet
        gt, gh, gw = T // t, H // p, W // p
        v = v[:, :gt * t, :gh * p, :gw * p]
        v = v.reshape(B, gt, t, gh, p, gw, p, C)
        v = v.transpose(0, 1, 3, 5, 2, 4, 6, 7).reshape(B, gt * gh * gw, -1)
        return self.proj(v), (gt, gh, gw)


class AttentivePool(nn.Module):
    """The teacher pools with a learned query, so the student does too.
    Mean pooling buries the few tokens that carry the event under the
    many that carry the unchanged room."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.q = mx.array((np.random.default_rng(0).normal(size=(dim,))
                           / np.sqrt(dim)).astype(np.float32))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)

    def __call__(self, x):
        a = mx.softmax(x @ self.q, axis=1)[..., None]
        y = self.proj(self.norm(mx.sum(a * x, axis=1)))
        return y / (mx.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)


class VideoStudent(nn.Module):
    """V-JEPA2's skeleton at 1/N the size, with FDNN neurons.

    teacher  256px, 64 frames, patch 16, tubelet 2 -> 8192 tokens,
             24 blocks, width 1024, 16 heads
    student  configurable; the default cuts tokens 16x and width 5x,
             which is where a ViT's cost actually lives.
    """

    def __init__(self, out_dim=1024, dim=192, depth=4, heads=3,
                 channels=256, k_width=4, patch=16, tubelet=2,
                 size=128, frames=16, seed=0):
        super().__init__()
        self.size, self.frames = size, frames
        self.embed = TubeletEmbed(dim, patch, tubelet)
        n_tok = (frames // tubelet) * (size // patch) ** 2
        self.pos = mx.array((np.random.default_rng(seed).normal(
            size=(1, n_tok, dim)) * 0.02).astype(np.float32))
        self.blocks = [Block(dim, heads, channels, k_width, seed=seed + i)
                       for i in range(depth)]
        self.pool = AttentivePool(dim, out_dim)

    def __call__(self, v):
        x, _ = self.embed(v)
        x = x + self.pos[:, :x.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.pool(x)

    def utilization(self, v):
        """Rule 3's raw signal: per-channel mean |activation| per block."""
        x, _ = self.embed(v)
        x = x + self.pos[:, :x.shape[1]]
        out = []
        for b in self.blocks:
            h = b.n1(x)
            x = x + b.attn(h, h, h)
            c = b.ff.channels_out(b.n2(x))
            out.append(np.abs(np.array(c)).mean((0, 1)))
            x = x + b.ff(b.n2(x))
        return out
