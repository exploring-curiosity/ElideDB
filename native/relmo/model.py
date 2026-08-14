"""RelMo — relational motion encoder.

Reads point trajectories (never pixels, never appearance) and emits
the two things that measurement showed were broken when hand-coded:

  1. GROUPING - which points belong to one physical thing. Hand-built
     common-fate clustering collapsed on 151/324 real moments because
     a carried object and its carrier genuinely move as one; a learned
     grouping can use the whole trajectory shape, not an instantaneous
     velocity agreement.
  2. CONTACT - which things are touching, over time. Image-plane box
     gaps measured BLIND (0.00 median for both touching and separated
     pairs); the simulator's contact solver can teach what contact
     looks like in motion.

No class names, no vocabulary, no task structure anywhere in the
model. Permutation-equivariant over points, so it never learns "the
arm is token 3". Scale/translation normalized per clip, so it carries
no world axis.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NT = 16                  # trajectory samples per point


def prep(xy, vis):
    """(B,T,P,2)/(B,T,P) -> per-point features, per-clip normalized.
    Translation: subtract the clip's own centroid. Scale: divide by
    the clip's own spatial spread. Nothing global, no axis defined."""
    B, T, P, _ = xy.shape
    if T != NT:
        idx = torch.linspace(0, T - 1, NT, device=xy.device).long()
        xy, vis = xy[:, idx], vis[:, idx]
    c = xy.mean(dim=(1, 2), keepdim=True)
    s = xy.reshape(B, -1, 2).std(dim=1).mean(-1)
    s = s.clamp(min=1e-3).view(B, 1, 1, 1)
    p = (xy - c) / s                                   # (B,NT,P,2)
    v = torch.diff(p, dim=1, prepend=p[:, :1])
    p = p.permute(0, 2, 1, 3).reshape(B, P, NT * 2)
    v = v.permute(0, 2, 1, 3).reshape(B, P, NT * 2)
    vv = vis.permute(0, 2, 1).float()
    return torch.cat([p, v, vv], -1)                   # (B,P,5*NT)


class RelMo(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, zdim=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(5 * NT, d), nn.GELU(),
                                 nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, batch_first=True,
            dropout=0.0, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.group = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                   nn.Linear(d, zdim))
        self.pair = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(),
                                  nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, NT))
        self.moment = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                    nn.Linear(d, zdim))
        self.d = d

    def tokens(self, xy, vis):
        return self.tr(self.inp(prep(xy, vis)))        # (B,P,d)

    def point_emb(self, tok):
        return F.normalize(self.group(tok), dim=-1)

    def group_pool(self, tok, assign):
        """assign: (B,P,G) soft membership -> (B,G,d) group tokens."""
        w = assign / assign.sum(1, keepdim=True).clamp(min=1e-6)
        return torch.einsum("bpd,bpg->bgd", tok, w)

    def contact(self, gtok, i, j):
        """per-frame contact logits for group pair (i,j)."""
        a = gtok.gather(1, i.view(-1, 1, 1).expand(-1, 1, self.d))
        b = gtok.gather(1, j.view(-1, 1, 1).expand(-1, 1, self.d))
        a, b = a.squeeze(1), b.squeeze(1)
        return self.pair(torch.cat([a + b, (a - b).abs(),
                                    a * b], -1))
    def embed(self, tok):
        """moment embedding: order-free summary of the whole set."""
        return F.normalize(self.moment(
            torch.cat([tok.mean(1), tok.max(1).values], -1)), dim=-1)


class RelMo2(RelMo):
    """v2: adds an internal 3D-LIFTING head.

    Contact and support are 3D facts and the image plane cannot
    express them - measured on the real corpus, the box gap between
    two things was 0.00 median for BOTH touching and separated pairs.
    The fix is not to run a pretrained depth model at inference
    (measured: monocular depth carries a rest-on-surface prior that
    erases exactly the lifted state that matters). It is to make the
    network infer latent depth from MOTION - parallax, scale change,
    occlusion order - supervised by the simulator's exact depth at
    training time only. At deployment nothing but tracks is read.

    Depth is supervised in a SCALE-FREE form (per-clip z-scored), so
    the model learns relative arrangement rather than metres, which
    is what a relation needs and what survives a new camera."""

    def __init__(self, d=128, layers=4, heads=4, zdim=64):
        super().__init__(d, layers, heads, zdim)
        self.depth = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                   nn.Linear(d, NT))

    def depth_pred(self, tok):
        return self.depth(tok)                     # (B,P,NT)
