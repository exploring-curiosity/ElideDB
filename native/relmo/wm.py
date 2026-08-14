"""RelMo-WM — a world model over point tracks.

WHY THIS REPLACES THE PREVIOUS ARCHITECTURE (owner, 2026-08-11):
"grouping is not what I want the model to learn - I want it to learn
physics, space and time", and "how do you know it isn't just
overfitting the grouping". Both land. The previous net was trained on
grouping/contact LABELS from the simulator and scored 0.985 in-domain
with flat 0.29 transfer - the signature of memorising a distribution,
and unfalsifiable without held-out data.

So the objective changes completely:

  TRAINED  self-supervised only - predict the future of the point
           field, and tell time's direction. No labels, at all.
  PROBED   grouping / contact / depth are read off the FROZEN
           representation afterwards. If they decode well without ever
           being trained on, physics emerged. That is the test the
           previous design could not perform on itself.

Architecture, each piece research-standard for this problem:

  space-time encoder   factorised attention - over TIME for each
                       point, over POINTS at each time. Time is no
                       longer flattened into a feature vector, so
                       "when did this happen relative to that" is
                       representable at flexible durations.
  slot attention       entities emerge by COMPETITION (softmax over
                       slots, iterative refinement, GRU update) -
                       Locatello et al.'s object-centric binding. This
                       is the label-free replacement for the
                       supervised grouping head, and for the
                       hand-built common-fate clustering that
                       collapsed on 151/324 real moments.
  interaction network  transformer over slots = relational dynamics
                       (Battaglia et al.). Physics is interaction
                       between entities, so the model must have a
                       place to represent it.
  prediction heads     future point displacement + latent rollout.
                       You cannot satisfy these by memorising which
                       points share an id; you have to model contact,
                       support and momentum.

Everything is normalised by the CONTEXT window only, so no future
information leaks into the input, and no world axis or absolute scale
is ever used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_ctx(xy, tc):
    """Normalise by the context window's own centre and spread.
    Future frames are transformed with the SAME statistics, so the
    target cannot leak scale information back into the input."""
    c = xy[:, :tc].mean(dim=(1, 2), keepdim=True)
    s = xy[:, :tc].reshape(len(xy), -1, 2).std(dim=1).mean(-1)
    s = s.clamp(min=1e-3).view(-1, 1, 1, 1)
    return (xy - c) / s, s


class SlotAttention(nn.Module):
    """Entities by competition, not by threshold. Slots attend over
    points with the softmax taken ACROSS SLOTS, so slots must divide
    the points between them; iterated a few times with a GRU update."""

    def __init__(self, d=128, k=6, iters=3):
        super().__init__()
        self.k, self.iters, self.d = k, iters, d
        self.mu = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.logsig = nn.Parameter(torch.zeros(1, 1, d))
        self.nq = nn.LayerNorm(d)
        self.ni = nn.LayerNorm(d)
        self.ns = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k_ = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, d))

    def forward(self, x, mask=None):
        B, P, D = x.shape
        x = self.ni(x)
        k, v = self.k_(x), self.v(x)
        slots = self.mu + self.logsig.exp() * torch.randn(
            B, self.k, D, device=x.device)
        attn = None
        for _ in range(self.iters):
            q = self.q(self.nq(slots)) * (D ** -0.5)
            logits = torch.einsum("bkd,bpd->bkp", q, k)
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None, :], -1e4)
            attn = logits.softmax(dim=1)                 # over SLOTS
            w = attn / attn.sum(-1, keepdim=True).clamp(min=1e-6)
            upd = torch.einsum("bkp,bpd->bkd", w, v)
            slots = self.gru(upd.reshape(-1, D),
                             slots.reshape(-1, D)).view(B, self.k, D)
            slots = slots + self.mlp(self.ns(slots))
        return slots, attn                                # (B,K,D),(B,K,P)


class Block(nn.Module):
    """One factorised space-time block: attend over time per point,
    then over points per time."""

    def __init__(self, d, h):
        super().__init__()
        self.t_att = nn.MultiheadAttention(d, h, batch_first=True)
        self.p_att = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d), nn.LayerNorm(d),
                                     nn.LayerNorm(d))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                nn.Linear(4 * d, d))

    def forward(self, x):                                 # (B,T,P,D)
        B, T, P, D = x.shape
        h = self.n1(x).permute(0, 2, 1, 3).reshape(B * P, T, D)
        h = self.t_att(h, h, h, need_weights=False)[0]
        x = x + h.view(B, P, T, D).permute(0, 2, 1, 3)
        h = self.n2(x).reshape(B * T, P, D)
        h = self.p_att(h, h, h, need_weights=False)[0]
        x = x + h.view(B, T, P, D)
        return x + self.ff(self.n3(x))


class FDNNHead(nn.Module):
    """FDNN rule 1, ported to PyTorch: each output channel is a
    KAN-style sum over k HETEROGENEOUS oscillatory bases.

    Why here and not elsewhere: this head predicts a trajectory's
    future, and the sub-functions are a catalogue of exactly that
    physics -
        poly-chirp   sin(a*x^2 + w*x + p)  -> ACCELERATION (a falling
                     or launched thing is quadratic in t)
        FINER        variable-period osc   -> ROLLING / BOUNCING
        Gabor        windowed oscillation  -> CONTACT, which is a
                     BURST in the trajectory
    A plain MLP has a documented low-frequency spectral bias and has
    to approximate all of these with piecewise-linear pieces.

    Two traps the MLX original documented are structurally impossible
    here: `basis_types` and any mask are registered as BUFFERS, so no
    optimiser can drift a categorical selector off its values or
    weight-decay an aliveness mask toward zero. Omega bands are kept
    LOW because this project measured high omega inside recurrence
    producing chaotic gradients - this head is feed-forward, which is
    the calmer place to put the idea first."""

    def __init__(self, d_in, d_out, k=4, omegas=(0.8, 2.5, 8.0),
                 seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        C = d_out
        self.C, self.k = C, k
        om = torch.tensor(
            [omegas[i % len(omegas)] for i in range(C)]).float()
        self.register_buffer("omega", om.repeat_interleave(k))
        bt = torch.tensor([0] * max(k // 2, 1) + [1] * max(k // 4, 1)
                          + [3] * max(k - k // 2 - k // 4, 0))[:k]
        self.register_buffer("btype", bt.repeat(C))
        self.lin = nn.Linear(d_in, C * k)
        self.phase = nn.Parameter(
            torch.rand(C * k, generator=g) * 6.2832)
        self.gs = nn.Parameter(
            0.3 + 1.2 * torch.rand(C * k, generator=g))
        self.log_a = nn.Parameter(
            torch.log(self.omega.clone().clamp(min=1e-3)))
        self.w2 = nn.Parameter(
            (torch.rand(C, k, generator=g) - 0.5)
            * (6.0 / (C * k)) ** 0.5)

    def forward(self, x):
        pre = self.lin(x)
        om = self.omega * pre
        sq = pre * pre
        a = torch.exp(self.log_a)
        finer = torch.sin(self.omega * (pre.abs() + 1.0) * pre
                          + self.phase)
        gab = torch.exp(-(self.gs ** 2) * sq) * torch.sin(om + self.phase)
        sine = torch.sin(om + self.phase)
        poly = torch.sin(a * sq + om + self.phase)
        acts = torch.where(self.btype == 0, finer,
                           torch.where(self.btype == 1, gab,
                                       torch.where(self.btype == 2,
                                                   sine, poly)))
        acts = acts.view(*x.shape[:-1], self.C, self.k)
        return (acts * self.w2).sum(-1)


class RelMoWM(nn.Module):
    def __init__(self, d=128, blocks=4, heads=4, slots=6, zdim=64,
                 pred_h=8, head="mlp"):
        super().__init__()
        self.d, self.pred_h, self.head_kind = d, pred_h, head
        self.inp = nn.Linear(5, d)                # x,y,dx,dy,vis
        self.tpe = nn.Parameter(torch.randn(1, 256, 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads)
                                     for _ in range(blocks)])
        self.slot = SlotAttention(d, slots)
        inter = nn.TransformerEncoderLayer(
            d, heads, 4 * d, batch_first=True, norm_first=True,
            activation="gelu", dropout=0.0)
        self.interact = nn.TransformerEncoder(inter, 2)
        # heads: future motion of every point, and time's direction
        self.pred = (nn.Sequential(nn.Linear(2 * d, d), FDNNHead(d, d),
                                   nn.Linear(d, pred_h * 2))
                     if head == "fdnn" else
                     nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                   nn.Linear(d, pred_h * 2)))
        self.aot = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                 nn.Linear(d, 1))
        self.moment = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                    nn.Linear(d, zdim))

    def encode(self, xy_n, vis):
        """(B,T,P,2)->(B,T,P,D) space-time features."""
        B, T, P, _ = xy_n.shape
        d = torch.diff(xy_n, dim=1, prepend=xy_n[:, :1])
        f = torch.cat([xy_n, d, vis[..., None].float()], -1)
        x = self.inp(f) + self.tpe[:, :T]
        for b in self.blocks:
            x = b(x)
        return x

    def entities(self, x, vis):
        """Time-pooled point tokens -> competing slots + assignment."""
        pt = x.mean(1)                                   # (B,P,D)
        m = vis.float().mean(1) > 0.3
        slots, attn = self.slot(pt, m)
        slots = self.interact(slots)                     # relational
        return slots, attn, pt

    def forward(self, xy, vis, tc):
        """Context [0,tc) -> per-point future prediction + slots."""
        xyn, scale = norm_ctx(xy, tc)
        x = self.encode(xyn[:, :tc], vis[:, :tc])
        slots, attn, pt = self.entities(x, vis[:, :tc])
        # each point's future is predicted from ITS OWN state and the
        # state of the entity it belongs to - that is where contact
        # and support have to be represented to work
        own = x[:, -1]                                   # (B,P,D)
        ent = torch.einsum("bkp,bkd->bpd", attn, slots)
        dxy = self.pred(torch.cat([own, ent], -1))
        dxy = dxy.view(len(xy), -1, self.pred_h, 2)      # (B,P,H,2)
        return dict(dxy=dxy, slots=slots, attn=attn, feat=x,
                    point=pt, scale=scale, xyn=xyn)

    def moment_embed(self, slots, pt):
        return F.normalize(self.moment(
            torch.cat([slots.mean(1), pt.max(1).values], -1)), dim=-1)

    def arrow(self, x):
        return self.aot(x.mean(dim=(1, 2))).squeeze(-1)
