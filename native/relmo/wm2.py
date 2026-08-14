"""RelMo-WM v2 — a world model over 3D point tracks.

Implements native/WM_PLAN.md. The three things that make this a world
model rather than a trajectory regressor:

  1. RECURRENT STATE. z_{t+1} = f(z_t) is applied step by step and
     rolled forward under its OWN predictions - no observations are
     fed in during the horizon - so error compounds the way physics
     does and the latent has to carry state.
  2. FACTORED LIKE THE WORLD. Points are bound into rigid PARTS,
     parts are connected by JOINTS, joints have a screw axis. A part
     moves by ONE SE(3) transform, so a rigid link CANNOT deform: the
     constraint is architectural, not learned and hoped for.
  3. LATENT ACTION. An inverse model infers the low-dimensional cause
     of each transition. Real video never has action labels, and the
     inferred action sequence is what "the same kind of thing
     happened" means for retrieval.

Everything is expressed in per-episode canonical units (scene scale,
frame time), which is what lets a 1.2 m arm, a 0.3 m arm and a hand
share one model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- utils
def canon(X, V, tc):
    """3D tracks -> canonical frame. Uses ONLY the context window, so
    no future information leaks into the input."""
    ctx = X[:, :tc]
    m = V[:, :tc, :, None].float()
    c = (ctx * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)     # (B,3)
    d = ctx - c[:, None, None]
    s = (d.pow(2).sum(-1).sqrt() * m[..., 0]).sum((1, 2))
    s = (s / m[..., 0].sum((1, 2)).clamp(min=1)).clamp(min=1e-3)
    return (X - c[:, None, None]) / s[:, None, None, None], c, s


def kabsch(A, B, w):
    """Best rigid transform mapping A->B under weights w.

    Closed form and differentiable. The RESIDUAL of this fit is the
    part-discovery signal: points that belong to one rigid link admit
    one transform with near-zero residual, points that do not cannot.
    A: (B,P,3) B: (B,P,3) w: (B,P)
    """
    w = w.clamp(min=1e-6)
    wn = w / w.sum(-1, keepdim=True)
    ca = (A * wn[..., None]).sum(1, keepdim=True)
    cb = (B * wn[..., None]).sum(1, keepdim=True)
    H = ((A - ca) * wn[..., None]).transpose(1, 2) @ (B - cb)
    U, _, Vh = torch.linalg.svd(H.double())
    dt = torch.linalg.det(Vh.transpose(1, 2) @ U.transpose(1, 2))
    E = torch.eye(3, device=A.device, dtype=torch.float64)
    E = E.expand(len(A), 3, 3).clone()
    E[:, 2, 2] = dt
    Rm = (Vh.transpose(1, 2) @ E @ U.transpose(1, 2)).float()
    t = cb.squeeze(1) - (Rm @ ca.squeeze(1)[..., None]).squeeze(-1)
    return Rm, t


def screw(R_rel):
    """Rotation angle + axis of a relative rotation.

    Joint type falls out of how these behave over time:
      angle ~ 0 and axis unstable        -> RIGID / PRISMATIC
      angle varies, axis CONSTANT        -> REVOLUTE
    so the classifier is a statistic over the episode, not a label.
    """
    tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    ang = ((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6).acos()
    ax = torch.stack([R_rel[..., 2, 1] - R_rel[..., 1, 2],
                      R_rel[..., 0, 2] - R_rel[..., 2, 0],
                      R_rel[..., 1, 0] - R_rel[..., 0, 1]], -1)
    ax = ax / ax.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return ang, ax


# --------------------------------------------------------------- blocks
class SlotBind(nn.Module):
    """Points -> parts by competition (Locatello slot attention).
    Softmax is over SLOTS, so slots must divide the points."""

    def __init__(self, d=128, k=8, iters=3):
        super().__init__()
        self.k, self.iters = k, iters
        self.mu = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.ls = nn.Parameter(torch.zeros(1, 1, d))
        self.nq, self.ni, self.ns = (nn.LayerNorm(d), nn.LayerNorm(d),
                                     nn.LayerNorm(d))
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
        s = self.mu + self.ls.exp() * torch.randn(B, self.k, D,
                                                  device=x.device)
        attn = None
        for _ in range(self.iters):
            q = self.q(self.nq(s)) * D ** -0.5
            lg = torch.einsum("bkd,bpd->bkp", q, k)
            if mask is not None:
                lg = lg.masked_fill(~mask[:, None, :], -1e4)
            attn = lg.softmax(1)
            w = attn / attn.sum(-1, keepdim=True).clamp(min=1e-6)
            s = self.gru(torch.einsum("bkp,bpd->bkd", w, v).reshape(-1, D),
                         s.reshape(-1, D)).view(B, self.k, D)
            s = s + self.mlp(self.ns(s))
        return s, attn


class STBlock(nn.Module):
    """Factorised space-time: over TIME per point, then over POINTS."""

    def __init__(self, d, h):
        super().__init__()
        self.ta = nn.MultiheadAttention(d, h, batch_first=True)
        self.pa = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d), nn.LayerNorm(d),
                                     nn.LayerNorm(d))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, P, D = x.shape
        h = self.n1(x).permute(0, 2, 1, 3).reshape(B * P, T, D)
        x = x + self.ta(h, h, h, need_weights=False)[0].view(
            B, P, T, D).permute(0, 2, 1, 3)
        h = self.n2(x).reshape(B * T, P, D)
        x = x + self.pa(h, h, h, need_weights=False)[0].view(B, T, P, D)
        return x + self.ff(self.n3(x))


class FDNNHead(nn.Module):
    """KAN-style heterogeneous oscillatory bases (FDNN rule 1).

    Placed on the trajectory decoder because its sub-functions ARE a
    catalogue of this physics: poly-chirp = acceleration, FINER =
    rolling/bouncing, Gabor = contact as a burst. Basis type and omega
    are BUFFERS, so no optimiser can drift a categorical selector.
    Deliberately NOT inside the recurrence - measured in this project
    to give chaotic gradients there."""

    def __init__(self, d_in, d_out, k=4, omegas=(0.8, 2.5, 8.0), seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        C = d_out
        self.C, self.k = C, k
        om = torch.tensor([omegas[i % len(omegas)] for i in range(C)]).float()
        self.register_buffer("omega", om.repeat_interleave(k))
        bt = torch.tensor([0] * max(k // 2, 1) + [1] * max(k // 4, 1)
                          + [3] * max(k - k // 2 - k // 4, 0))[:k]
        self.register_buffer("btype", bt.repeat(C))
        self.lin = nn.Linear(d_in, C * k)
        self.phase = nn.Parameter(torch.rand(C * k, generator=g) * 6.2832)
        self.gs = nn.Parameter(0.3 + 1.2 * torch.rand(C * k, generator=g))
        self.log_a = nn.Parameter(torch.log(self.omega.clone().clamp(min=1e-3)))
        self.w2 = nn.Parameter((torch.rand(C, k, generator=g) - 0.5)
                               * (6.0 / (C * k)) ** 0.5)

    def forward(self, x):
        pre = self.lin(x)
        om, sq = self.omega * pre, pre * pre
        a = self.log_a.exp()
        finer = torch.sin(self.omega * (pre.abs() + 1.0) * pre + self.phase)
        gab = torch.exp(-(self.gs ** 2) * sq) * torch.sin(om + self.phase)
        sine = torch.sin(om + self.phase)
        poly = torch.sin(a * sq + om + self.phase)
        acts = torch.where(self.btype == 0, finer,
                           torch.where(self.btype == 1, gab,
                                       torch.where(self.btype == 2, sine,
                                                   poly)))
        return (acts.view(*x.shape[:-1], self.C, self.k) * self.w2).sum(-1)


# ---------------------------------------------------------------- model
class RelMoWM2(nn.Module):
    def __init__(self, d=128, blocks=4, heads=4, slots=8, adim=8,
                 head="fdnn", readout="mean"):
        super().__init__()
        self.d, self.K, self.adim = d, slots, adim
        # HOW THE CONTEXT IS READ OUT INTO SLOTS. "mean" pools the whole
        # 16-frame context, which is what run relmowm2_fdnn used - and it
        # throws away the CURRENT state. Measured consequences at step
        # 15500: G2 part ARI 0.235 against an 0.857 ceiling for the same
        # slot count, and a G4 R2 curve that is nearly FLAT in horizon
        # (0.189 at h=1 -> 0.070 at h=8) while const-velocity decays
        # 0.957 -> 0.462. A predictor whose error barely depends on how
        # far ahead it looks is emitting a constant mean motion; bodies
        # are separated by instantaneous velocity differences, and a
        # time-mean erases both. "last+mean" keeps the pooled context AND
        # the last frame. Default stays "mean" so run A's checkpoints
        # still load for comparison.
        self.readout = readout
        self.rd = nn.Linear(2 * d, d) if readout == "last+mean" else None
        self.inp = nn.Linear(7, d)              # xyz, dxyz, vis
        self.tpe = nn.Parameter(torch.randn(1, 256, 1, d) * 0.02)
        self.blocks = nn.ModuleList([STBlock(d, heads) for _ in range(blocks)])
        self.slot = SlotBind(d, slots)
        inter = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                           norm_first=True, activation="gelu",
                                           dropout=0.0)
        self.interact = nn.TransformerEncoder(inter, 2)
        # recurrent transition over the PART set, conditioned on action
        self.trans = nn.GRUCell(d + adim, d)
        self.inv = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                 nn.Linear(d, adim))       # latent action
        dec = (nn.Sequential(nn.Linear(d, d), FDNNHead(d, d),
                             nn.Linear(d, 6))
               if head == "fdnn" else
               nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6)))
        self.dec = dec                                     # se(3) delta
        self.lift = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, 1))         # depth head
        self.aot = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                 nn.Linear(d, 1))
        self.jnt = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                 nn.Linear(d, 4))          # rigid/rev/pri/free

    def encode(self, X, V):
        dx = torch.diff(X, dim=1, prepend=X[:, :1])
        f = torch.cat([X, dx, V[..., None].float()], -1)
        h = self.inp(f) + self.tpe[:, :X.shape[1]]
        for b in self.blocks:
            h = b(h)
        return h

    def parts(self, h, V):
        pt = (self.rd(torch.cat([h[:, -1], h.mean(1)], -1))
              if self.rd is not None else h.mean(1))
        m = V.float().mean(1) > 0.3
        s, attn = self.slot(pt, m)
        return self.interact(s), attn

    def rollout(self, X, V, tc, H):
        """Encode the context, then step the latent forward H times
        WITHOUT looking at the future. Each step emits a per-part
        se(3) delta which is applied to that part's points."""
        h = self.encode(X[:, :tc], V[:, :tc])
        slots, attn = self.parts(h, V[:, :tc])
        B, K, D = slots.shape
        z = slots.reshape(B * K, D)
        cur = X[:, tc - 1]                               # (B,P,3)
        outs, acts = [], []
        for _ in range(H):
            a = self.inv(torch.cat([z, z.detach()], -1))  # inferred cause
            z = self.trans(torch.cat([z, a], -1), z)
            se3 = self.dec(z).view(B, K, 6)
            w, t = se3[..., :3], se3[..., 3:]
            # Apply each slot's rigid motion to ALL points, then blend
            # the resulting POSITIONS by slot assignment.
            #
            # The obvious shortcut - blend the axes and angles first,
            # then rotate once - is WRONG: a weighted sum of unit axes
            # is not unit-norm, so Rodrigues receives the wrong
            # rotation magnitude and the "rigid" transform is not
            # rigid. Blending positions is a convex combination of
            # valid rigid motions, which is safe and differentiable,
            # and as attention sharpens it converges to exactly one
            # rigid transform per part.
            wsum = attn.sum(-1, keepdim=True).clamp(min=1e-6)   # (B,K,1)
            cen = torch.einsum("bkp,bpd->bkd", attn, cur) / wsum
            pc = cur[:, None] - cen[:, :, None]              # (B,K,P,3)
            # Rodrigues in SINC form - no axis normalisation, no clamp.
            #
            # HONEST NOTE ON WHY THIS IS HERE. It replaced axis = w/||w||
            # on the theory that the 1/||w|| term was strangling gradients,
            # because the trained head emits ||w|| with median 1.2e-3 rad
            # and max 5.5e-3. That theory was then MEASURED AND REFUTED:
            # grad-norm through both forms is identical to the printed
            # digit (29.795 vs 29.795 at ||w||~1e-3), since the old clamp
            # at 1e-6 sits three decades below anything the head emits.
            # Kept only because it is verifiably the same map (max abs
            # difference 4.441e-16, rigidity 4.441e-16) while being finite
            # at w=0 without a magic constant. It is NOT the fix for the
            # head's collapse to near-identity, which remains open.
            th2 = (w * w).sum(-1, keepdim=True)[:, :, None]  # (B,K,1,1)
            th = th2.clamp(min=1e-12).sqrt()
            sml = th2 < 1e-8
            sinc = torch.where(sml, 1.0 - th2 / 6.0, torch.sin(th) / th)
            cosc = torch.where(sml, 0.5 - th2 / 24.0,
                               (1.0 - torch.cos(th)) / th2.clamp(min=1e-12))
            ww = w[:, :, None].expand_as(pc)                 # (B,K,P,3)
            wxp = torch.cross(ww, pc, dim=-1)
            rot = pc + sinc * wxp + cosc * torch.cross(ww, wxp, dim=-1)
            moved = rot + cen[:, :, None] + t[:, :, None]    # (B,K,P,3)
            cur = torch.einsum("bkp,bkpd->bpd", attn, moved)
            outs.append(cur)
            acts.append(a.view(B, K, -1))
        return dict(pred=torch.stack(outs, 1), slots=slots, attn=attn,
                    feat=h, actions=torch.stack(acts, 1))

    def arrow(self, h):
        return self.aot(h.mean((1, 2))).squeeze(-1)
