"""RelMo-WM v3 — per-point flow prediction. The slot bottleneck is gone.

WHY v3 EXISTS. v2 predicted one SE(3) transform PER SLOT and blended the
results by slot attention. Three runs measured it into the ground:

  slot latents pairwise cosine 0.9796  -> all 8 slots decode to nearly the
      SAME transform, so the model was predicting ONE global rigid motion
      for the whole scene and could not express differential motion at all
  direction cos(pred,true) ~0.15, magnitude shrunk ~20x -> shrinking is the
      CORRECT hedge once direction is wrong (verified: scaling outputs by
      alpha made both the loss and R2 monotonically worse)
  G2 PASSED while G4 FELL 0.040 -> 0.016 -> binding lives in the attention,
      the collapse lives in the slot CONTENT, so fixing one did nothing

And the bottleneck's own ceiling was below the baseline it had to beat:
oracle transforms through the model's own attention score 0.826 at h=1
against const-velocity's 0.957. Even TRUE bodies with hard assignment give
0.903. Per-part rigid prediction cannot win at short horizon on this data.

WHAT THE FIELD DOES (2025-2026), and what is taken from each:

  PointWorld (NVlabs 2601.03782) - the closest published match to this
    exact task: scene points in, PER-POINT 3D DISPLACEMENT out, no slots
    and no rigid decomposition; rigidity is learned from data. That is the
    central change here - the prediction ceiling goes 0.62 -> 0.97.
  GNS (Sanchez-Gonzalez ICML'20) - random-walk NOISE INJECTION on inputs
    during training so the model meets its own error distribution, and
    predict motion increments rather than absolute state. v2 had neither.
  Annealed WTA (2409.11172) - deterministic L2 on a multimodal future
    regresses to the mean, which is precisely "predict almost nothing".
    M modes with winner-takes-all keeps the modes apart.
  SlotContrast (CVPR'25) / STAITUS (2606.23436) - slot collapse is a named
    failure with published fixes. Rather than chase them, the slots are
    demoted to an AUXILIARY head: they still earn G2/G3 and the retrieval
    descriptor, but they no longer gate the prediction.

Direct multi-step decode (all H at once) rather than autoregressive
rollout: it removes compounding error as a variable entirely, which is
what v2's flat-in-horizon R2 curve was made of.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from relmo.wm2 import FDNNHead, SlotBind, STBlock, canon, kabsch, screw  # noqa: F401


class RelMoWM3(nn.Module):
    def __init__(self, d=384, blocks=6, heads=6, slots=8, hz=8, modes=4,
                 head="fdnn", pmax=192):
        super().__init__()
        self.d, self.K, self.H, self.M = d, slots, hz, modes
        self.inp = nn.Linear(7, d)                  # xyz, dxyz, vis
        self.tpe = nn.Parameter(torch.randn(1, 256, 1, d) * 0.02)
        self.blocks = nn.ModuleList([STBlock(d, heads) for _ in range(blocks)])
        # per-point readout carries the CURRENT frame beside the pooled
        # context - a pure time-mean was measured to erase instantaneous
        # velocity, which is the one thing short-horizon prediction needs
        self.rd = nn.Linear(2 * d, d)
        pl = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                        norm_first=True, activation="gelu",
                                        dropout=0.0)
        self.interact = nn.TransformerEncoder(pl, 2)   # points talk to points
        # M modes x H steps x 3 - displacement from the last context frame,
        # NOT absolute position: the metric is displacement R2 and the loss
        # should optimise the same quantity it is scored on
        self.dec = (nn.Sequential(nn.Linear(d, d), FDNNHead(d, d),
                                  nn.Linear(d, modes * hz * 3))
                    if head == "fdnn" else
                    nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, modes * hz * 3)))
        self.mode_logit = nn.Linear(d, modes)
        # ---- auxiliary heads. None of these sit on the prediction path.
        self.slot = SlotBind(d, slots)
        self.aot = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.lift = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.jnt = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                 nn.Linear(d, 4))

    def encode(self, X, V):
        dx = torch.diff(X, dim=1, prepend=X[:, :1])
        f = torch.cat([X, dx, V[..., None].float()], -1)
        h = self.inp(f) + self.tpe[:, :X.shape[1]]
        for b in self.blocks:
            h = b(h)
        return h                                    # (B,T,P,d)

    def readout(self, h):
        return self.rd(torch.cat([h[:, -1], h.mean(1)], -1))   # (B,P,d)

    def parts(self, g, V):
        """Auxiliary only: slots for G2/G3 and the retrieval descriptor."""
        m = V.float().mean(1) > 0.3
        return self.slot(g, m)

    def forward(self, X, V, tc, noise=0.0):
        """Predict per-point displacement for h=1..H, all steps at once."""
        Xc = X[:, :tc]
        if noise > 0:
            # GNS random-walk noise: perturb the CONTEXT the model reads,
            # leave the targets alone, so it learns to correct drift of the
            # same kind its own predictions will contain.
            w = torch.randn_like(Xc) * noise
            Xc = Xc + torch.cumsum(w, dim=1) / (tc ** 0.5)
        h = self.encode(Xc, V[:, :tc])
        g = self.readout(h)                          # (B,P,d)
        g = self.interact(g)
        B, P, _ = g.shape
        disp = self.dec(g).view(B, P, self.M, self.H, 3)
        return dict(disp=disp.permute(0, 2, 3, 1, 4),   # (B,M,H,P,3)
                    logit=self.mode_logit(g.mean(1)),   # (B,M)
                    feat=h, pt=g)

    def arrow(self, h):
        return self.aot(h.mean((1, 2))).squeeze(-1)
