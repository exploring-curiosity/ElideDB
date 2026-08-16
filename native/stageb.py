"""Stage B: label-free contrastive head, per PROBLEM2.md amendments.
Views     = the two temporal HALVES of an event (sub-window positives)
NN-pos    = nearest neighbour among other events (NNCLR - repetitive
            corpora make naive negatives FALSE negatives)
AoT-neg   = the REVERSED clip's representation (hard arrow-of-time
            negative; only for events with real motion)
Inputs    = diff-aware token features + flow. Truth never enters;
            training IS the write path."""
import sys
sys.path.insert(0, 'native')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from pathlib import Path

Z1 = np.load('data/cache/stagea_prim_v2.npz')
Z2 = np.load('data/cache/stagea2_prim_v2.npz')
N = len(Z1['prim'])
dev = 'mps'

delta = torch.tensor(Z1['delta'].astype(np.float32))
gidx = torch.argsort(delta, 1, descending=True)[:, :48]
flow = torch.tensor(((Z1['flow'] - Z1['flow'].mean(0))
                     / np.maximum(Z1['flow'].std(0), 1e-6))
                    .astype(np.float32)).to(dev)


def toks(arr):
    t = torch.tensor(arr.astype(np.float32))
    return torch.stack([t[i, gidx[i]] for i in range(N)]).to(dev)


H1, H2 = toks(Z2['h1']), toks(Z2['h2'])
H1R, H2R = toks(Z2['h1r']), toks(Z2['h2r'])


class Head(nn.Module):
    def __init__(self, d=1024, h=256, z=128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, h), nn.GELU(),
                                 nn.Linear(h, h))
        self.proj = nn.Sequential(nn.Linear(h + 17, h), nn.GELU(),
                                  nn.Linear(h, z))

    def forward(self, T, fl):
        p = self.phi(T).mean(1)
        return Fn.normalize(self.proj(torch.cat([p, fl], -1)), dim=-1)


def train(seed=0, epochs=300, tau=0.15, nn_pos=True):
    torch.manual_seed(seed)
    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4,
                            weight_decay=1e-4)
    for ep_ in range(epochs):
        z1 = head(H1, flow)
        z2 = head(H2, flow)
        zr1 = head(H2R, flow)          # reversed clip, its two views
        zr2 = head(H1R, flow)
        anchors = z1
        pos = z2
        if nn_pos and ep_ > 50:
            with torch.no_grad():
                S = z1 @ z2.T
                S.fill_diagonal_(-2)
                nnidx = S.argmax(1)
            pos = 0.5 * (z2 + z2[nnidx])
            pos = Fn.normalize(pos, dim=-1)
        # logits: positive vs (other events) U (reversed selves)
        logits = torch.cat([
            (anchors * pos).sum(-1, keepdim=True),
            anchors @ z2.T + torch.eye(N, device=dev) * -9,
            (anchors * zr1).sum(-1, keepdim=True),
            anchors @ zr2.T], 1) / tau
        loss = Fn.cross_entropy(
            logits, torch.zeros(N, dtype=torch.long, device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = Fn.normalize(head(H1, flow) + head(H2, flow), dim=-1)
    return z.cpu().numpy(), float(loss)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent))
    from stagea_bench import bench, csls_diffuse, rankz
    import stagea_bench as SB
    Zs = []
    for seed in (0, 1, 2):
        z, l = train(seed)
        Zs.append(z)
        print(f'seed {seed} final loss {l:.3f}')
    S = sum(z @ z.T for z in Zs) / 3
    bench(S, 'Stage B head (3-seed mean)')
    bench(csls_diffuse(S), 'Stage B + L8')
    Sf = SB.rankz(S) + SB.rankz(SB.Sf)
    bench(Sf, 'Stage B + flow fusion')
    bench(csls_diffuse((Sf - Sf.mean()) / Sf.std()), 'B + flow + L8')
