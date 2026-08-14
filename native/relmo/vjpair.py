"""Label-free positive pairs from TWO INDEPENDENT CHANNELS AGREEING.

THE GAP THIS FILLS. Everything trained so far optimised per-step regression of
physical quantities plus reconstruction. Nothing in the loss ever saw a PAIR of
clips, so nothing optimised the geometry that cosine + DTW reads at query time.
A label-supervised linear probe reaches 0.862 on the identical frozen features
while the regression-trained recurrence reaches 0.73 - the difference is
discriminative versus generative, not information.

WHY AGREEMENT, AND WHY IT IS NOT A HARDWIRED CATEGORY. Declaring classes
("opening / closing / static x hinged / sliding / free") writes the answer into
the code and was rejected. Clustering only moves the choice into a distance
function. Agreement declares nothing: a pair is positive when the physics trace
AND the SigLIP appearance trace INDEPENDENTLY both rank it among each other's
nearest, negative when both rank it far, and dropped when they disagree. No
class list, no class count, no semantic threshold - only "two unrelated
measurements concur", which is a property of the data.

MEASURED on the 291 training episodes (agreement rate vs the grading labels,
used to CHECK the rule, never inside it):

    rule                       pairs   % same group
    any pair (chance)          84390          0.189
    physics top-5               1164          0.918
    siglip  top-5               1164          0.703
    BOTH agree top-5             436          0.995
    BOTH agree top-10            690          0.981
    BOTH agree top-20           1346          0.952

Neither channel alone is trustworthy - SigLIP is barely above chance at k=20 -
but their intersection is near-perfect. Note also that physics ranks pairs far
better (0.918 @5) than it describes them (oracle 0.490 as a descriptor), which
is the whole argument for using it pairwise instead of as a regression target.

TRAINING TIME ONLY. Physics comes from sim state, so this runs where sim state
exists and never at serve.

    python -m relmo.vjpair --k-pos 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjphys  # noqa: E402
from relmo.vjeval import l2  # noqa: E402
from relmo.vjeval5 import dtw_from_cost  # noqa: E402

OUT = R.BASE / "vjpair"


def seq_sim(seqs):
    """Symmetric subsequence-DTW similarity between (T,D) sequences."""
    X = np.stack([l2(np.asarray(s, np.float32)) for s in seqs])
    n = len(X)
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        C = 1.0 - np.einsum("sd,nkd->nsk", X[i], X)
        S[i] = -dtw_from_cost(C)
    return (S + S.T) / 2


def ranks(S):
    r = np.argsort(np.argsort(-S, 1), 1).astype(np.int32)
    np.fill_diagonal(r, 1 << 20)              # never pair an episode with itself
    return r


def build(ids, recs, dataset="rcasa", k_pos=10, far_frac=0.5):
    """-> (pos, neg) boolean (n,n). Both channels must agree."""
    Y = [vjphys.load(dataset, i) for i in ids]
    A = np.concatenate(Y)
    Z = [(y - A.mean(0)) / (A.std(0) + 1e-9) for y in Y]
    rp = ranks(seq_sim(Z))
    rg = ranks(seq_sim([recs[i]["sig"] for i in ids]))
    n = len(ids)
    far = int(far_frac * n)
    pos = (rp < k_pos) & (rg < k_pos)
    neg = (rp > far) & (rg > far)
    pos = pos | pos.T                         # symmetry: agreement is mutual
    neg = neg & neg.T
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--k-pos", type=int, default=10)
    a = ap.parse_args()

    from relmo.vjood import recs_for
    from relmo.vjsplit import load as load_split
    from relmo.vjeval import group_key, parse

    recs = recs_for(a.dataset)
    ids = [i for i in sorted(load_split()["train"])
           if i in recs and vjphys.load(a.dataset, i) is not None]
    pos, neg = build(ids, recs, a.dataset, a.k_pos)
    g = np.array([group_key(parse(i)) for i in ids])
    same = g[:, None] == g[None, :]
    np.fill_diagonal(same, False)
    n = len(ids)
    print(f"{n} train episodes")
    print(f"  positives {int(pos.sum()):6d}  {same[pos].mean():.3f} same-group "
          f"(chance {same.sum()/(n*n-n):.3f})")
    print(f"  negatives {int(neg.sum()):6d}  {same[neg].mean():.3f} same-group")
    print("  (agreement vs labels is a CHECK on the rule; the rule never "
          "reads a label)")
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.dataset}_k{a.k_pos}.npz"
    np.savez_compressed(f, ids=np.array(ids), pos=pos, neg=neg)
    print(f"wrote {f}")
    R.log("vjpair", dataset=a.dataset, k_pos=a.k_pos, n=n,
          n_pos=int(pos.sum()), n_neg=int(neg.sum()),
          pos_purity=round(float(same[pos].mean()), 4))


def load(dataset="rcasa", k_pos=10):
    f = OUT / f"{dataset}_k{k_pos}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f} - run: python -m relmo.vjpair "
                         f"--k-pos {k_pos}")
    z = np.load(f, allow_pickle=True)
    return [str(x) for x in z["ids"]], z["pos"], z["neg"]


if __name__ == "__main__":
    main()
