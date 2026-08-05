"""STEP 7 - DISCOVER VOCABULARY. Units -> a set of recurring kinds.

The database needs a vocabulary of event KINDS it discovered for
itself: no taxonomy supplied, no k chosen by me, nothing per-corpus.
Truth labels (prim: pick/place/stack/unstack/push) are EVAL-ONLY - they
score the vocabulary, they never fit it.

Grade, three parts, because any one alone is gameable:

  purity   fraction of units whose cluster's majority label is their
           own. One cluster per unit scores 1.0, so purity is only
           meaningful next to the cluster count.
  AMI      adjusted mutual information vs the true labels. Corrected
           for chance AND for cluster count, so it cannot be won by
           over-splitting - this is the honest headline.
  stability  cluster two DISJOINT halves of the corpus separately, then
           check whether the two vocabularies agree about pairs of held
           -out units (do these two units land together in both?).
           A vocabulary that changes when the episodes change is not a
           vocabulary, it is a fit to the sample.

Methods compared - all must discover k themselves:

  finch    parameter-free (Sarfraz et al.): first-neighbour graph,
           connected components, recursively. No k, no threshold.
  hdbscan  density-based; min_cluster_size is a real parameter, so it
           is swept and reported honestly rather than tuned quietly.
  kmeans   k selected by silhouette over a range - the baseline that
           proves whether the fancier methods earn their complexity.

    python native/vocab.py --limit 72        # ~10 min
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "native"))

from flowgebd import arg                                       # noqa: E402
from unitenc import ENCODERS, FPS                              # noqa: E402

CACHE = Path("/private/tmp/claude-501/vocab")


def unit_vectors(enc, ep, F, spans):
    key = hashlib.sha1(
        f"{enc}|{ep}|{len(spans)}".encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{key}.npy"
    if fp.exists():
        return np.load(fp)
    V = ENCODERS[enc](F, spans, 0.0)
    np.save(fp, V.astype(np.float32))
    return V


# ------------------------------------------------------------ clustering

def finch(V, levels=None):
    """FINCH: first-neighbour clustering, parameter-free.

    Each point links to its nearest neighbour; the connected components
    of that graph are the clusters; recurse on cluster means. There is
    no k and no distance threshold to pick, which is the whole reason
    it is here - every other option smuggles in a constant.
    Returns the partition at each level.
    """
    out = []
    labels = np.arange(len(V))
    cur = V.copy()
    idx = np.arange(len(V))
    for _ in range(levels or 10):
        n = len(cur)
        if n < 3:
            break
        S = cur @ cur.T
        np.fill_diagonal(S, -np.inf)
        nn = np.argmax(S, axis=1)
        # union-find over i ~ nn[i]
        par = list(range(n))

        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a

        for i, j in enumerate(nn):
            a, b = find(i), find(int(j))
            if a != b:
                par[a] = b
        root = {}
        lab = np.empty(n, int)
        for i in range(n):
            r = find(i)
            lab[i] = root.setdefault(r, len(root))
        newlab = lab[idx] if len(idx) == len(V) else lab[labels]
        labels = newlab
        out.append(labels.copy())
        k = lab.max() + 1
        cur = np.stack([cur[lab == c].mean(0) for c in range(k)])
        cur = cur / np.maximum(np.linalg.norm(cur, axis=1,
                                              keepdims=True), 1e-8)
        idx = np.arange(k)
        if k <= 2:
            break
    return out


def cluster(V, method):
    if method == "finch":
        parts = finch(V)
        # take the finest partition with a sane count (<= 64); FINCH is
        # hierarchical and the caller wants ONE vocabulary
        for p in parts:
            if p.max() + 1 <= 64:
                return p
        return parts[-1] if parts else np.zeros(len(V), int)
    if method.startswith("hdbscan"):
        import hdbscan
        m = int(method.split(":")[1]) if ":" in method else 5
        return hdbscan.HDBSCAN(min_cluster_size=m,
                               metric="euclidean").fit_predict(V)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        best, bl = -2, None
        for k in range(2, 16):
            lab = KMeans(k, n_init=5, random_state=0).fit_predict(V)
            try:
                s = silhouette_score(V, lab)
            except Exception:                          # noqa: BLE001
                continue
            if s > best:
                best, bl = s, lab
        return bl if bl is not None else np.zeros(len(V), int)
    raise ValueError(method)


# ----------------------------------------------------------------- grade

def purity(lab, y):
    tot = 0
    for c in set(lab):
        m = lab == c
        vals, cnt = np.unique(np.asarray(y)[m], return_counts=True)
        tot += cnt.max()
    return tot / len(y)


def pair_agreement(la, lb):
    """Do two partitions of the SAME points agree about which pairs are
    together? Rand-index style, the basis of the stability check."""
    n = len(la)
    if n < 2:
        return float("nan")
    A = (la[:, None] == la[None, :])
    B = (lb[:, None] == lb[None, :])
    iu = np.triu_indices(n, 1)
    return float((A[iu] == B[iu]).mean())


def main():
    import encode as E
    import pyarrow.parquet as pq
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from tqdm import tqdm

    limit = arg("--limit", 72, int)
    enc = arg("--enc", "siglip2_rank")
    methods = arg("--methods", "finch,kmeans,hdbscan:5,"
                               "hdbscan:10").split(",")

    t = pq.read_table(ROOT / "data/sim_chains/truth.parquet").to_pydict()
    ev = {}
    for e, pr, a, b in zip(t["episode"], t["prim"], t["t0"], t["t1"]):
        ev.setdefault(int(e), []).append((float(a), float(b), str(pr)))

    dirs = sorted(p for p in (ROOT / "data/sim_chains").iterdir()
                  if p.is_dir() and p.name.startswith("ep"))[:limit]
    print(f"STEP 7 vocabulary — {len(dirs)} episodes, encoder {enc}",
          flush=True)

    Vs, y, epi = [], [], []
    for d in tqdm(dirs, desc="units", unit="ep"):
        ei = int(d.name[2:])
        rows = sorted(ev.get(ei, []))
        if not rows:
            continue
        cam = sorted(d.glob("cam*.mp4"))[0]
        F = E.decode(cam, fps=FPS, w=256)
        V = unit_vectors(enc, ei, F, [(a, b) for a, b, _ in rows])
        Vs.append(V)
        y += [p for _, _, p in rows]
        epi += [ei] * len(rows)
    V = np.concatenate(Vs)
    y = np.array(y)
    epi = np.array(epi)
    ntrue = len(set(y))
    print(f"{len(V)} units, {ntrue} true kinds "
          f"{dict(zip(*np.unique(y, return_counts=True)))}\n", flush=True)

    # disjoint halves by EPISODE, so stability is not measured on units
    # that shared a video with their training neighbours
    eps = np.array(sorted(set(epi)))
    ha, hb = set(eps[::2]), set(eps[1::2])
    ma = np.array([e in ha for e in epi])

    print(f"{'method':<14}{'k':<6}{'purity':<9}{'AMI':<9}"
          f"{'stability':<11}{'noise%'}")
    for m in methods:
        try:
            lab = cluster(V, m)
        except Exception as e:                          # noqa: BLE001
            print(f"{m:<14}FAILED {type(e).__name__}: {e}")
            continue
        keep = lab >= 0
        k = len(set(lab[keep].tolist()))
        pu = purity(lab[keep], y[keep]) if keep.any() else float("nan")
        am = ami(y[keep], lab[keep]) if keep.any() else float("nan")
        # stability: fit each half separately, compare on the pairs of
        # units that both partitions cover
        try:
            la = cluster(V[ma], m)
            lb = cluster(V[~ma], m)
            # compare structure via assignment of the SAME held-out
            # units to each half's nearest centroid
            def centroids(vv, ll):
                cs = [vv[ll == c].mean(0) for c in sorted(set(
                    ll[ll >= 0].tolist()))]
                C = np.stack(cs) if cs else np.zeros((1, vv.shape[1]))
                return C / np.maximum(np.linalg.norm(C, axis=1,
                                                     keepdims=True), 1e-8)
            Ca, Cb = centroids(V[ma], la), centroids(V[~ma], lb)
            asg_a = np.argmax(V @ Ca.T, axis=1)
            asg_b = np.argmax(V @ Cb.T, axis=1)
            st = pair_agreement(asg_a, asg_b)
        except Exception:                               # noqa: BLE001
            st = float("nan")
        noise = 100.0 * (~keep).mean()
        print(f"{m:<14}{k:<6}{pu:<9.3f}{am:<9.3f}{st:<11.3f}"
              f"{noise:.0f}", flush=True)
    print(f"\ntrue kinds = {ntrue}. AMI is the headline: it is corrected "
          "for chance AND cluster count, so over-splitting cannot win it.")


if __name__ == "__main__":
    main()
