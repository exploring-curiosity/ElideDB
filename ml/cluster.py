#!/usr/bin/env python3
"""Cluster window embeddings: PCA(50) -> HDBSCAN -> clusters.json.

- HDBSCAN runs on PCA-reduced vectors (density estimates degrade in 1000+
  dims); it discovers scene types without choosing k, and points it cannot
  place get label -1 (noise) — the C++ index always scans those, never drops
  them.
- Centroids are computed in the FULL embedding space (normalized member
  mean): the coarse stage must rank in the same metric space the fine stage
  scores in, or the prune would be unsound. This is IVF with learned cells.
- UMAP (if installed) writes umap.npy for the 2D demo map. Visualization
  ONLY: UMAP preserves neighborhoods, not distances — never used in retrieval.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pca-dims", type=int, default=50)
    ap.add_argument("--min-cluster-size", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.store) / "ml" / args.run_id
    meta = json.loads((run_dir / "meta.json").read_text())
    n = len(json.loads((run_dir / "windows.json").read_text())["windows"])
    vecs = np.fromfile(run_dir / "embeddings.f32",
                       dtype=np.float32).reshape(n, meta["dim"])

    from sklearn.decomposition import PCA
    import hdbscan

    pca_dims = min(args.pca_dims, n, meta["dim"])
    reduced = PCA(n_components=pca_dims, random_state=0).fit_transform(vecs)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size).fit_predict(reduced)

    k = labels.max() + 1
    centroids = []
    for c in range(k):
        m = vecs[labels == c].mean(axis=0)
        centroids.append((m / np.linalg.norm(m)).tolist())
    noise = int((labels == -1).sum())
    sizes = [int((labels == c).sum()) for c in range(k)]
    (run_dir / "clusters.json").write_text(json.dumps({
        "labels": labels.tolist(),
        "centroids": centroids,
        "pca_dims": pca_dims,
        "min_cluster_size": args.min_cluster_size,
    }))
    print(f"{k} clusters, sizes {sizes}, {noise} noise points "
          f"({100.0 * noise / n:.1f}%)")

    try:
        import umap
        xy = umap.UMAP(n_components=2, random_state=0).fit_transform(reduced)
        np.save(run_dir / "umap.npy", xy.astype(np.float32))
        print("wrote umap.npy (visualization only)")
    except ImportError:
        print("umap-learn not installed; skipping 2D map")
    print(f"next: python3 ml/register_run.py --store {args.store} "
          f"--run-id {args.run_id}")


if __name__ == "__main__":
    main()
