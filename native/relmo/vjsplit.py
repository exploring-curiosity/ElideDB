"""Scene-grouped, family-stratified train/val/test split. Frozen to disk.

WHY NOT relmo/splits.py. That module hashes the EPISODE ID. On rcasa that
leaks twice over:

  1. episodes ship as camera variants of one rollout - CloseCabinet_episode_
     000000__robot0_agentview_left and ..._right are the same physical event
     seen from two places. An id hash puts near-duplicates on opposite sides
     of the split and the held-out number reads as generalisation.
  2. a scene (layout_id, style_id) recurs across episodes. Sharing a kitchen
     between train and test lets appearance memorisation inflate the number.

So the grouping unit is the SCENE: all 188 of them, every episode in a scene
moves together, and camera variants follow for free because they share it.

STRATIFIED because the corpus is unbalanced 60:11 across task families. A
uniform random scene split starves LoadDishwasher (11 episodes over 8 scenes)
to zero on one side. Greedy assignment places each scene where its family
composition is furthest below quota.

  train 65 / val 15 / test 20 - the owner's bar is train < 70.

    python -m relmo.vjsplit --write
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

FRAC = {"train": 0.65, "val": 0.15, "test": 0.20}
OUT = R.BASE / "splits_vjz.json"


def build(dataset="rcasa", seed=0):
    man = R.read_manifest(dataset)
    eps = man["episodes"]
    by_scene = defaultdict(list)
    for e in eps:
        by_scene[str(e.get("scene", e["id"]))].append(e)

    fam_tot = Counter(e["task"] for e in eps)
    quota = {s: {f: n * FRAC[s] for f, n in fam_tot.items()} for s in FRAC}
    have = {s: Counter() for s in FRAC}
    out = {s: [] for s in FRAC}

    # largest scenes first: a big scene placed late cannot be compensated for
    order = sorted(by_scene, key=lambda k: (-len(by_scene[k]), k))
    for sc in order:
        grp = by_scene[sc]
        comp = Counter(e["task"] for e in grp)
        # deficit = how far below quota this split is for the families present,
        # normalised by family size so a rare family dominates the decision
        best, bestv = None, None
        for s in FRAC:
            d = sum((quota[s][f] - have[s][f]) / fam_tot[f] * n
                    for f, n in comp.items())
            if bestv is None or d > bestv:
                best, bestv = s, d
        out[best].extend(e["id"] for e in grp)
        have[best].update(comp)
    return out, fam_tot, have, len(by_scene)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    out, fam_tot, have, n_scene = build(a.dataset)
    n = {s: len(v) for s, v in out.items()}
    tot = sum(n.values())
    print(f"{a.dataset}: {tot} episodes over {n_scene} scenes")
    print(f"{'split':7s} {'eps':>5s} {'frac':>6s}  target")
    for s in FRAC:
        print(f"{s:7s} {n[s]:5d} {n[s]/tot:6.3f}  {FRAC[s]:.2f}")

    print(f"\n{'task family':28s} {'tot':>4s} {'train':>6s} {'val':>5s} "
          f"{'test':>5s}   {'test support-1':>14s}")
    for f in sorted(fam_tot, key=lambda x: -fam_tot[x]):
        te = have['test'][f]
        flag = "  <- ungradeable" if te - 1 < 5 else ""
        print(f"{f:28s} {fam_tot[f]:4d} {have['train'][f]:6d} "
              f"{have['val'][f]:5d} {have['test'][f]:5d}   {te-1:14d}{flag}")

    # a scene must never straddle two splits - that is the whole point
    man = R.read_manifest(a.dataset)
    scene_of = {e["id"]: str(e.get("scene", e["id"])) for e in man["episodes"]}
    seen = {}
    for s, ids in out.items():
        for i in ids:
            sc = scene_of[i]
            assert seen.setdefault(sc, s) == s, f"scene {sc} straddles splits"
    print(f"\nverified: {len(seen)} scenes, none straddling a split")

    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(
            dict(dataset=a.dataset, frac=FRAC, n_scenes=n_scene,
                 counts=n, splits=out), indent=1))
        print(f"wrote {OUT}")
        R.log("vjsplit", dataset=a.dataset, n_scenes=n_scene, **n)


def load():
    """{'train': set(ids), ...}. Raises if the split was never frozen."""
    if not OUT.exists():
        raise SystemExit("no frozen split - run: python -m relmo.vjsplit --write")
    d = json.loads(OUT.read_text())
    return {k: set(v) for k, v in d["splits"].items()}


if __name__ == "__main__":
    main()
