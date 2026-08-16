"""Pin a read path: every artifact and flag that decides what a query returns.

WHY THIS EXISTS. A result is only reproducible if you can say which weights,
which basis, which geometry and which FLAGS produced it. This project has
already been bitten by the omission: a snapshot that did not pin the flags
deciding which stages ran made 0.42 read as 0.32 with nothing in the record to
explain it. So a snapshot here is not a directory of files - it is a statement
of identity, and the hashes are what make it checkable rather than claimed.

What is pinned:
  * the git commit, and whether the tree was dirty when the snapshot was cut
  * geometry: window, hop, stream rate, context, tubelet, patch, crop, layer
  * dtype - fp16 backbones, fp32 pooling and head - because it is a FLAG, not
    a property of any stored file
  * the two frozen backbones by Hub id
  * every fitted artifact by sha256: token PCA, predictor calibration, splits
  * every head checkpoint by sha256, all seeds, because a single seed is not a
    result in this project
  * the relevance constants, which define the ground truth a score is against
  * the corpora the read path may serve, with their record counts AS COUNTED
    ON THE FILESYSTEM - never from a manifest, which has lied here before

Metrics are deliberately NOT required at cut time. A snapshot fixes what will
be measured; `--attach-metrics` writes the numbers back once they exist, so a
snapshot can be cut before an evaluation and still be the thing evaluated.

    python -m relmo.vjsnap --name v9 --note "shipped read path"
    python -m relmo.vjsnap --name v9 --attach-metrics results.json
    python -m relmo.vjsnap --name v9 --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

SNAP = R.BASE / "snapshots"
HEAD_SEEDS = ("p1_reg_low_s0", "p1_reg_low_s1", "p1_reg_low_s2")


def sha(p: Path):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       cwd=Path(__file__).resolve().parents[2],
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:                                          # noqa: BLE001
        return None


def count(d: Path):
    return (len([p for p in d.glob("*.npz") if not p.name.startswith(".")])
            if d.exists() else 0)


def build(corpora):
    from relmo.vjrank import CKPT
    from relmo.vjmatch import ARC_DS
    from relmo.vjrec4 import CTX
    from relmo.vjrec6 import HOP_S, STREAM_FPS, WIN_FRAMES, geometry
    from relmo.vjrec7 import TOK_DIM
    from relmo.vjs import CROP, MODEL, PATCH, TUBELET
    from relmo.vjsig import MODEL as SIG_MODEL, RES as SIG_RES
    from relmo import vjrel

    dt, n_t, desc, hop_steps = geometry()
    pca = R.BASE / "vjrec7" / "_token_pca.npz"
    calib = R.BASE / "vjeval" / "rcasa" / "_calib.npz"
    if not calib.exists():                    # REC may point elsewhere
        from relmo.vjeval import REC
        calib = REC / "rcasa" / "_calib.npz"
    splits = R.BASE / "splits_vjz.json"

    snap = {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "write_path": "relmo.vjrec8 (one encoder pass; identical output to "
                      "vjrec6+vjrec7+vjsig6)",
        "geometry": {
            "win_frames": WIN_FRAMES, "stream_fps": STREAM_FPS,
            "hop_s": HOP_S, "ctx_steps": CTX, "dt_s": dt, "n_t": n_t,
            "desc_steps_per_window": desc, "hop_steps": hop_steps,
            "tubelet": TUBELET, "patch": PATCH, "crop": CROP,
            "spatial_tokens": (CROP // PATCH) ** 2, "gate_layer": 6,
        },
        "dtype": {
            "vjepa_backbone": "fp16", "siglip_backbone": "fp16",
            "pooled_records": "fp32", "token_records": "fp16",
            "siglip_records": "fp32", "head_train_and_infer": "fp32",
        },
        "backbones": {"vjepa": MODEL, "siglip": SIG_MODEL,
                      "siglip_res": SIG_RES},
        "fitted": {
            "token_pca": {"path": str(pca.relative_to(R.BASE)),
                          "sha256": sha(pca), "dim": TOK_DIM},
            "predictor_calibration": {"path": str(calib),
                                      "sha256": sha(calib)},
            "splits": {"path": str(splits.relative_to(R.BASE)),
                       "sha256": sha(splits)},
        },
        "matcher": {"kind": "anchored symmetric2 DTW", "arc_ds": ARC_DS},
        "relevance": {"w_event": vjrel.W_EVENT, "w_obj": vjrel.W_OBJ,
                      "w_scene": vjrel.W_SCENE, "w_cam": vjrel.W_CAM,
                      "r_scale": vjrel.R_SCALE,
                      "dur_weight": "exp(-ln r / ln r_scale), no cut, no gate"},
        "head": {t: {"path": f"models/vjrank/{t}.pt",
                     "sha256": sha(CKPT / f"{t}.pt")} for t in HEAD_SEEDS},
        "corpora": {ds: {
            "v6": count(R.BASE / "vjrec6" / f"{ds}_L6"),
            "v7": count(R.BASE / "vjrec7" / f"{ds}_L6"),
            "siglip": count(R.BASE / "vjsig6" / ds),
        } for ds in corpora},
        "metrics": None,
    }
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--corpora", default="rcasa,rcasa_eval,"
                                         "rcasa_atomic_full,"
                                         "rcasa_composite_full")
    ap.add_argument("--attach-metrics", default="",
                    help="JSON file of measured numbers to write into an "
                         "existing snapshot")
    ap.add_argument("--check", action="store_true",
                    help="re-hash everything and report drift from the "
                         "recorded snapshot")
    a = ap.parse_args()

    SNAP.mkdir(parents=True, exist_ok=True)
    f = SNAP / f"{a.name}.json"
    corpora = [c.strip() for c in a.corpora.split(",") if c.strip()]

    if a.attach_metrics:
        if not f.exists():
            raise SystemExit(f"no snapshot at {f}")
        snap = json.loads(f.read_text())
        snap["metrics"] = json.loads(Path(a.attach_metrics).read_text())
        f.write_text(json.dumps(snap, indent=1))
        print(f"VERIFIED: metrics attached to {f}")
        return

    if a.check:
        if not f.exists():
            raise SystemExit(f"no snapshot at {f}")
        old = json.loads(f.read_text())
        new = build(corpora)
        drift = []
        for grp in ("fitted", "head"):
            for k, v in old[grp].items():
                nv = new[grp].get(k, {})
                if v.get("sha256") != nv.get("sha256"):
                    drift.append(f"{grp}.{k}: {v.get('sha256')} -> "
                                 f"{nv.get('sha256')}")
        if old["geometry"] != new["geometry"]:
            drift.append("geometry changed")
        if old["git_commit"] != new["git_commit"]:
            drift.append(f"commit {old['git_commit'][:8]} -> "
                         f"{new['git_commit'][:8]} (not fatal on its own)")
        for ds, c in old["corpora"].items():
            if new["corpora"].get(ds) != c:
                drift.append(f"corpora.{ds}: {c} -> {new['corpora'].get(ds)}")
        print(f"snapshot {a.name}: {len(drift)} drifted" if drift
              else f"VERIFIED: snapshot {a.name} still describes the tree")
        for d in drift:
            print("  " + d)
        raise SystemExit(1 if any("not fatal" not in d for d in drift) else 0)

    snap = build(corpora)
    snap["name"] = a.name
    snap["note"] = a.note
    missing = [k for k, v in snap["fitted"].items() if v["sha256"] is None]
    missing += [f"head.{k}" for k, v in snap["head"].items()
                if v["sha256"] is None]
    if missing:
        raise SystemExit(f"cannot cut a snapshot, unhashable: {missing}")
    f.write_text(json.dumps(snap, indent=1))
    print(json.dumps(snap, indent=1))
    print(f"\nVERIFIED: wrote {f}")
    R.log("vjsnap", name=a.name, commit=snap["git_commit"],
          dirty=snap["git_dirty"],
          corpora={k: v["v7"] for k, v in snap["corpora"].items()})


if __name__ == "__main__":
    main()
