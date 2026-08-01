"""Snapshot the teacher: what it is, what it scored, what it costs.

A teacher is only useful as a distillation target if the exact thing
that produced the labels can be named later. This writes a manifest
pinning the artifacts, the stage configuration, and the measured
numbers, so a student can always be traced to the teacher it copied.

  python scripts/version_teacher.py [--tag v1]
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402


def digest(p: Path):
    if not p.exists():
        return None
    if p.is_dir():
        h = hashlib.sha1()
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                h.update(str(f.stat().st_size).encode())
        return h.hexdigest()[:12]
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def main():
    argv = sys.argv
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else "v1"
    db = Store.open("lake/bench")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    man = {
        "tag": tag,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        "store": "lake/bench",
        # WITHOUT THIS THE MANIFEST IS NOT A SNAPSHOT. v1 listed the ITM
        # cascade among its stages but not the flag that turns it on,
        # and ITM is off by default because it costs 0.4s/episode. Re-
        # running "the versioned teacher" therefore scored 0.32, not the
        # recorded 0.42, against a store whose every table digest still
        # matched - an hour of bisecting an identical store. Pinning the
        # artifacts is only half of reproducibility; the switches that
        # decide which stages execute are the other half.
        "env": {"ELIDEDB_ITM": os.environ.get("ELIDEDB_ITM", "0"),
                "ELIDEDB_ANCHOR_W": os.environ.get("ELIDEDB_ANCHOR_W",
                                                   "8.0")},
        "reproduce": "ELIDEDB_ITM=1 python scripts/bench_truth.py",
        "metric": {
            "definition": "yield=true/support, prec=true/returned, "
                          "k=ceil(1.5*support)",
            "mean_yield": 0.45, "mean_prec": 0.32,
            "per_query_yield": {"q00": 0.38, "q01": 0.41, "q02": 0.50,
                                "q03": 0.62, "q04": 0.92, "q05": 0.76,
                                "q07": 0.58, "q08": 0.39, "q09": 0.00,
                                "q10": 0.00},
            "per_query_prec": {"q00": 0.25, "q01": 0.27, "q02": 0.33,
                               "q03": 0.62, "q04": 0.61, "q05": 0.51,
                               "q07": 0.39, "q08": 0.26, "q09": 0.00,
                               "q10": 0.00},
            "no_match_gate": "q06 PASS",
        },
        "stages": [
            "cosine channels (pe, sig2, iv2, obj, act, vid, conj) -> RRF",
            "PRF Rocchio round",
            "ITM cross-encoder cascade over top-N (rerank STAGE, not a "
            "weighted channel: as a channel it cost 0.38 -> 0.27)",
            "routed event-transition filter (unsupervised corroboration "
            "gate) + motion-space density",
            "transition ANCHOR: corpus-derived motion direction per "
            "transition kind, weighted by a self-supervised reliability "
            "the kind earns without labels (q04 0.72 -> 0.92, q05 0.58 "
            "-> 0.76, every other query bit-identical)",
            "fitted filter / NMS / confidence cut",
        ],
        "models": {
            "appearance": "google/siglip2-so400m-patch14-384",
            "video_text": "models/iv2_stage2_1b (InternVideo2-Stage2 1B)",
            "cross_encoder": "iv2 itm_head (the discarded checkpoint head)",
            "namer": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "detector": "IDEA-Research/grounding-dino-base",
        },
        "tables": {},
        "artifacts": {},
    }
    for t in sorted(db.tables()):
        st = db.table(t).state()
        man["tables"][t] = {"version": st.version,
                            "rows": db.table(t).scan().num_rows}
    for a in ("_set_weights.json", "_vocab.json", "_channel_weights.json"):
        man["artifacts"][a] = digest(Path("lake/bench") / a)
    for a in ("artifacts/verbs_v2.json", "artifacts/cavity.json"):
        man["artifacts"][a] = digest(ROOT / a)

    out = ROOT / f"models/teacher_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(man, indent=1))
    print(json.dumps({"wrote": str(out), "tag": tag, "commit": commit,
                      "tables": len(man["tables"])}, indent=1))


if __name__ == "__main__":
    main()
