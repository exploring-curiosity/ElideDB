"""Score a channel's vectors on the three layers, teacher or student.

Labels come from the WRITE PATH - event kinds produced by geometry,
object ids produced by the identity store - never from the truthset.
The truthset is eval-only, and a metric that steers per-channel
iteration must not read it.

  python scripts/bench_channel.py --npz scratch_distill/vjepa_bridge4h_400.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from elidedb import Store                                    # noqa: E402
from elidedb.chanbench import report                         # noqa: E402


def write_path_labels(store, keys):
    """{kind: bool per clip} from events, plus object-id groups.

    keys is [(stream, episode_ts)] in the same order as the vectors.
    """
    db = Store.open(store)
    lab, groups = {}, None
    if "events" in db.tables():
        ev = db.table("events").scan().to_pydict()
        have = {}
        for i in range(len(ev["ts"])):
            have.setdefault((str(ev["stream"][i]), int(ev["ts"][i])),
                            set()).add(str(ev["kind"][i]))
        kinds = sorted({k for v in have.values() for k in v})
        for k in kinds:
            y = np.array([k in have.get(key, ()) for key in keys], int)
            if 5 <= y.sum() <= len(y) - 5:
                lab[k] = y
    if "instances" in db.tables():
        inst = db.table("instances").scan().to_pydict()
        best = {}
        for i in range(len(inst["ts"])):
            key = (str(inst["stream"][i]), int(inst["ep_ts"][i]))
            n = int(inst["n_frames"][i])
            if n > best.get(key, (0, -1))[0]:
                best[key] = (n, int(inst["object_id"][i]))
        g = np.array([best.get(k, (0, -1))[1] for k in keys])
        if len(set(g[g >= 0])) >= 2:
            groups = g
    return lab, groups


def main():
    argv = sys.argv
    npz = argv[argv.index("--npz") + 1]
    store = argv[argv.index("--store") + 1] if "--store" in argv else "lake/bridge4h"
    z = np.load(npz, allow_pickle=True)
    Y = z["Y"]
    keys = [tuple(k) for k in z["keys"]] if "keys" in z.files else None

    labels, groups = ({}, None)
    if keys:
        keys = [(str(a), int(b)) for a, b in keys]
        labels, groups = write_path_labels(store, keys)
    r = report(Y, labels or None, groups)
    print(json.dumps({"vectors": npz, "n": len(Y), **r}, indent=1))


if __name__ == "__main__":
    main()
