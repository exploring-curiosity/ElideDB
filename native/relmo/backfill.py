"""Add window_starts to track files that were written without it.

tracks2 copies a fixed list of privileged fields from the episode into
the track file, and window_starts was not on it. train_wm2.load() opens
the TRACK file to pick t0 from the motion windows, so the field never
reached the code that needed it: every rcasa sample silently fell back
to a uniform t0. Measured on this corpus, that is median window
displacement 0.0004 m instead of 0.1717 m - the trainer would have spent
the run learning to predict stillness.

tracks2 is fixed, but a tracking job already in flight holds the old
code in memory, so files written before the fix still lack the field.
Re-tracking them would cost ~40 s each on the GPU; copying three small
arrays across costs milliseconds and touches nothing else in the file.

Idempotent, and safe to run while tracking continues: tracks2 writes to
a .tmp_ name and renames, so a file that exists is finished.

    python -m relmo.backfill --dataset rcasa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402

FIELDS = ("window_starts", "window_len", "n_windows",
          # the relational channel, dropped by a key-name mismatch
          "contact_pairs", "contact_n", "qpos", "xquat", "body_parentid",
          "jnt_type", "jnt_axis", "jnt_range", "jnt_bodyid", "jnt_qposadr",
          "target_bodies", "body_mass")


def run(dataset="rcasa"):
    from tqdm import tqdm
    man = R.read_manifest(dataset)
    by = {e["id"]: e for e in man["episodes"]}
    files = sorted((R.TRACKS / dataset).glob("*.npz"))
    fixed = already = missing_src = 0
    for f in tqdm(files, unit="f", desc=f"backfill/{dataset}"):
        z = np.load(f)
        if all(k in z.files for k in FIELDS if k != "n_windows"):
            already += 1
            continue
        e = by.get(f.stem)
        if e is None:
            missing_src += 1
            continue
        sp = R.dataset_dir(dataset) / e["shard"] / f.stem / "state.npz"
        if not sp.exists():
            missing_src += 1
            continue
        st = np.load(sp)
        pay = {k: z[k] for k in z.files}
        for k in FIELDS:
            if k in st.files:
                pay[k] = st[k]
        if "contact_pairs" not in pay:
            missing_src += 1
            continue
        tmp = f.with_name(f".bf_{f.name}")
        np.savez_compressed(tmp, **pay)
        tmp.rename(f)
        fixed += 1
    return dict(dataset=dataset, files=len(files), fixed=fixed,
                already_ok=already, no_source=missing_src)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    a = ap.parse_args()
    r = run(a.dataset)
    print("\n" + json.dumps(r, indent=1))
    R.log("backfill_window_starts", **r)
    # prove it took, on disk, rather than trusting the counter
    files = sorted((R.TRACKS / a.dataset).glob("*.npz"))
    have = sum("window_starts" in np.load(f).files for f in files)
    n = [len(np.load(f)["window_starts"]) for f in files
         if "window_starts" in np.load(f).files]
    print(f"VERIFIED on disk: {have}/{len(files)} track files carry "
          f"window_starts, median {int(np.median(n)) if n else 0} per file")
