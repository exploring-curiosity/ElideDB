#!/usr/bin/env python3
"""Point a database's `video` column at S3 or back at local disk.

    .venv-libero/bin/python showreel/deploy/repoint.py --to local
    BUCKET=... .venv-libero/bin/python showreel/deploy/repoint.py --to s3

Which database is decided by PRECEDENT_DSN, so this is run once per deployment.
It exists because "works locally" has to keep being true after the corpus moves:
local Postgres is the local deployment and CockroachDB is the cloud one, and a
laptop with no AWS credentials should still be able to play a clip.

Going back to local reads the paths out of RelMo's manifests rather than the
store, because the store is 17.64 GB to open and the manifest is the file that
actually knows where the video is.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "native"))
sys.path.insert(0, str(ROOT / "showreel"))

import db                                                          # noqa: E402

WITH_VIDEO = ("rcasa_atomic_full", "rcasa_composite_full")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", choices=("s3", "local"), required=True)
    a = ap.parse_args()

    if a.to == "s3":
        bucket = os.environ.get("BUCKET") or os.environ.get("PRECEDENT_BUCKET")
        if not bucket:
            print("set BUCKET", file=sys.stderr)
            return 2
        n = db.q("SELECT count(*) n FROM moments WHERE video NOT LIKE 's3://%'")[0]["n"]
        db.q(f"UPDATE moments SET video = 's3://{bucket}/clips/' || rec_id || '.mp4' "
             "WHERE video NOT LIKE 's3://%'")
        print(f"{n} rows now point at s3://{bucket}/clips/")
        return 0

    from tqdm import tqdm

    from relmo import registry as R

    have = {}
    for ds in WITH_VIDEO:
        for e in R.read_manifest(ds)["episodes"]:
            v = e.get("video") or ""
            if v and os.path.exists(v):
                have[e["id"]] = v
    rows = db.q("SELECT rec_id FROM moments WHERE video LIKE 's3://%'")
    hit = [(have[r["rec_id"]], r["rec_id"]) for r in rows if r["rec_id"] in have]
    for v, rid in tqdm(hit, desc="repoint local", unit="row"):
        db.q("UPDATE moments SET video=%s WHERE rec_id=%s", (v, rid))
    print(f"{len(hit)} rows point at local files, {len(rows) - len(hit)} had none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
