#!/usr/bin/env python3
"""Put the corpus in S3 and repoint the memory at it.

    BUCKET=precedent-... .venv-libero/bin/python showreel/deploy/upload_assets.py

Two asset classes, both outside the database, uploaded to one private bucket:

    clips/{rec_id}.mp4   3,402 files, 1.34 GB   read by the BROWSER
    traces/{rec_id}.npy  3,556 files, 1.16 GB   read by the SIDECAR

2.5 GB total, inside S3's always-free 5 GiB. Nothing here is made public: the
console hands out presigned URLs with an hour of life, so the only way to read a
clip is to have asked this app for it.

Only the database this DSN points at is repointed, which is why local Postgres
and CockroachDB end up different: the laptop keeps playing files off disk with
no AWS credentials at all, and the cloud deployment reads S3. `repoint.py` moves
either one in either direction.

Resumable by construction. It lists what the bucket already holds and uploads
the difference, so an interrupted run costs the objects in flight rather than
the run. That matters at 7,000 objects over a home connection.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "showreel"))

import boto3                                                       # noqa: E402
from botocore.config import Config                                 # noqa: E402
from tqdm import tqdm                                              # noqa: E402

import blob                                                        # noqa: E402
import db                                                          # noqa: E402

BUCKET = os.environ.get("BUCKET") or os.environ.get("PRECEDENT_BUCKET") or ""
REGION = os.environ.get("AWS_REGION", "us-east-1")
WORKERS = int(os.environ.get("UPLOAD_WORKERS", "16"))


def s3():
    return boto3.client("s3", region_name=REGION,
                        config=Config(max_pool_connections=WORKERS + 8,
                                      retries={"max_attempts": 5, "mode": "adaptive"}))


def already(c, prefix: str) -> set[str]:
    """Keys the bucket already holds under a prefix. This is what makes it resumable."""
    have, tok = set(), None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
        if tok:
            kw["ContinuationToken"] = tok
        r = c.list_objects_v2(**kw)
        have |= {o["Key"] for o in r.get("Contents", [])}
        if not r.get("IsTruncated"):
            return have
        tok = r["NextContinuationToken"]


def push(c, jobs: list[tuple[pathlib.Path, str, str]], label: str) -> int:
    """Upload (path, key, content-type) triples, counting completions not submissions."""
    done = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(c.upload_file, str(p), BUCKET, k,
                          ExtraArgs={"ContentType": ct}): k for p, k, ct in jobs}
        for f in tqdm(cf.as_completed(futs), total=len(futs), desc=label, unit="obj"):
            f.result()
            done += 1
    return done


def main() -> int:
    if not BUCKET:
        print("set BUCKET to the bucket name", file=sys.stderr)
        return 2
    c = s3()
    print(f"bucket s3://{BUCKET} in {REGION}\n")

    # ---- traces --------------------------------------------------------
    have = already(c, "traces/")
    jobs = [(p, f"traces/{p.stem}.npy", "application/octet-stream")
            for p in sorted(blob.CACHE.glob("*.npy"))
            if f"traces/{p.stem}.npy" not in have]
    print(f"traces: {len(have)} already there, {len(jobs)} to send")
    if jobs:
        push(c, jobs, "traces")

    # ---- clips ---------------------------------------------------------
    # The `video` column is the source of truth for which clips matter, and
    # rewriting it to an s3:// key is what actually moves the deployment. A file
    # uploaded but not recorded here is invisible to the app.
    rows = db.q("SELECT rec_id, video FROM moments ORDER BY rec_id")
    local = [r for r in rows if not (r["video"] or "").startswith("s3://")]
    have = already(c, "clips/")
    jobs, rewrite = [], []
    missing = 0
    for r in local:
        key = f"clips/{r['rec_id']}.mp4"
        if not os.path.exists(r["video"]):
            missing += 1
            continue
        if key not in have:
            jobs.append((pathlib.Path(r["video"]), key, "video/mp4"))
        rewrite.append((f"s3://{BUCKET}/{key}", r["rec_id"]))
    print(f"clips: {len(rows) - len(local)} already s3, {len(jobs)} to send, "
          f"{missing} with no file on disk")
    if jobs:
        push(c, jobs, "clips")

    # ---- repoint -------------------------------------------------------
    # One statement per 500 rows, not one per row. Against CockroachDB Cloud a
    # round trip is about 700 ms from here, so 3,402 single-row updates is 40
    # minutes of latency to change one column. The key is derivable from the
    # primary key, so the new value can be computed inside the statement and the
    # only thing sent over the wire is the list of ids.
    ids = [rid for _u, rid in rewrite]
    for i in tqdm(range(0, len(ids), 500), desc="repoint", unit="batch"):
        db.q(f"UPDATE moments SET video = 's3://{BUCKET}/clips/' || rec_id || '.mp4' "
             "WHERE rec_id = ANY(%s)", (ids[i:i + 500],))

    n = db.q("SELECT count(*) n FROM moments WHERE video LIKE 's3://%'")[0]["n"]
    tr = len(already(c, "traces/"))
    print(f"\n{n} of {len(rows)} moments point at S3, {tr} traces in the bucket")
    return 0


if __name__ == "__main__":
    sys.exit(main())
