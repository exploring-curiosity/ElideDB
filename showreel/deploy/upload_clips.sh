#!/usr/bin/env bash
# Push the 1.34 GB of clips to S3 and repoint the memory at them.
#
#   BUCKET=precedent-clips-yourname ./showreel/deploy/upload_clips.sh
#
# 1.34 GB sits inside S3's always-free 5 GiB. The `video` column becomes an
# s3:// key rather than a local path, and the console serves each clip with a
# short-lived presigned URL — the bucket itself stays private.
set -euo pipefail
: "${BUCKET:?set BUCKET}"
REGION="${AWS_REGION:-us-east-2}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

"$ROOT/.venv-libero/bin/python" - <<'PY'
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "showreel"))
import db
rows = db.q("SELECT rec_id, video FROM moments WHERE video NOT LIKE 's3://%'")
print(f"{len(rows)} clips to upload")
bucket = os.environ["BUCKET"]
for i, r in enumerate(rows, 1):
    key = f"clips/{r['rec_id']}.mp4"
    subprocess.run(["aws", "s3", "cp", r["video"], f"s3://{bucket}/{key}",
                    "--only-show-errors"], check=True)
    db.q("UPDATE moments SET video=%s WHERE rec_id=%s",
         (f"s3://{bucket}/{key}", r["rec_id"]))
    if i % 100 == 0:
        print(f"  {i}/{len(rows)}")
print("done")
PY
