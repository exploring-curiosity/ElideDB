#!/bin/sh
# First start fetches the demo store (Space repos cap at 1 GB, so it ships
# as a public dataset). Nothing else is downloaded: this service loads no
# model, which is the point of it.
set -e
cd /app
if [ -n "$DEMO_STORE_DATASET" ] && [ ! -d /app/lake/bench ]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["DEMO_STORE_DATASET"],
                  repo_type="dataset", local_dir="/app/lake")
print("demo store downloaded", flush=True)
PY
fi
exec python deploy/qbe_serve.py
