#!/bin/sh
# First start fetches what the image does not carry: the demo store
# (Space repos cap at 1 GB, so it ships as a public dataset), the
# InternVideo2 assembly, the WordNet corpus, and every text tower
# checkpoint. All of it happens BEFORE the server accepts traffic and
# caches under /app, so no user query ever pays a download.
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
python scripts/get_iv2.py
# warm and serve in ONE process: the towers a separate warm process
# loads die with it, and the first user query would pay them again
exec python deploy/serve.py
