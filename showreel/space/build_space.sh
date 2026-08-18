#!/usr/bin/env bash
# Assemble the Hugging Face Space repo. Nothing is built here that is not run.
#
#   ./showreel/space/build_space.sh  ~/precedent-space
#
# The Space gets the app and the ten RelMo modules the ranker actually imports,
# and no data at all. The corpus is in S3 and the memory is in CockroachDB, so
# the repo stays a few hundred kilobytes and pushes in seconds rather than
# carrying 2.35 GB through git-lfs on every deploy.
set -euo pipefail
OUT="${1:-$HOME/precedent-space}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# The exact transitive closure of `from relmo.vjzeval import PAD_COST, _pad` and
# `from relmo.vjmatch import dtw`, taken from sys.modules rather than guessed.
# All of it is numpy, which is why the two interpreters collapse to one here.
RELMO=(__init__ registry vjeval vjeval5 vjmatch vjrec4 vjrec5 vjrel vjs vjzeval)

mkdir -p "$OUT/showreel" "$OUT/native/relmo" "$OUT/space"
cp "$ROOT"/showreel/*.py "$ROOT"/showreel/*.html "$ROOT"/showreel/*.css \
   "$ROOT"/showreel/schema.sql "$OUT/showreel/"
rm -f "$OUT/showreel/dump_traces.py" "$OUT/showreel/migrate.py" \
      "$OUT/showreel/ingest.py" "$OUT/showreel/watch.py"
for m in "${RELMO[@]}"; do cp "$ROOT/native/relmo/$m.py" "$OUT/native/relmo/"; done
cp "$ROOT/showreel/space/requirements.txt" "$OUT/space/"
cp "$ROOT/showreel/space/Dockerfile"       "$OUT/Dockerfile"
cp "$ROOT/showreel/space/space_card.md"    "$OUT/README.md"
printf '__pycache__/\n*.pyc\n.env*\n' > "$OUT/.gitignore"

python3 - "$OUT" <<'PY'
import pathlib, sys
out = pathlib.Path(sys.argv[1])
n = sum(1 for _ in out.rglob("*.py"))
kb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024
print(f"\n{out}: {n} python files, {kb:.0f} KB total")
PY

cat <<EOF

Set these as Space SECRETS (Settings -> Variables and secrets):

  PRECEDENT_DSN          the CockroachDB connection string
  PRECEDENT_ENGINE       cockroach
  PRECEDENT_BUCKET       the S3 bucket name
  AWS_REGION             us-east-1
  AWS_ACCESS_KEY_ID      the precedent-space reader key, NOT precedent-deploy
  AWS_SECRET_ACCESS_KEY  its secret

Then:

  cd $OUT && git init && git remote add origin https://huggingface.co/spaces/<you>/precedent
  git add -A && git commit -m "precedent: agentic memory on CockroachDB and S3"
  git push -u origin main
EOF
