#!/bin/sh
# First start fetches what the image does not carry: the InternVideo2
# assembly (weights from the public ziyjiang mirror) and the WordNet
# corpus. Both cache under /app and are skipped when present.
set -e
cd /app
python - <<'PY'
import nltk
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
PY
python scripts/get_iv2.py
exec python -c "import sys; sys.argv=['desk']; from elidedb.desk import main; main()"
