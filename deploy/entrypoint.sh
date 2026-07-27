#!/bin/sh
# First start fetches what the image does not carry: the InternVideo2
# assembly, the WordNet corpus, and every text tower checkpoint. All
# of it happens BEFORE the server accepts traffic and caches under
# /app, so no user query ever pays a download.
set -e
cd /app
python scripts/get_iv2.py
python deploy/warm.py
exec python -c "import sys; sys.argv=['desk']; from elidedb.desk import main; main()"
