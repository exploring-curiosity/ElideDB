#!/bin/zsh
# ElideDB Desk — QbE (the vision-native stores).
#
# Kept SEPARATE from the lake Desk on purpose. They serve different store
# formats: the lake Desk opens elidedb.Store (tables, streams, schema),
# this one opens stores/vision (window rows + a memory-mapped channel
# column). One launcher cannot serve both, and pointing the existing .app
# here would silently change what a familiar icon opens.
#
# Restart-on-stale, same reason as the lake launcher: Python caches
# modules at import, so a server left running from an older checkout
# serves code that is no longer in the tree.
REPO="/Users/sudharshanramesh/Studies/MyProjects/StreetDex"
cd "$REPO" || exit 1
PY="$(command -v python3 || echo /usr/bin/python3)"
[[ -x "$REPO/myenv/bin/python" ]] && PY="$REPO/myenv/bin/python"

WANT=$("$PY" -c "import sys;sys.path.insert(0,'$REPO/native');import vdesk;print(vdesk.build_id())" 2>/dev/null)
HAVE=$(curl -s --max-time 2 http://127.0.0.1:8788/api/version 2>/dev/null | sed -n 's/.*"build"[^"]*"\([^"]*\)".*/\1/p')
if [[ -n "$HAVE" && "$HAVE" != "$WANT" ]]; then
  echo "vdesk: running build $HAVE != source $WANT — restarting" >&2
  lsof -ti :8788 | xargs kill -9 2>/dev/null; sleep 1; HAVE=""
fi
if [[ -z "$HAVE" ]]; then
  nohup "$PY" "$REPO/native/vdesk.py" --port 8788 >/tmp/elidedb_vdesk.log 2>&1 &
  for i in {1..60}; do sleep 0.25
    curl -s --max-time 1 http://127.0.0.1:8788/api/version >/dev/null 2>&1 && break
  done
fi
open "http://localhost:8788"
