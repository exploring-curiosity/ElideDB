"""Warm THEN serve, in one process.

The first deployment warmed the towers in a separate process, whose
loaded models died with it; the server then re-loaded everything on
the first user query and also paid the per-store first-touch builds
(mmap sidecars, action class vectors). On two cloud vCPUs that made
the first query take many minutes. This process loads every tower,
runs throwaway queries that exercise every channel and build every
per-store cache, and only then binds the port. When the platform
reports RUNNING, queries are genuinely warm.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "python")


def step(tag, fn):
    t0 = time.time()
    try:
        fn()
        print(f"warm {tag}: {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        print(f"warm {tag} FAILED: {type(e).__name__}: {e}", flush=True)


def main():
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    step("pe", lambda: __import__(
        "elidedb.pe", fromlist=["_text_vec"])._text_vec("warm"))
    step("sig2", lambda: __import__(
        "elidedb.sig2", fromlist=["_text_vec"])._text_vec("warm"))
    step("iv2", lambda: __import__(
        "elidedb.iv2", fromlist=["text_vec"]).text_vec("warm"))
    step("xclip", lambda: __import__(
        "elidedb.vid", fromlist=["_text_vec"])._text_vec("warm"))
    step("siglip", lambda: __import__(
        "elidedb.embeddings", fromlist=["embed_text"]).embed_text(
            "warm"))

    # throwaway queries: one directional, one two-atom descriptive,
    # per store. Together they touch every channel, build the mmap
    # sidecars, the action class vectors, and the vocabulary cache,
    # which are the per-store first-touch costs a user must never pay.
    from elidedb import desk as D
    D.discover()
    from elidedb.scenario import search_set
    for key, db in list(D.STORES.items()):
        step(f"store {key} dir", lambda db=db: search_set(
            db, "the robot arm closes the drawer", purity="fast",
            k_max=5))
        step(f"store {key} con", lambda db=db: search_set(
            db, "put the object on the surface", purity="fast",
            k_max=5))

    print("warm complete, serving", flush=True)
    sys.argv = ["desk"]
    D.main()


if __name__ == "__main__":
    main()
