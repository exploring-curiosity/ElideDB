"""Boot-time warm for the demo service: fetch and load every text
tower BEFORE the server accepts traffic, so no user query ever pays a
download. Idempotent; cached weights make later boots fast."""
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
        print(f"warm {tag} FAILED: {type(e).__name__}: {e}",
              flush=True)


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
    print("warm complete", flush=True)


if __name__ == "__main__":
    main()
