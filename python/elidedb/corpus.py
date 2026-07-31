"""S5: dataset LAYOUT as configuration, not as constants in six scripts.

`CAM = "observation.images.image_0"`, `EPOCH_NS`, `FILE_STRIDE_NS` and
`FPS = 5.0` were copied into bridge_ingest, bridge4h, bridge_full,
rebuild_bench and full_write. None of them is a task prior - they
describe where the bytes are, not what the bytes mean - but six copies
of a dataset's shape is still the dataset hardwired into the engine, and
pointing ElideDB at a driving log meant editing six files.

A corpus descriptor is data. It lives beside the raw files, it is read,
and the engine has no opinion about its contents.

    {
      "name": "bridge",
      "root": "data/bridge",
      "streams": [{"id": "observation.images.image_0",
                   "path": "videos/{id}/chunk-000/file-{file:03d}.mp4",
                   "fps": 5.0}],
      "epoch_ns": 1704067200000000000,
      "file_stride_ns": 20000000000000,
      "episodes": {"table": "meta/episodes/chunk-000/file-000.parquet",
                   "index": "episode_index", "length": "length"},
      "exclude": ["tasks"]
    }

`exclude` is the important field and the reason this is not merely
tidiness: it is where a corpus declares which of its own columns are
ANNOTATION rather than observation. Bridge's `tasks` column says what
each clip is about, which is precisely what the database is supposed to
work out from pixels; ingesting it would turn every retrieval number
into a join against a label. bridge_ingest.py enforced that by hand, in
a docstring. Here it is a field the ingest path honours for any corpus,
so the next dataset cannot forget.
"""
from __future__ import annotations

import json
from pathlib import Path

# Fields every corpus must supply. No defaults for the ones that encode
# the dataset's shape: guessing an epoch or a frame rate silently
# misaligns every timestamp in the store, and a loud failure is cheaper.
REQUIRED = ("name", "root", "streams")


class Corpus:
    """A dataset's shape, read from disk. Never inferred, never guessed."""

    def __init__(self, d: dict, path=None):
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            raise ValueError(f"corpus descriptor missing {missing}")
        self.d, self.path = d, path
        self.name = d["name"]
        self.root = Path(d["root"])
        self.streams = d["streams"]
        self.epoch_ns = int(d.get("epoch_ns", 0))
        self.file_stride_ns = int(d.get("file_stride_ns", 0))
        # columns the corpus declares as ANNOTATION - never ingested
        self.exclude = set(d.get("exclude", ()))

    @staticmethod
    def load(path):
        p = Path(path)
        if p.is_dir():
            p = p / "corpus.json"
        return Corpus(json.loads(p.read_text()), p)

    def stream(self, sid=None):
        if sid is None:
            return self.streams[0]
        for s in self.streams:
            if s["id"] == sid:
                return s
        raise KeyError(f"no stream {sid} in corpus {self.name}")

    def fps(self, sid=None):
        return float(self.stream(sid).get("fps", 0)) or None

    def media(self, file_index, sid=None):
        s = self.stream(sid)
        return self.root / s["path"].format(id=s["id"], file=int(file_index))

    def base_ns(self, file_index):
        """Timeline origin for a file. 0/0 means the corpus carries real
        timestamps and no synthetic clock is being imposed."""
        return self.epoch_ns + int(file_index) * self.file_stride_ns

    def is_annotation(self, column: str) -> bool:
        """True for columns the corpus declares as annotation.

        The ingest path must refuse these. A column that says what a clip
        is ABOUT is the answer to the question the database exists to
        answer, and ingesting it makes every later number a lookup.
        """
        return column in self.exclude
