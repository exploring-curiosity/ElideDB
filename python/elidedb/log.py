"""Transaction log: the Delta-Lake idea at StreetDex scale.

A table's state is not "the files in the directory" — it is the fold of an
append-only log of JSON commits. That one move buys, with plain files:

- **snapshot isolation / time travel**: version N is immutable forever; a
  reader at version N never sees version N+1's files.
- **atomic multi-file commits**: a commit lands as one exclusively-created
  log entry (O_EXCL is the lock — single writer, many readers).
- **file-level zone maps**: every added file records rows/bytes/min_ts/max_ts,
  so a time-window query prunes whole files from the log alone, before any
  Parquet footer is opened. (Row-group pruning inside surviving files is
  Parquet's own statistics — two layers, same idea.)
- **schema-on-log**: the schema travels with the commit, so evolution is an
  append, never a rewrite.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileEntry:
    path: str  # relative to the table dir
    rows: int
    bytes: int
    min_ts: int
    max_ts: int

    def to_json(self):
        return self.__dict__

    @staticmethod
    def from_json(d):
        return FileEntry(d["path"], d["rows"], d["bytes"], d["min_ts"], d["max_ts"])


@dataclass
class TableState:
    version: int = 0
    kind: str = "timeseries"
    schema: str = ""
    files: list[FileEntry] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def rows(self):
        return sum(f.rows for f in self.files)

    @property
    def bytes(self):
        return sum(f.bytes for f in self.files)

    @property
    def min_ts(self):
        return min((f.min_ts for f in self.files), default=0)

    @property
    def max_ts(self):
        return max((f.max_ts for f in self.files), default=0)


class TableLog:
    def __init__(self, table_dir: Path):
        self.dir = Path(table_dir)
        self.log_dir = self.dir / "_log"

    def versions(self) -> list[int]:
        if not self.log_dir.is_dir():
            return []
        return sorted(int(p.stem) for p in self.log_dir.glob("*.json"))

    def read_state(self, version: int | None = None) -> TableState:
        st = TableState()
        for v in self.versions():
            if version is not None and v > version:
                break
            entry = json.loads((self.log_dir / f"{v:020d}.json").read_text())
            st.version = v
            st.kind = entry.get("table_kind", st.kind)
            st.schema = entry.get("schema", st.schema)
            st.meta.update(entry.get("meta", {}))
            removed = set(entry.get("remove", []))
            if removed:
                st.files = [f for f in st.files if f.path not in removed]
            st.files += [FileEntry.from_json(f) for f in entry.get("add", [])]
        return st

    def commit(self, *, op: str, kind: str, schema: str = "",
               add: list[FileEntry] = (), remove: list[str] = (),
               meta: dict | None = None) -> int:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        version = (self.versions() or [0])[-1] + 1
        entry = {
            "version": version,
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "op": op,
            "table_kind": kind,
            "schema": schema,
            "add": [f.to_json() for f in add],
            "remove": list(remove),
            "meta": meta or {},
        }
        path = self.log_dir / f"{version:020d}.json"
        # O_EXCL: two writers racing on the same version — one wins, one gets
        # a clean error instead of a corrupted table. This IS the transaction.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(entry, indent=1))
        return version

    def history(self) -> list[dict]:
        out = []
        for v in self.versions():
            e = json.loads((self.log_dir / f"{v:020d}.json").read_text())
            out.append({"version": v, "ts_utc": e["ts_utc"], "op": e["op"],
                        "added_files": len(e.get("add", [])),
                        "added_rows": sum(a["rows"] for a in e.get("add", [])),
                        "meta": e.get("meta", {})})
        return out
