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


CHECKPOINT_EVERY = 10  # Delta checkpoints every 10th commit; same dial here


def _fsync_dir(path: Path):
    """Durability of a file's *existence* requires fsyncing its directory —
    the metadata write that makes the entry reachable after power loss."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_file(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CommitConflict(RuntimeError):
    """A concurrent commit removed files this transaction depended on."""


class TableLog:
    def __init__(self, table_dir: Path):
        self.dir = Path(table_dir)
        self.log_dir = self.dir / "_log"

    def versions(self) -> list[int]:
        if not self.log_dir.is_dir():
            return []
        return sorted(int(p.stem) for p in self.log_dir.glob("*.json")
                      if p.stem.isdigit())

    def _checkpoints(self) -> list[int]:
        if not self.log_dir.is_dir():
            return []
        return sorted(int(p.name.split(".")[0])
                      for p in self.log_dir.glob("*.checkpoint.json"))

    def read_state(self, version: int | None = None) -> TableState:
        st = TableState()
        start = 0
        # Checkpoints make the fold O(commits since checkpoint) instead of
        # O(all commits) — the log stays an audit trail without becoming a
        # read cost. Same move as Delta's _last_checkpoint.
        for v in reversed(self._checkpoints()):
            if version is None or v <= version:
                c = json.loads(
                    (self.log_dir / f"{v:020d}.checkpoint.json").read_text())
                st.version = c["version"]
                st.kind = c["kind"]
                st.schema = c["schema"]
                st.meta = dict(c["meta"])
                st.files = [FileEntry.from_json(f) for f in c["files"]]
                start = v
                break
        for v in self.versions():
            if v <= start:
                continue
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
               meta: dict | None = None, retries: int = 5) -> int:
        """Optimistic concurrency, Delta-style: O_EXCL on the next log entry
        is the lock. Losing the race means retrying at the next version —
        append-vs-append never truly conflicts. Commits that REMOVE files
        revalidate against the fresh state first: if a concurrent writer
        already removed one of ours, that is a real conflict and we fail
        cleanly instead of double-applying."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(retries):
            version = (self.versions() or [0])[-1] + 1
            if remove:
                active = {f.path for f in self.read_state().files}
                missing = [r for r in remove if r not in active]
                if missing:
                    raise CommitConflict(
                        "files no longer active (concurrent rewrite?): "
                        f"{missing[:3]}")
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
            # Torn-write-proof commit: the entry is fully written and fsynced
            # under a temp name, then hard-LINKED to its final name. link()
            # is atomic AND exclusive (fails if the name exists), so a
            # partially-written entry can never appear under a version name —
            # a crash leaves only an ignorable *.tmp. Atomicity + durability
            # in one primitive; fsync of the directory makes the rename
            # itself survive power loss.
            tmp = self.log_dir / f".{version:020d}.{os.getpid()}.tmp"
            fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(entry, indent=1))
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                os.unlink(tmp)
                continue  # lost the race — re-read state, take the next slot
            os.unlink(tmp)
            _fsync_dir(self.log_dir)
            if version % CHECKPOINT_EVERY == 0:
                st = self.read_state(version)
                cp_tmp = self.log_dir / f".cp{version}.{os.getpid()}.tmp"
                cp_tmp.write_text(
                    json.dumps({"version": st.version, "kind": st.kind,
                                "schema": st.schema, "meta": st.meta,
                                "files": [f.to_json() for f in st.files]}))
                os.replace(cp_tmp,  # checkpoints are derived: replace is fine
                           self.log_dir / f"{version:020d}.checkpoint.json")
            return version
        raise CommitConflict(f"lost the commit race {retries} times")

    def history(self) -> list[dict]:
        out = []
        for v in self.versions():
            e = json.loads((self.log_dir / f"{v:020d}.json").read_text())
            out.append({"version": v, "ts_utc": e["ts_utc"], "op": e["op"],
                        "added_files": len(e.get("add", [])),
                        "added_rows": sum(a["rows"] for a in e.get("add", [])),
                        "meta": e.get("meta", {})})
        return out
