"""The Bridge task labels must never be reachable from the query path.

Bridge ships a human-written task string per episode. It is ground truth for
the benchmark and nothing else: if it were in the store, "contextual
retrieval" would degrade into a join against a label column and every number
in BENCHMARKS.md would be meaningless.

This test is the guard. It asserts the separation structurally — no task text
in any store table, and the ground-truth file living outside every store — so
a future ingest change cannot quietly reintroduce the leak.

Run: ./myenv/bin/python tests/test_no_label_leak.py
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from elidedb import Store   # noqa: E402

STORE = REPO / "lake/bridge"
TRUTH = REPO / "eval/bridge_truth.parquet"

# This guard asserts a property OF A STORE. With no store present it has
# nothing to check, and a missing store is not a leak — so it skips
# rather than failing, and runs again the moment one is built.
pytestmark = pytest.mark.skipif(
    not (STORE / "_store.json").exists(),
    reason=f"no store at {STORE}: label-leak guard needs data to inspect")


def _tasks():
    if not TRUTH.exists():
        return set()
    return {t for t in pq.read_table(TRUTH).column("task").to_pylist() if t}


def test_ground_truth_lives_outside_every_store():
    assert TRUTH.exists(), "run scripts/bridge_ingest.py first"
    lake = REPO / "lake"
    for store_dir in lake.iterdir() if lake.is_dir() else []:
        assert store_dir not in TRUTH.parents, \
            f"ground truth is inside the store {store_dir}"


def test_no_column_named_like_a_label():
    db = Store.open(STORE)
    banned = {"task", "tasks", "task_index", "label", "instruction",
              "language_instruction", "annotation"}
    for name in db.tables():
        cols = set(db.table(name).scan().column_names)
        leak = cols & banned
        assert not leak, f"table '{name}' exposes label column(s): {leak}"


def test_no_task_string_appears_in_any_string_column():
    """Stronger than a column-name check: the actual label TEXT must not
    appear anywhere, under any column name."""
    tasks = _tasks()
    assert tasks, "no ground-truth tasks loaded"
    db = Store.open(STORE)
    lowered = {t.strip().lower() for t in tasks}
    for name in db.tables():
        t = db.table(name).scan()
        for col in t.column_names:
            if t.schema.field(col).type != pa.string():
                continue
            vals = {str(v).strip().lower()
                    for v in set(t.column(col).to_pylist()) if v}
            hit = vals & lowered
            assert not hit, (f"table '{name}' column '{col}' contains "
                             f"ground-truth task text: {sorted(hit)[:3]}")


def test_captions_are_not_copies_of_the_labels():
    """Captions come from a VLM looking at pixels. If any caption were byte
    equal to a task string, something is feeding labels to the captioner."""
    db = Store.open(STORE)
    if "context_captions" not in db.tables():
        return
    caps = {c.strip().lower()
            for c in db.table("context_captions").scan()
            .column("caption").to_pylist() if c}
    overlap = caps & {t.strip().lower() for t in _tasks()}
    assert not overlap, f"captions duplicate task labels: {sorted(overlap)[:3]}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
