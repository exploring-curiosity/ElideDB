"""Read accounting and the no-cache policy.

Both are load-bearing for every number this project reports, and both
are the kind of machinery that fails silently: an instrument that
under-counts still prints a plausible figure, and a cache policy that
quietly stops applying still returns correct rows. So the properties
are tested rather than eyeballed once.

measure() monkey-patches two methods for the duration of a block. The
thing that would hurt is LEAKAGE - a patched method surviving the block
and folding a later query's bytes into a stale counter - so restoration
is tested on the exception path too.

corpus_bytes takes a MAX across folded stats, never a sum. It is a
denominator; summing denominators across the dozen tables a retrieval
query touches would inflate it until the elision figure flattered
itself, which is the same class of bug as the nested-column one that
made vector bytes vanish.
"""
import sys
import tempfile
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                              # noqa: E402
from elidedb.store import QueryStats, Table, _fold, _NOCACHE  # noqa: E402


def tiny_store():
    d = Path(tempfile.mkdtemp()) / "s"
    db = Store.create(d, "t")
    for name in ("a", "b"):
        db.table(name).append(pa.table({
            "ts": pa.array([1, 2, 3], pa.int64()),
            "v": pa.array([10, 20, 30], pa.int64())}))
    return db


def test_measure_accumulates_across_tables():
    db = tiny_store()
    with db.measure() as st:
        db.table("a").scan()
        db.table("b").scan()
    assert st.files_touched == 2
    assert st.bytes_touched > 0


def test_measure_corpus_is_the_whole_store_not_one_table():
    db = tiny_store()
    whole = sum(db.table(t).state().bytes for t in db.tables())
    with db.measure() as st:
        db.table("a").scan()
    assert st.corpus_bytes == whole, "denominator must be the store"


def test_measure_restores_methods():
    db = tiny_store()
    before = (Table.scan, Table.scan_values)
    with db.measure():
        pass
    assert (Table.scan, Table.scan_values) == before


def test_measure_restores_methods_after_an_exception():
    db = tiny_store()
    before = (Table.scan, Table.scan_values)
    try:
        with db.measure():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert (Table.scan, Table.scan_values) == before, "patch leaked"


def test_measure_still_honours_a_callers_own_stats():
    """A caller inside the block that passes its own QueryStats must
    still get its numbers - the block observes, it does not steal."""
    db = tiny_store()
    mine = QueryStats()
    with db.measure() as st:
        db.table("a").scan(stats=mine)
    assert mine.bytes_touched > 0
    assert st.bytes_touched == mine.bytes_touched


def test_fold_takes_max_of_corpus_not_sum():
    a, b = QueryStats(), QueryStats()
    a.corpus_bytes, b.corpus_bytes = 100, 40
    b.bytes_touched = 7
    _fold(a, b)
    assert a.corpus_bytes == 100, "summing denominators inflates elision"
    assert a.bytes_touched == 7


def test_elided_pct_never_exceeds_one_hundred():
    st = QueryStats()
    st.corpus_bytes, st.bytes_touched = 100, 250   # over-read is clamped
    assert st.elided_pct == 0.0


def test_nocache_is_the_default():
    """The buffer pool is deliberately unbuilt; until it exists the
    engine must not rest on the OS page cache, or a repeat read borrows
    work the benchmark never paid for."""
    assert _NOCACHE is True


def test_drop_caches_reports_and_clears():
    db = tiny_store()
    from elidedb import context
    context._CTX_CACHE["k"] = 1
    out = db.drop_caches()
    assert context._CTX_CACHE == {}
    assert out["context._CTX_CACHE"] == 1
    assert "F_NOCACHE" in out["file_pages"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}: {e}")
    print("all measure tests pass" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
