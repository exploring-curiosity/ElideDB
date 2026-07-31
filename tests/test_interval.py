"""Interval overlap. The bug class here returns WRONG ANSWERS, quietly.

Raw capture is a continuous stream; episodes are something the engine
must produce, not something it is given. So the query unit is an
interval, and presence is run-length encoded: "this object was there
from t0 to t1" is ONE row whether that is 200 ms or four hours.

Three layers all asked the wrong question, and all three had to be
fixed together:

    commit log       max_ts is max(START), so a file holding an interval
                     that began before the window and had not ended was
                     pruned entirely
    row-group stats  same comparison, one layer down
    reader predicate `ts >= t0` asks "did it START inside the window",
                     which is a different question from "was it there"

Verified before the fix: an object present 100..900, queried [400, 500],
returned ZERO rows. Every table until now was point-like or short-lived
so it never fired - which is exactly why it needs a test rather than a
memory of having fixed it.

Run: python tests/test_interval.py
"""
import sys
import tempfile
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb import Store                                # noqa: E402
from elidedb.log import FileEntry                        # noqa: E402


def store_with_intervals():
    d = Path(tempfile.mkdtemp()) / "s"
    db = Store.create(d, "t")
    # obj 1 spans the whole timeline; obj 2 is brief and late
    db.table("iv").append(pa.table({
        "ts": pa.array([100, 200], pa.int64()),
        "t1": pa.array([900, 250], pa.int64()),
        "obj": pa.array([1, 2], pa.int64())}))
    return db


def ids(tb):
    return sorted(tb.column("obj").to_pylist()) if len(tb) else []


def test_long_interval_found_from_inside():
    """The regression. Window sits entirely inside a live interval and
    touches no interval START at all."""
    db = store_with_intervals()
    assert ids(db.table("iv").scan(t0=400, t1=500)) == [1]


def test_overlap_returns_both():
    db = store_with_intervals()
    assert ids(db.table("iv").scan(t0=210, t1=240)) == [1, 2]


def test_window_after_everything_is_empty():
    db = store_with_intervals()
    assert ids(db.table("iv").scan(t0=950, t1=999)) == []


def test_open_ended_window():
    db = store_with_intervals()
    assert ids(db.table("iv").scan(t1=150)) == [1]


def test_point_table_unchanged():
    """A table with no t1 is point-like and must behave exactly as
    before - the fix must not widen ordinary time-range scans."""
    d = Path(tempfile.mkdtemp()) / "s"
    db = Store.create(d, "t")
    db.table("pt").append(pa.table({
        "ts": pa.array([10, 20, 30], pa.int64()),
        "v": pa.array([1, 2, 3], pa.int64())}))
    assert db.table("pt").scan(t0=15, t1=25).column("v").to_pylist() == [2]


def test_max_end_recorded_in_the_log():
    db = store_with_intervals()
    f = db.table("iv").state().files[-1]
    assert f.max_ts == 200, "max_ts is still max(start), by definition"
    assert f.max_end == 900, "max_end must be max(t1)"


def test_overlaps_falls_back_when_max_end_absent():
    """Files written before max_end existed carry 0. They must fall back
    to max_ts rather than be treated as ending at time zero, which would
    prune every one of them."""
    f = FileEntry("p", 1, 1, 100, 200, {}, 0)
    assert f.overlaps(150, 250) is True
    assert f.overlaps(0, 50) is False
    assert f.overlaps(300, 400) is False


def test_overlaps_uses_end_not_start():
    f = FileEntry("p", 1, 1, 100, 200, {}, 900)
    assert f.overlaps(400, 500) is True, "must not prune on max(start)"
    assert f.overlaps(950, 999) is False


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
    print("all interval tests pass" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
