"""Pruning layers: a skip is only allowed when it is PROVABLY empty.

Pruning is the one class of optimisation that can silently return wrong
answers - a dropped file or row group is not slow, it is missing. So
the properties tested here are correctness properties, not performance
ones.

  may_contain      must answer True whenever it cannot prove empty,
                   including when the statistic is absent. Every file
                   written before zone maps existed has no zone map, and
                   those must degrade to "read it", never to "skip it".
  numeric order    statistics must be compared in the COLUMN's order.
                   scan_values used to stringify both sides, and "51" <
                   "7" lexically, so a lookup for object 7 would skip
                   the group holding it. Wrong answer, no error.
  round trip       a zone map has to survive the log's JSON, and its
                   absence has to stay absent rather than becoming a
                   null that a reader might compare against.

Run: python tests/test_prune.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from elidedb.log import FileEntry              # noqa: E402


def entry(zone=None):
    return FileEntry("p.parquet", 10, 100, 0, 99, zone or {})


def test_absent_zone_map_means_unknown_not_empty():
    assert entry().may_contain("object_id", 5, 5) is True


def test_absent_column_in_a_present_zone_map_is_unknown():
    e = entry({"value": ["a", "b"]})
    assert e.may_contain("object_id", 5, 5) is True


def test_disjoint_range_is_prunable():
    e = entry({"object_id": [10, 20]})
    assert e.may_contain("object_id", 21, 30) is False
    assert e.may_contain("object_id", 0, 9) is False


def test_overlapping_range_is_kept():
    e = entry({"object_id": [10, 20]})
    for lo, hi in ((10, 10), (20, 20), (0, 10), (20, 99), (5, 25)):
        assert e.may_contain("object_id", lo, hi), (lo, hi)


def test_numeric_zone_map_does_not_use_string_order():
    """The bug this guards: "51" < "7" lexically. A file holding
    object_id 51..131 must NOT be pruned for a lookup of object 7 by
    string comparison, and must be pruned by numeric comparison."""
    e = entry({"object_id": [51, 131]})
    assert e.may_contain("object_id", 7, 7) is False
    assert e.may_contain("object_id", 60, 60) is True
    # and the string-ordered answer would have been the opposite
    assert not (str(7) < str(51))


def test_string_zone_map_still_works():
    e = entry({"value": ["apple", "melon"]})
    assert e.may_contain("value", "banana", "banana") is True
    assert e.may_contain("value", "zebra", "zebra") is False


def test_zone_map_round_trips_through_json():
    e = entry({"object_id": [3, 9]})
    assert FileEntry.from_json(e.to_json()).zone == {"object_id": [3, 9]}


def test_empty_zone_map_is_omitted_from_json():
    """Old readers must see a byte-for-byte familiar entry."""
    assert "zone" not in entry().to_json()
    assert FileEntry.from_json({"path": "p", "rows": 1, "bytes": 1,
                                "min_ts": 0, "max_ts": 1}).zone == {}


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
    print("all prune tests pass" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
