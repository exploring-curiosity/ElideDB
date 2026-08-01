"""Shared helpers for benchmark ENTRY POINTS. Not a program.

The leading underscore is the contract: this file has no `main()` and is
never run. Seven scripts used to `from bench_product import QUERIES`,
which made a runnable benchmark into a library - importing it executed
its module body and coupled every caller to its CLI. Anything with a
`main()` is a program; anything imported is a module; nothing is both.

The queries themselves are DATA and live in `eval/queries.json`, beside
the truthset. They are dataset-specific eval content, so they belong
neither in a script nor in the `elidedb` package - putting them in the
engine would be embedding the evaluation into the thing evaluated.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def queries():
    """The eval query set, in fixed order (index == query id)."""
    return list(json.loads((ROOT / "eval/queries.json").read_text())["queries"])
