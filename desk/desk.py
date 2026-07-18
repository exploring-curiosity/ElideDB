#!/usr/bin/env python3
"""Compatibility shim: Desk now lives inside the elidedb package."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from elidedb.desk import main
if __name__ == "__main__":
    main()
