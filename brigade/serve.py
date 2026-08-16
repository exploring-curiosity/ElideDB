#!/usr/bin/env python3
"""Launcher: `python3 brigade/serve.py` from the repo root.

Running this file puts its own directory (…/brigade) at the head of sys.path, so
`import brigade` resolves to the package beside it with no PYTHONPATH juggling.
"""

import sys

from brigade.run import main

if __name__ == "__main__":
    sys.exit(main())
