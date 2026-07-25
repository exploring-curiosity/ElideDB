"""Append truthset verdicts for one sheet: grade.py s0000.png 1,0,0,1
(order = rows top to bottom; count must match the manifest rows)."""
import json
import sys
from pathlib import Path

OUT = Path("eval/truthsets")
sheet, votes = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
manifest = json.loads((OUT / "sheets/manifest.json").read_text())
rows = manifest[sheet]
assert len(votes) == len(rows), f"{len(rows)} rows, {len(votes)} votes"
with (OUT / "verdicts.jsonl").open("a") as f:
    for (qi, s, a, b), v in zip(rows, votes):
        f.write(json.dumps({"q": qi, "stream": s, "t0": a,
                            "v": v}) + "\n")
print(f"{sheet}: {sum(votes)}/{len(votes)} true")
