"""Set-retrieval bench: yield, TRUE purity (eval labels), latency."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pyarrow.parquet as pq
from elidedb import Store
from elidedb.scenario import search_set
from regress10 import QUERIES

def main():
    db = Store.open("lake/bridge4h")
    t = pq.read_table("eval/bridge4h_truth.parquet").to_pydict()
    epm = db.table("episodes").scan()
    stream_of = dict(zip(epm.column("episode_index").to_pylist(), epm.column("stream").to_pylist()))
    lab = {}
    for i, a, b, k in zip(t["episode_index"], t["ts"], t["t1"], t["task"]):
        if k: lab[(stream_of.get(int(i)), int(a))] = k.lower()
    mode = sys.argv[1] if len(sys.argv) > 1 else "audited"
    tot_del = tot_ok = 0
    lats = []
    for q, pred in QUERIES:
        r = search_set(db, q, purity=mode)
        labs = [lab.get((c["stream"], c["t0"]), "") for c in r["clips"]]
        ok = sum(pred(l) for l in labs)
        tot_del += len(labs); tot_ok += ok
        lats.append(r["ms"])
        pu = 100 * ok / max(len(labs), 1)
        print(f"  {len(labs):3d} delivered, purity {pu:3.0f}%"
              f"{'  est ' + str(r['est_purity']) if r['est_purity'] is not None else ''}"
              f"  {r['ms']:6.0f}ms  {q}")
    print(f"\n== {mode}: {tot_del} clips delivered, TRUE purity "
          f"{100 * tot_ok / max(tot_del, 1):.0f}%, p50 latency "
          f"{sorted(lats)[len(lats)//2]:.0f}ms ==")

if __name__ == "__main__":
    main()
