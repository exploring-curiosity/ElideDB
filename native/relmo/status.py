"""One-command status of the autonomous loop.

Answers, in order, the only questions worth asking between checks:
  is anything actually RUNNING, and how far along
  is the DATA growing, and does its tracker still pass validation
  are the MODELS training, and against what baseline
  what BROKE

Deliberately reads artifacts, not exit codes - a job can exit 0 having
written nothing, so progress is counted from files on disk and from
the daemon pid files, never from "it said it finished".

    python -m relmo.status            human
    python -m relmo.status --json     machine (for the watcher)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import daemon as D  # noqa: E402

_PCT = re.compile(r"(\d+)%\|.*?\| *(\d+)/(\d+) \[([\d:]+)<([\d:?]+)")


def tail_progress(log: Path):
    """Last tqdm state from a daemon log: (done, total, eta)."""
    if not log.exists():
        return None
    txt = log.read_bytes()[-4000:].decode("utf8", "replace")
    hits = _PCT.findall(txt.replace("\r", "\n"))
    if not hits:
        return None
    _, done, total, el, eta = hits[-1]
    return dict(done=int(done), total=int(total), elapsed=el, eta=eta)


def jobs():
    out = []
    if not D.LOGS.exists():
        return out
    for pidf in sorted(D.LOGS.glob("*.pid")):
        tag = pidf.stem
        log = D.LOGS / f"{tag}.log"
        out.append(dict(job=tag, pid=D.alive(tag),
                        running=D.alive(tag) is not None,
                        progress=tail_progress(log),
                        log_age_s=int(time.time() - log.stat().st_mtime)
                        if log.exists() else None))
    return out


def datasets():
    """Every dataset that EXISTS, not just every one already tracked -
    a corpus mid-import has no track dir yet, and a status view that
    skips it is blind to the job actually running."""
    out = []
    names = sorted({d.name for d in R.DATASETS.glob("*") if d.is_dir()}
                   | {d.name for d in R.TRACKS.glob("*") if d.is_dir()})
    for name in names:
        d = R.TRACKS / name
        man = R.read_manifest(name)
        tvs = sorted(d.glob("trackval_*.json"),
                     key=lambda p: p.stat().st_mtime)
        tv = tvs[-1] if tvs else d / "_none_"
        cfg = man.get("config", {})
        rec = dict(dataset=name,
                   episodes=man.get("n_episodes",
                                    len(man.get("episodes", []))),
                   tracks=len(list(d.glob("*.npz"))) if d.exists() else 0,
                   fingerprint=man.get("fingerprint"),
                   kind=("scanned" if cfg.get("scanned_assets") else
                         "textured" if cfg.get("textured") else "flat")
                   + ("+cam" if cfg.get("moving_camera") else ""))
        if tv.exists():
            t = json.loads(tv.read_text())
            rec["trackval"] = dict(
                tracker=t.get("tracker"), split=t.get("split", "all"),
                epe_moving_med=t["epe_px"]["moving"]["med"],
                jump_frac_med=t["jump_frac"]["moving"]["med"],
                vel_snr=t["vel_snr_at_motion"])
        out.append(rec)
    return out


def models():
    out = []
    for d in sorted(R.MODELS.glob("*")):
        m = d / "metrics.jsonl"
        if not m.exists():
            continue
        rows = [json.loads(x) for x in m.read_text().splitlines()
                if x.strip()]
        if not rows:
            continue
        last = rows[-1]
        # runs report different metrics (WM: motion-R2/AoT; tracker
        # finetune: position L1) - show whatever the run actually
        # logged rather than hardcoding one trainer's schema
        keys = [k for k in last
                if k not in ("run", "step", "n_train", "n_val",
                             "min_per_1k")]
        ev = d / "evals.jsonl"
        gate = None
        if ev.exists():
            er = [json.loads(x) for x in ev.read_text().splitlines()
                  if x.strip()]
            if er:
                gate = er[-1]
        out.append(dict(run=d.name, step=last.get("step"),
                        metrics={k: last[k] for k in keys[:4]},
                        gate=gate,
                        n_ckpt=len(list(d.glob("ckpt_*.pt"))),
                        updated_s=int(time.time() - m.stat().st_mtime)))
    return out


def collect():
    led = []
    if R.LEDGER.exists():
        led = [json.loads(x) for x in R.LEDGER.read_text().splitlines()
               if x.strip()]
    # errors that matter are RECENT ones on a LIVE dataset - the
    # ledger is append-only and carries 35k stale gen_errors from a
    # retired generator, which would read as an emergency forever
    errs = [r for r in led if str(r.get("kind", "")).endswith("error")]
    live = {d["dataset"] for d in datasets()
            if d["dataset"] not in ("physgen_v1",)}
    recent = [r for r in errs[-4000:] if r.get("dataset") in live]
    return dict(jobs=jobs(), datasets=datasets(), models=models(),
                errors_total=len(errs), errors_live=len(recent),
                last_error=recent[-1].get("error", "")[-160:]
                if recent else None)


def render(s):
    L = []
    L.append("JOBS")
    if not s["jobs"]:
        L.append("  (none)")
    for j in s["jobs"]:
        p = j["progress"]
        bar = (f"{p['done']}/{p['total']} eta {p['eta']}"
               if p else "no progress line")
        L.append(f"  {'RUN ' if j['running'] else 'DEAD'} {j['job']:<10s}"
                 f" pid {j['pid'] or '-':<7} {bar}"
                 f"  (log {j['log_age_s']}s ago)")
    L.append("\nDATA")
    for d in s["datasets"]:
        tv = d.get("trackval")
        tvs = (f"  trackval[{tv['split']}] {tv['tracker']}: "
               f"epe {tv['epe_moving_med']}px  "
               f"jump {tv['jump_frac_med']}  snr {tv['vel_snr']}x"
               if tv else "  trackval: NOT RUN")
        L.append(f"  {d['dataset']:<12s} eps {d['episodes']:>5d}  "
                 f"tracks {d['tracks']:>5d}  {d['kind']:<12s} "
                 f"{d['fingerprint']}")
        L.append(f"  {'':<12s}{tvs}")
    L.append("\nMODELS")
    if not s["models"]:
        L.append("  (none)")
    for m in s["models"]:
        met = "  ".join(f"{k} {v}" for k, v in m["metrics"].items())
        L.append(f"  {m['run']:<10s} step {m['step']:>7}  {met}  "
                 f"ckpts {m['n_ckpt']}  ({m['updated_s']}s ago)")
        g = m.get("gate")
        if g and "promote" in g:
            L.append(f"  {'':<10s} gate@{g['step']}: "
                     f"{'PROMOTED' if g['promote'] else 'rejected'}  "
                     f"moving {g.get('fz_mov')}->{g.get('ft_mov')}  "
                     f"bg {g.get('fz_bg')}->{g.get('ft_bg')}")
    L.append(f"\nERRORS live {s['errors_live']}"
             f"  (total incl. retired datasets {s['errors_total']})"
             + (f"\n  last: {s['last_error']}" if s["last_error"] else ""))
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    st = collect()
    print(json.dumps(st, indent=1) if a.json else render(st))
