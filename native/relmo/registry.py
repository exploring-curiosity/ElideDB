"""Versioning + ledger for the RelMo loop.

MLOps rules enforced here:
  - datasets are IMMUTABLE and versioned; a shard is written to a temp
    name and renamed only when complete, so a crash never leaves a
    half-shard that later looks finished;
  - every dataset carries a manifest: generator version, config, seed
    range, per-episode stats, and a content fingerprint;
  - every training run records the dataset fingerprint it consumed,
    its own config hash, and its git commit - a checkpoint that
    cannot name its data is not a checkpoint;
  - the ledger is append-only JSONL: every generation batch, every
    train step milestone, every eval. Nothing is ever overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "relmo"
DATASETS = BASE / "datasets"
TRACKS = BASE / "tracks"
MODELS = BASE / "models"
LEDGER = BASE / "ledger.jsonl"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True).stdout.strip() or "?"
    except Exception:
        return "?"


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def log(kind: str, **fields):
    """Append-only ledger. Every event in the loop lands here."""
    BASE.mkdir(parents=True, exist_ok=True)
    rec = dict(ts=now(), kind=kind, commit=git_commit(), **fields)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def dataset_dir(name: str) -> Path:
    return DATASETS / name


def read_manifest(name: str) -> dict:
    f = dataset_dir(name) / "manifest.json"
    if not f.exists():
        return dict(name=name, episodes=[], gen_version=None, config={})
    return json.loads(f.read_text())


def write_manifest(name: str, man: dict):
    d = dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    man["fingerprint"] = hashlib.sha256(
        json.dumps(sorted(e["id"] for e in man["episodes"])).encode()
    ).hexdigest()[:12]
    man["n_episodes"] = len(man["episodes"])
    man["updated"] = now()
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, d / "manifest.json")
    return man


def model_dir(run_id: str) -> Path:
    p = MODELS / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def best_pointer(run_id: str, step: int, metric: float, name: str):
    """Promote a checkpoint. Kept as a small json rather than a
    symlink so it survives copying the tree between machines."""
    (model_dir(run_id) / "best.json").write_text(json.dumps(
        dict(step=step, metric=metric, metric_name=name,
             ckpt=f"ckpt_{step:07d}.pt", updated=now()), indent=1))
