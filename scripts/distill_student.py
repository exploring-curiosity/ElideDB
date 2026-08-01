"""Distil teacher_v1 into a two-tower student. Retrieval becomes a dot.

The teacher is correct-ish and slow: its cross-encoder pass is 0.4s
per episode cold, and it cannot be stored (1,025x1,408 tokens per
episode = 3.24 GB, four times the raw source, and unpoolable - a 4x
token reduction drops rank correlation to 0.18). So the teacher can
never be the read path. A student can, if it is shaped so that all the
video work happens at WRITE time:

    episode tower   channel vectors already in the store -> 256-d,
                    computed once at ingest, stored like any embedding
    query tower     text -> 256-d, one small forward per query
    retrieval       a single matmul over the corpus

TRAINING QUERIES ARE NOT THE TRUTHSET. The 10 benchmark queries are
eval-only, so training on the teacher's opinion of them would make the
benchmark a memory test. Training queries are generated from the
store's OWN attested vocabulary (_vocab.json, already corpus-derived)
crossed with generic English templates - the same construction the
vocab channel uses, no dataset metadata. The truthset queries are held
out entirely and only ever scored.

SAME METHODOLOGY, FASTER - the student is a PIPELINE, not a model.
A single bi-encoder cannot imitate this teacher no matter how it is
trained, because a dot product is one bilinear form and the teacher is
five sequential stages (channels->RRF, a PRF pass conditioned on the
first pass, an ITM cross-encoder cascade over the top-N, a routed
event+density gate, then the cut). That is a representational gap, and
the measurement showed its exact shape: the one-tower student tracked
the teacher where the teacher wins by CHANNELS (q03 0.71 vs 0.62, q01
0.53 vs 0.41) and collapsed to zero where it wins by a STAGE (q07
0.58 -> 0.00, the ITM stage; q02 0.50 -> 0.07, the event structure).

So the student mirrors the stages:
    stage 1  bi-encoder recall     one matmul over the corpus
    stage 2  interaction reranker  a small head over [q, e, q*e, |q-e|]
                                   on the top-N only - the same cascade
                                   shape as ITM, at ~1e-4 of its cost
Both are distilled from the teacher's ORDER, not its scores, because
the teacher's scale is RRF-arbitrary while its ordering is measured.

  python scripts/distill_student.py [--nq 200] [--epochs 60]
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from elidedb import Store                                    # noqa: E402

# WAS ten hand-written sentence frames - "put the {a} on the {b}",
# "open the {a}", ... - used to synthesise the text side of student
# training. That is the strongest form of hardwiring in the system,
# because a student trained on them has the task grammar baked into its
# WEIGHTS rather than into an `if` someone can later delete.
#
# The frames now come from the corpus: whatever phrasings the store's
# own labels attest, so a driving corpus trains on driving language and
# this one on manipulation language, with no list in between.
def corpus_templates(store, limit=32):
    """Attested phrasings, most frequent first. () if the store cannot
    supply any - in which case the caller must not invent some."""
    from collections import Counter
    from elidedb.derive import attested
    if "labels" not in store.tables():
        return ()
    vals = [str(v) for v in
            store.table("labels").scan().column("value").to_pylist()]
    vocab = set(attested(vals, min_count=2))
    keep = [v for v in vals if v and set(v.split()) & vocab]
    return tuple(w for w, _ in Counter(keep).most_common(limit))
DIM = 256


def episode_features(db, keys):
    """Everything the store already knows about an episode, concatenated.
    This is the student's INPUT and it is all precomputed - the point of
    the exercise is that nothing here needs a model at query time."""
    from elidedb.embeddings import _vec_table
    blocks = []
    for sp in ("pe_vectors", "sig2_vectors", "iv2_vectors",
               "motion_vectors", "vjepa_vectors"):
        try:
            tb, V = _vec_table(db, sp)
        except Exception:
            continue
        V = np.asarray(V, np.float32)
        em = defaultdict(list)
        for r, (s, a) in enumerate(zip(tb.column("stream").to_pylist(),
                                       tb.column("ts").to_pylist())):
            em[(str(s), int(a))].append(r)
        E = np.stack([V[em[(k[0], k[1])]].mean(0) if (k[0], k[1]) in em
                      else np.zeros(V.shape[1], np.float32) for k in keys])
        blocks.append(E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8))
    # the event structure, as the teacher computes it
    kinds = ["open", "close", "put_into", "put_on", "take_out", "adjust"]
    H = np.zeros((len(keys), len(kinds)), np.float32)
    if "events" in db.tables():
        kidx = {(k[0], k[1]): i for i, k in enumerate(keys)}
        ev = db.table("events").scan().to_pydict()
        for r in range(len(ev["ts"])):
            i = kidx.get((str(ev["stream"][r]), int(ev["ts"][r])))
            if i is not None and ev["kind"][r] in kinds:
                H[i, kinds.index(ev["kind"][r])] += 1.0
        H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    blocks.append(H)
    return np.hstack(blocks).astype(np.float32)


def main():
    import torch
    import torch.nn as nn

    from elidedb.scenario import _episodes, search_set
    argv = sys.argv
    nq = int(argv[argv.index("--nq") + 1]) if "--nq" in argv else 200
    epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv \
        else 60

    db = Store.open("lake/bench")
    keys = _episodes(db)
    kidx = {(k[0], k[1]): i for i, k in enumerate(keys)}
    X = episode_features(db, keys)
    print(f"episode features {X.shape}")

    # ---- training queries from the store's own attested vocabulary
    vocab = json.loads((Path("lake/bench") / "_vocab.json").read_text())
    terms = sorted({t for t in vocab} |
                   {v for lst in vocab.values() for v in (lst or [])})
    rng = np.random.default_rng(0)
    frames_ = corpus_templates(db)
    if not frames_:
        raise SystemExit(
            "no attested phrasings in this store - refusing to invent a "
            "sentence grammar. Ingest a corpus first; a student trained "
            "on hand-written frames has the task taxonomy in its weights.")
    qs = []
    guard = 0
    while len(qs) < nq and guard < nq * 50:
        guard += 1
        # an attested phrasing, with attested terms substituted into it
        # where it has slots; no frame is authored here
        tp = frames_[rng.integers(len(frames_))]
        a, b = terms[rng.integers(len(terms))], terms[rng.integers(len(terms))]
        q = tp.format(a=a, b=b) if "{a}" in tp else tp
        if q not in qs:
            qs.append(q)
    print(f"{len(qs)} training queries from {len(terms)} attested terms "
          f"and {len(frames_)} attested phrasings")

    # ---- teacher labels: its ORDER over the corpus, per query
    from elidedb.pe import _text_vec as pe_text
    # teacher labels are the expensive part (575s for 200 queries);
    # cache them so the student can be retrained in seconds
    cache = ROOT / "artifacts/teacher_labels.npz"
    t0 = time.time()
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["Y"]) >= nq:
            Y, QT = z["Y"][:nq], z["QT"][:nq]
            print(f"teacher labels from cache {Y.shape}")
            label_s = 0.0
            Y = np.asarray(Y); QT = np.asarray(QT)
            _skip = True
        else:
            _skip = False
    else:
        _skip = False
    Y2, QT2 = [], []
    if not _skip:
        for i, q in enumerate(qs):
            r = search_set(db, q, purity="fast", k_max=100,
                           return_ranking=True)
            # RANK, not score: the teacher's scale is RRF-arbitrary and
            # a sentinel for unranked episodes poisons any statistic
            # taken over it. The first attempt filled -1e9 and used
            # Yt.std() to temper the softmax target - std went to ~1e9,
            # every target became uniform, the loss froze at ln(200)
            # and the student learned nothing (0.07 yield). Ranks are
            # scale-free and have no sentinel.
            sc = np.full(len(keys), float(len(keys)), np.float32)
            for rank, (s_, a_, v) in enumerate(r.get("ranking", [])):
                j = kidx.get((s_, a_))
                if j is not None:
                    sc[j] = float(rank)
            Y2.append(sc)
            QT2.append(np.asarray(pe_text(q), np.float32))
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  teacher {i+1}/{len(qs)} {el:.0f}s "
                      f"ETA {el/(i+1)*len(qs)/60:.0f}min", flush=True)
        Y = np.stack(Y2); QT = np.stack(QT2)
        QT /= np.linalg.norm(QT, axis=1, keepdims=True) + 1e-8
        label_s = time.time() - t0
        np.savez(cache, Y=Y, QT=QT)
        print(f"teacher labels {Y.shape} in {label_s:.0f}s "
              f"({label_s/len(Y):.1f}s/query)")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(X, device=dev)
    Qt = torch.tensor(QT, device=dev)
    Yt = torch.tensor(Y, device=dev)

    class Tower(nn.Module):
        def __init__(self, d_in, d=DIM):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(),
                                   nn.Linear(512, d))

        def forward(self, x):
            y = self.f(x)
            return y / (y.norm(dim=-1, keepdim=True) + 1e-8)

    class Rerank(nn.Module):
        """Stage 2: the cascade, distilled. Sees the query and the
        episode TOGETHER (concat, product, absolute difference), which
        is the interaction a dot product structurally cannot express."""
        def __init__(self, d=DIM):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(4 * d, 256), nn.GELU(),
                                   nn.Linear(256, 64), nn.GELU(),
                                   nn.Linear(64, 1))

        def forward(self, qe, ee):
            q = qe.unsqueeze(1).expand_as(ee)
            return self.f(torch.cat([q, ee, q * ee, (q - ee).abs()],
                                    -1)).squeeze(-1)

    ep_t, q_t = Tower(X.shape[1]).to(dev), Tower(QT.shape[1]).to(dev)
    rr_t = Rerank().to(dev)
    nparam = sum(p.numel() for p in list(ep_t.parameters())
                 + list(q_t.parameters()))
    opt = torch.optim.AdamW(list(ep_t.parameters()) + list(q_t.parameters())
                            + list(rr_t.parameters()),
                            lr=3e-4, weight_decay=1e-2)
    print(f"student {nparam/1e6:.2f}M params, training on {len(Y)} queries")

    # listwise: match the teacher's ORDER (softmax over its top slice),
    # not its RRF-arbitrary scale
    # Yt holds RANKS (lower = better), so ascending, and the target is
    # an RRF-shaped decay over position - bounded, scale-free, and it
    # cannot be flattened by an outlier the way a score softmax was.
    TOP = 200
    order = torch.argsort(Yt, dim=1, descending=False)[:, :TOP]
    pos = torch.arange(TOP, device=dev, dtype=torch.float32)
    tgt = (1.0 / (10.0 + pos)).expand(len(Yt), TOP).contiguous()
    tgt = tgt / tgt.sum(1, keepdim=True)
    t1 = time.time()
    for e in range(epochs):
        perm = torch.randperm(len(Yt), device=dev)
        tot = 0.0
        for i in range(0, len(perm), 16):
            b = perm[i:i + 16]
            qe = q_t(Qt[b])
            cand = order[b]
            ee = ep_t(Xt[cand.reshape(-1)]).reshape(len(b), TOP, DIM)
            logits = torch.einsum("bd,bkd->bk", qe, ee) * 20.0
            # stage 1 learns recall over the whole slice; stage 2 learns
            # to ORDER the head of it, which is where the teacher's
            # cascade does its work
            loss = -(tgt[b] * torch.log_softmax(logits, 1)).sum(1).mean()
            HEAD = 50
            r2 = rr_t(qe, ee[:, :HEAD])
            t2 = tgt[b][:, :HEAD]
            t2 = t2 / t2.sum(1, keepdim=True)
            loss = loss - (t2 * torch.log_softmax(r2, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if (e + 1) % 20 == 0:
            print(f"  epoch {e+1} loss {tot/max(1,len(perm)//16):.4f}",
                  flush=True)
    train_s = time.time() - t1

    out = ROOT / "models/student_v1"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"ep": ep_t.state_dict(), "q": q_t.state_dict(),
                "rr": rr_t.state_dict(),
                "d_ep": X.shape[1], "d_q": QT.shape[1], "dim": DIM},
               out / "student.pt")
    # the episode side is WRITE-TIME work: materialize it now
    with torch.no_grad():
        EMB = ep_t(Xt).cpu().numpy().astype(np.float16)
    np.savez(out / "episode_emb.npz", E=EMB,
             streams=np.array([k[0] for k in keys]),
             ts=np.array([k[1] for k in keys], np.int64))
    (out / "meta.json").write_text(json.dumps(
        {"teacher": "models/teacher_v1.json", "params": int(nparam),
         "train_queries": int(len(Y)), "dim": DIM,
         "label_seconds": round(label_s, 1),
         "train_seconds": round(train_s, 1)}, indent=1))
    print(json.dumps({"saved": str(out), "params_M": round(nparam/1e6, 2),
                      "label_s": round(label_s), "train_s": round(train_s),
                      "episode_emb_MB": round(EMB.nbytes/1e6, 2)}))


if __name__ == "__main__":
    main()
