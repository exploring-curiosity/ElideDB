"""Text -> z. Put a text query into the space the example queries already work in.

WHY. Query-by-example runs through the trained ranker and reaches P@support
0.871. Query-by-text runs through the raw SigLIP text tower and sits at chance
(0.224 against 0.221), because that tower collapses the direction of the
action: "Open the cabinet doors" and "Close the cabinet doors" are 0.9974
apart. Open/hinged and Close/hinged are different classes, so no image-side
quality can rescue it. The fix is not a better tower - it is to LEARN a map
from the text embedding into the ranker's own embedding space, so text is
scored by the same geometry that already separates these events.

    g(text) -> e in R^128,   score = cos(g(text), e_clip)

The ranker is FROZEN here; only g is fitted. That is deliberate - the video
side already works and text must come to it, not the reverse.

CAN A COLLAPSED INPUT BE RECOVERED? 0.9974 cosine still leaves a direction; the
information is present but low-variance, which is exactly the situation where
cosine fails and a supervised head succeeds - the same diagnosis that made the
ranker work. So it is plausible. It is not guaranteed, and the benchmark below
is built to tell the difference between recovery and memorisation.

THE MEMORISATION PROBLEM, stated up front because it dominates the result. The
corpus has 54 unique instructions over 447 episodes, and the articulation
families have one or two each ("Close the microwave door." is the ONLY
CloseMicrowave string). A head trained on those can memorise twelve strings and
score near-perfectly while having learned nothing about language. So three
evaluations are reported, and only the last two are evidence:

  seen       query strings the head trained on. Memorisable. Reported as a
             ceiling, never as a result.
  held-out   phrasings withheld from training. Only PickPlace families have
             enough distinct strings (10-16) for this to be possible at all -
             which means EVERY held-out phrasing is a PickPlace object swap
             ("apple" for "bowl"), and none of them test open-vs-close. Its
             0.932 is therefore NOT evidence that language generalises; it
             says object nouns are interchangeable, which was never in doubt.
  paraphrase strings I wrote that appear nowhere in the corpus. These are
             AUTHORED BY ME and flagged as such - they test whether the head
             generalises past the exact training strings, and they are the
             weakest link in this benchmark because their wording is mine.

    python -m relmo.vjtext2z --tag t2z_s0
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrank, vjrel  # noqa: E402
from relmo.vjeval import group_key, parse  # noqa: E402
from relmo.vjood import recs_for  # noqa: E402
from relmo.vjrankeval import load_ckpt  # noqa: E402
from relmo.vjsig import MODEL as SIG_MODEL  # noqa: E402
from relmo.vjsplit import load as load_split  # noqa: E402

CKPT = R.BASE / "models" / "vjtext2z"

# Eval-only paraphrases. AUTHORED BY ME, present nowhere in the corpus, used
# only to test generalisation past the exact training strings. Never trained on.
PARAPHRASE = {
    "OpenCabinet": ["someone swings a cupboard door open",
                    "pulling the kitchen unit open"],
    "CloseCabinet": ["someone pushes the cupboard door shut",
                     "the kitchen unit is closed again"],
    "OpenMicrowave": ["the microwave hatch is pulled open",
                      "opening the front of the microwave"],
    "CloseMicrowave": ["the microwave hatch is pushed shut",
                       "shutting the front of the microwave"],
    "OpenDrawer": ["a drawer is pulled out on its rails",
                   "sliding the drawer outward"],
    "CloseDrawer": ["a drawer is pushed back in",
                    "sliding the drawer shut"],
    "PickPlaceCounterToSink": ["moving an item from the worktop into the basin",
                               "an object is carried over to the sink"],
    "PickPlaceCounterToCabinet": ["putting something away inside the cupboard",
                                  "an item is stowed in the unit"],
    "PickPlaceCounterToDrawer": ["placing an object down into the drawer",
                                 "an item goes into the open drawer"],
}


def instr_by_task(dataset="rcasa"):
    man = R.read_manifest(dataset)
    out = defaultdict(set)
    per_ep = {}
    for e in man["episodes"]:
        if e.get("instruction"):
            out[e["task"]].add(e["instruction"])
            per_ep[e["id"]] = e["instruction"]
    return {k: sorted(v) for k, v in out.items()}, per_ep


def text_rel(task, ep_ids, meta):
    """Graded relevance of a text query (from `task`) against episodes.

    Same shape as vjrel: the event group is the positive boundary, matching
    object adds. Scene and camera terms cannot apply - a string has neither.
    """
    m = parse(task + "_episode_000000__x")
    cls, obj = group_key(m), m["obj"]
    rel = np.zeros(len(ep_ids), np.float32)
    for k, i in enumerate(ep_ids):
        mi = parse(i)
        if group_key(mi) == cls:
            rel[k] = vjrel.W_EVENT + (vjrel.W_OBJ if mi["obj"] == obj else 0.0)
    return rel


def embed_text(strings, torch):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SIG_MODEL)
    sig = AutoModel.from_pretrained(SIG_MODEL, dtype=torch.float32).eval()
    with torch.no_grad():
        t = tok(strings, padding="max_length", max_length=64, truncation=True,
                return_tensors="pt")
        T = sig.get_text_features(**t)
    return torch.nn.functional.normalize(T, dim=-1)


def clip_emb(model, recs, ids, torch):
    """The ranker's own pooled embedding - what cos() is taken against."""
    with torch.no_grad():
        Z = model.encode(
            torch.tensor(np.stack([recs[i]["a"] for i in ids])),
            torch.tensor(np.stack([recs[i]["g"] for i in ids])),
            torch.tensor(np.stack([recs[i]["sig"] for i in ids])))
        return torch.nn.functional.normalize(
            torch.nn.functional.normalize(Z, dim=-1).mean(1), dim=-1)


def prec_at(order, correct, k):
    k = min(k, len(order))
    return float(correct[order[:k]].sum()) / k if k else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker", default="rk_noscorer_s0")
    ap.add_argument("--dim-h", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--holdout-frac", type=float, default=0.4,
                    help="fraction of DISTINCT phrasings withheld, where a "
                         "family has enough of them to withhold any")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    sp = load_split()
    Rin = recs_for("rcasa")
    meta = vjrel.meta_table("rcasa")
    model, _ = load_ckpt(a.ranker)
    for p_ in model.parameters():
        p_.requires_grad_(False)

    tr_ids = [i for i in sorted(sp["train"]) if i in Rin]
    pool = sorted((sp["val"] | sp["test"]) & set(Rin))
    E_tr = clip_emb(model, Rin, tr_ids, torch)
    E_pool = clip_emb(model, Rin, pool, torch)
    gp = np.array([group_key(parse(i)) for i in pool])

    by_task, _ = instr_by_task("rcasa")
    train_str, held_str = [], []
    for t, ss in by_task.items():
        n_hold = int(len(ss) * a.holdout_frac) if len(ss) >= 4 else 0
        idx = rng.permutation(len(ss))
        for j, k in enumerate(idx):
            (held_str if j < n_hold else train_str).append((t, ss[k]))
    print(f"{sum(len(v) for v in by_task.values())} unique instructions | "
          f"train {len(train_str)} | held-out phrasings {len(held_str)}")

    all_str = [s for _, s in train_str] + [s for _, s in held_str]
    para = [(t, s) for t, ss in PARAPHRASE.items() for s in ss]
    all_str += [s for _, s in para]
    T_all = embed_text(all_str, torch)
    n_tr, n_hd = len(train_str), len(held_str)
    T_tr, T_hd, T_pa = (T_all[:n_tr], T_all[n_tr:n_tr + n_hd],
                        T_all[n_tr + n_hd:])

    REL_tr = torch.tensor(np.stack([text_rel(t, tr_ids, meta)
                                    for t, _ in train_str]))
    head = nn.Sequential(nn.LayerNorm(T_all.shape[1]),
                         nn.Linear(T_all.shape[1], a.dim_h), nn.GELU(),
                         nn.Linear(a.dim_h, model.d))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    print(f"  head {sum(p.numel() for p in head.parameters())/1e3:.0f}k params "
          f"-> {model.d}-d, ranker FROZEN", flush=True)

    from tqdm import tqdm
    for _ in tqdm(range(a.epochs), unit="ep", desc="text2z"):
        e = torch.nn.functional.normalize(head(T_tr), dim=-1)
        s = e @ E_tr.T                                   # (n_text, n_train_ep)
        loss = vjrank.lambda_rank(s, REL_tr,
                                  torch.ones_like(REL_tr, dtype=torch.bool),
                                  torch)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def bench(name, strings, Tx, authored=False):
        with torch.no_grad():
            e = torch.nn.functional.normalize(head(Tx), dim=-1)
            S = (e @ E_pool.T).numpy()
        rows = []
        for qi, (task, _) in enumerate(strings):
            cls = group_key(parse(task + "_episode_000000__x"))
            corr = gp == cls
            sup = int(corr.sum())
            if sup < 5:
                continue
            o = np.argsort(-S[qi])
            rows.append((sup, prec_at(o, corr, sup), prec_at(o, corr, 10),
                         sup / len(pool)))
        if not rows:
            print(f"{name:34s}  (no gradeable queries)")
            return
        w = np.array([r[0] for r in rows], float)
        flag = "   <- strings AUTHORED BY ME" if authored else ""
        print(f"{name:34s} {len(rows):4d} "
              f"{np.average([r[1] for r in rows], weights=w):8.3f} "
              f"{np.average([r[2] for r in rows], weights=w):7.3f} "
              f"{np.average([r[3] for r in rows], weights=w):8.3f}{flag}")

    print(f"\n{'query set':34s} {'n':>4s} {'P@sup':>8s} {'P@10':>7s} "
          f"{'chance':>8s}")
    # every held-out string is a PickPlace object swap - see the docstring
    bench("seen strings (MEMORISABLE)", train_str, T_tr)
    bench("held-out phrasings", held_str, T_hd)
    bench("paraphrases (never in corpus)", para, T_pa, authored=True)

    CKPT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"t2z_s{a.seed}"
    torch.save(dict(head=head.state_dict(), ranker=a.ranker, dim=model.d),
               CKPT / f"{tag}.pt")
    R.log("vjtext2z", tag=tag, ranker=a.ranker, seed=a.seed,
          n_train_str=len(train_str), n_held=len(held_str))


if __name__ == "__main__":
    main()
