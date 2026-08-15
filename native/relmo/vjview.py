"""Watchable page: what does the system return, and how deep are the analogies?

The earlier version of this page ranked over a pool with the query's own object
REMOVED. That was right for isolating the cross-object question but wrong as a
demo: it is not what a user would ever see. This version ranks over the FULL
corpus, exactly as a real query would, and then shows two things per query:

  ROW 1  the top of the ranking, unfiltered. Expect same-object matches here,
         and that is CORRECT - another clip of this same cabinet opening really
         is the nearest moment. Ranking it first is not a failure.

  ROW 2  the highest-ranked results on a DIFFERENT object, each labelled with
         its true rank in the full list. This is the "cabinet, fridge or
         microwave" question: not whether analogies beat near-duplicates, but
         whether they are found at all, and how deep you must look.

Measuring only the top 10 hid this completely: the first ten slots are
structurally occupied by same-object matches, so cross-object hits could not
appear there no matter how well the system worked, and their absence was read
as a failure. At k = support, v2 retrieves cross-object same-verb at 1.13x
chance - the only arm above chance (scene-only 1.02x, v1 0.96x).

COLOURS      green  same kind of event, DIFFERENT object   <- the prize
             grey   same kind of event, same object        <- correct, easy
             red    different kind of event

The ranking never sees the family names; they only colour the borders and are
parsed for grading alone.

    python -m relmo.vjview --dataset rcasa --queries 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relmo import registry as R  # noqa: E402
from relmo import vjrank2, vjrel  # noqa: E402
from relmo.vjeval import group_key, l2, parse  # noqa: E402
from relmo.vjmatch import dtw  # noqa: E402
from relmo.vjzeval import PAD_COST, _pad  # noqa: E402

SHOW = 6
CKPT = "p1_reg_low_s0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rcasa")
    ap.add_argument("--queries", type=int, default=12)
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()

    AM = vjrel.all_meta()
    print(f"loading per-token corpus and {a.ckpt}...", flush=True)
    D = vjrank2.load_corpus()
    model, _ = vjrank2.load_ckpt(a.ckpt)
    Z = vjrank2.encode_all(model, D)
    names = [i for i in sorted(Z) if i in AM and parse(i)]
    meta = [parse(i) for i in names]
    verb = np.array([group_key(m) for m in meta])
    obj = np.array([m["obj"] for m in meta])
    epid = np.array([AM[i]["rollout"] for i in names])
    dur = np.array([AM[i]["dur"] for i in names])
    P, ok = _pad([l2(Z[i]) for i in names])
    L = np.array([len(Z[i]) for i in names])
    REL, _ = vjrel.relevance(names, AM)

    # per-step activity for the sparkline, from the gate the pooling head sees
    traces = []
    for i in names:
        g = D[i]["gate"].sum(1)
        traces.append((g / (g.max() + 1e-9)).round(3).tolist())

    rng = np.random.default_rng(0)
    combos = sorted({(v, o) for v, o in zip(verb, obj)
                     if len(np.unique(obj[verb == v])) > 1})
    per = max(1, a.queries // max(len(combos), 1))
    picks = []
    for v, o in combos:
        idx = np.where((verb == v) & (obj == o))[0]
        picks += [int(x) for x in rng.choice(idx, min(per, len(idx)),
                                             replace=False)]
    picks = picks[:a.queries]

    def card(j, rank=None, qv=None, qo=None, qd=None):
        kind = ("q" if qv is None else
                "cross" if (verb[j] == qv and obj[j] != qo) else
                "same" if verb[j] == qv else "wrong")
        # the GT now decays relevance with the DURATION RATIO, so show it:
        # a match at 2.9x duration is a different moment, not a near miss
        ratio = (max(dur[j], qd) / max(min(dur[j], qd), 1e-9)) if qd else None
        return dict(verb=meta[j]["verb"], obj=meta[j]["obj"], kind=kind,
                    rank=rank, trace=traces[j], dur=round(float(dur[j]), 1),
                    ratio=(round(float(ratio), 2) if ratio else None),
                    src=f"../datasets/{a.dataset}/shard_0000/{names[j]}"
                        f"/frames.mp4")

    out = []
    for i in picks:
        keep = epid != epid[i]
        full = np.where(keep)[0]
        C = 1.0 - np.einsum("sd,nkd->nsk", l2(Z[names[i]]), P[keep])
        C = np.where(ok[keep][:, None, :], C, PAD_COST)
        s = -dtw(C, False, L[keep])
        order = full[np.argsort(-s)]
        rank_of = {int(j): r + 1 for r, j in enumerate(order)}
        sv = verb[order] == verb[i]
        cx = sv & (obj[order] != obj[i])
        sup = int(sv.sum())
        if sup < 5 or cx.sum() < 3:
            continue
        rel = REL[i][order]
        k = sup
        disc = 1 / np.log2(np.arange(2, k + 2))
        nd = float((rel[:k] * disc).sum()
                   / max((np.sort(rel)[::-1][:k] * disc).sum(), 1e-9))
        prec = float(sv[:sup].sum() / sup)
        rec = float(cx[:sup].sum() / max(cx.sum(), 1))
        first = next((rank_of[int(j)] for j in order
                      if obj[j] != obj[i] and verb[j] == verb[i]), None)
        diff = [int(j) for j in order if obj[j] != obj[i]][:SHOW]
        out.append(dict(
            query=card(i), verb=meta[i]["verb"], obj=meta[i]["obj"],
            support=sup, n_cross=int(cx.sum()), prec=round(prec, 3),
            ndcg=round(nd, 3), recall=round(rec, 3), first=first,
            pool=int(keep.sum()),
            top=[card(int(j), rank_of[int(j)], verb[i], obj[i], dur[i])
                 for j in order[:SHOW]],
            cross=[card(j, rank_of[j], verb[i], obj[i], dur[i])
                   for j in diff]))

    vd = R.BASE / "viewer"
    vd.mkdir(parents=True, exist_ok=True)
    # inline the data. A sibling results.json + fetch() is blocked by the
    # file:// origin policy, which is why the page rendered its header and an
    # empty #app - the script threw before it ever wrote any content. Inlining
    # makes the page work by double-clicking it, with no server.
    (vd / "index.html").write_text(
        HTML.replace("/*__RESULTS__*/[]", json.dumps(out)))
    (vd / "results.json").write_text(json.dumps(out))
    print(f"wrote {len(out)} queries -> {vd/'index.html'}")
    print(f"{'query':22s} {'support':>8s} {'prec@sup':>9s} {'NDCG@sup':>9s} "
          f"{'cross rec':>10s} {'1st cross-obj':>14s}")
    for q in out:
        print(f"{q['verb']+q['obj']:22s} {q['support']:8d} {q['prec']:9.3f} "
              f"{q['ndcg']:9.3f} {q['recall']:10.3f} {str(q['first']):>14s}")
    print(f"\nmean prec@support {np.mean([q['prec'] for q in out]):.3f}  "
          f"mean NDCG@support {np.mean([q['ndcg'] for q in out]):.3f}  "
          f"(this page is a DEMO over the full corpus, not the held-out "
          f"protocol; test-split numbers are in RESULTS.md)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ElideDB - when did something like this happen?</title>
<style>
:root{--bg:#fff;--fg:#111;--dim:#666;--line:#e3e3e3;--ok:#1a7f37;--no:#c0392b;--same:#7a7a7a;--card:#fafafa}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--dim:#999;--line:#2c2c2c;--ok:#3fb950;--no:#f85149;--same:#8b8b8b;--card:#1a1a1a}}
:root[data-theme=dark]{--bg:#111;--fg:#eee;--dim:#999;--line:#2c2c2c;--ok:#3fb950;--no:#f85149;--same:#8b8b8b;--card:#1a1a1a}
:root[data-theme=light]{--bg:#fff;--fg:#111;--dim:#666;--line:#e3e3e3;--ok:#1a7f37;--no:#c0392b;--same:#7a7a7a;--card:#fafafa}
body{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 6px}p.sub{color:var(--dim);margin:0 0 8px;max-width:74ch}
.q{border-top:1px solid var(--line);padding:20px 0}
.qh{display:flex;gap:18px;align-items:flex-start;margin-bottom:14px}
.lab{font-weight:600}.dim{color:var(--dim);font-weight:400}
.row{display:flex;gap:10px;overflow-x:auto;padding-bottom:6px}
.c{flex:0 0 150px;background:var(--card);border:2px solid var(--line);border-radius:8px;padding:6px;position:relative}
.c.cross{border-color:var(--ok)}.c.wrong{border-color:var(--no)}.c.same{border-color:var(--same)}
video{width:100%;border-radius:4px;display:block;background:#000}
.n{font-size:11px;margin-top:5px;line-height:1.35}
.rk{position:absolute;top:9px;right:9px;background:rgba(0,0,0,.72);color:#fff;
 font-size:10px;padding:1px 6px;border-radius:9px;font-variant-numeric:tabular-nums}
svg{display:block;margin-top:4px}
.arm{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:14px 0 5px}
.stat{font-size:12px;color:var(--dim);margin-top:2px;font-variant-numeric:tabular-nums}
.key{font-size:12px;color:var(--dim);margin-bottom:18px}
b.ok{color:var(--ok)}b.no{color:var(--no)}b.sm{color:var(--same)}
</style>
<h1>When did something like this happen? &mdash; cabinet, fridge or microwave</h1>
<p class="sub">Learned 256-token pooling over V-JEPA&nbsp;2 plus SigLIP&nbsp;2 appearance, matched by
anchored symmetric2 DTW over a 128-d state. Ranked across the <em>whole</em> corpus, exactly as a real
query would be. The top of the list is dominated by the query's own object &mdash; that is correct,
another clip of the same cabinet opening really is the nearest moment. The question is whether the
<em>analogies</em> are found at all and how deep you have to look, which is what the second row shows.</p>
<p class="sub"><b>This page is a demo over the full corpus and includes training recordings, so it reads
higher than the real numbers.</b> Held out, on test queries against a train-disjoint pool:
prec@support&nbsp;0.856&nbsp;&plusmn;&nbsp;0.014, NDCG@support&nbsp;0.741, wAUC&nbsp;0.943 &mdash;
against a random floor of 0.217&nbsp;/&nbsp;0.150&nbsp;/&nbsp;0.502.</p>
<p class="key"><b class="ok">Green</b> and <b class="sm">grey</b> are both <b>correct</b> &mdash; same kind of event.
Green additionally means a <em>different object</em>, the harder case.
<b class="no">Red</b> is wrong: a different kind of event. Badge = rank in the full ranking.
The ranking never sees these labels. Each result also shows its duration and its duration
<em>ratio</em> to the query &mdash; the ground truth decays relevance as that ratio grows, because a
much slower replay is a different moment, not a near miss.</p>
</head><body>
<div id="app">loading&hellip;</div>
<script>
const DATA = /*__RESULTS__*/[];
function spark(t){const w=138,h=22,n=t.length;
 const d=t.map((v,i)=>`${i?'L':'M'}${(i/(n-1)*w).toFixed(1)},${(h-v*(h-2)-1).toFixed(1)}`).join(' ');
 return `<svg width="${w}" height="${h}"><path d="${d}" fill="none" stroke="currentColor" stroke-width="1.4" opacity=".7"/></svg>`}
function card(c){
 // data-src + IntersectionObserver, NOT src+autoplay. This page holds ~230
 // clips; loading them all at once exceeds the browser's per-host connection
 // cap, every request queues, and the whole page renders as black rectangles.
 return `<div class="c ${c.kind}"><video data-src="${c.src}" muted loop playsinline preload="none"></video>
 ${c.rank?`<span class="rk">#${c.rank}</span>`:''}
 <div class="n"><span class="lab">${c.verb}</span><span class="dim">${c.obj}</span>
 <span class="dim"> &middot; ${c.dur}s${c.ratio?` &middot; ${c.ratio}\u00d7`:''}</span></div>${spark(c.trace)}</div>`}
(function(qs){
 document.getElementById('app').innerHTML=qs.map(q=>`<div class="q">
 <div class="qh">${card(q.query)}
 <div><div class="lab" style="font-size:17px">${q.verb} ${q.obj}</div>
 <div class="dim">query clip &middot; ranked over all ${q.pool} other clips</div>
 <div class="stat">${q.n_cross} clips in the corpus are <b>${q.verb}</b> on a different object.</div>
 <div class="stat">prec@support(${q.support}) <b>${q.prec}</b> &middot;
 NDCG@support <b>${q.ndcg}</b> &middot; random floor 0.217 / 0.150</div>
 <div class="stat">of those, cross-object recall <b>${q.recall}</b> &middot;
 first different object at rank <b>${q.first??'&mdash;'}</b></div></div></div>
 <div class="arm">top of the ranking &mdash; unfiltered</div>
 <div class="row">${q.top.map(card).join('')}</div>
 <div class="arm">highest-ranked results on a different object</div>
 <div class="row">${q.cross.map(card).join('')}</div></div>`).join('');
 const io=new IntersectionObserver((es)=>{es.forEach(e=>{const v=e.target;
   if(e.isIntersecting){if(!v.src){v.src=v.dataset.src}v.play().catch(()=>{})}
   else{v.pause()}})},{rootMargin:'240px'});
 document.querySelectorAll('video').forEach(v=>io.observe(v))})(DATA);
</script></body></html>"""


if __name__ == "__main__":
    main()
