# NO-HARDWIRE violations — audit, 2026-07-31

**The rule:** dataset or task priors written into code are forbidden. A
constant is fine when it is a *tuning* dial measured on the corpus, or a
property of a model. It is a violation when it encodes **what the data
contains** or **what the task is** — because then the system is not
recognising structure, it is being told the answer, and every metric
measured against it is partly circular.

Found by audit after the V-JEPA2 channel benchmark exposed the verb set
as hand-written. Ordered by severity: how much of a reported number each
one is responsible for.

---

## S1 — the verb vocabulary and its assignment rules

**The core violation. Everything downstream of it inherits the problem.**

| where | what |
|---|---|
| `python/elidedb/plan.py:37` | `FLAG_KINDS = ("open","close","put_into","put_on","take_out","adjust","contact","release")` — a literal tuple of eight verbs |
| `scripts/write_once.py:177` | `"open" if d >= CAV_MIN else "close" if d <= -CAV_MIN else None` |
| `scripts/write_once.py:199` | `"adjust" if disp < REL_MIN else "take_out" if ... else "put_into" if ... else "put_on"` — a hand-written if/else chain |
| `scripts/build_teacher.py:101-102` | `REL_MIN = 0.5`, `CAV_MIN = 0.015` — the thresholds that chain turns on |

Not derived from the data and not from dataset metadata — chosen by hand.

**What it costs, measured:** `adjust` is the `else` branch of that chain,
so it is a catch-all rather than a class, and it scores AUC **0.553** on
the V-JEPA2 teacher against 0.898 for `put_on`. The write path's own
record already said 82% of participant motion typed as `adjust`. A class
that absorbs everything unclassified is not a class.

**It also makes the channel benchmark partly circular:** `chanbench`
scores a teacher on `events.kind` labels, and those labels are produced
by this if/else. Teacher AUC on `open`/`close`/`put_into`/`put_on` is
partly a measure of how well the teacher reproduces my own thresholds.
`contact`/`release` are the exception and the only ones to trust — they
come from track onset/offset, which is geometry, not a named category.

**Fix direction:** transitions should be *discovered* — cluster the
motion/geometry signal and let the corpus say how many kinds there are
and where the boundaries fall, then name them at read time if a human
wants names. The event teacher already produces continuous quantities
(displacement, cavity change, containment); the sin is quantising them
through a hand-authored ladder.

---

## S2 — English → transition maps in the query path

| where | what |
|---|---|
| `python/elidedb/scenario.py:166` | `_PREPK = (("on top","put_on"), ("onto","put_on"), ("into","put_into"), ("out of","take_out"), (" in ","put_into"), (" on ","put_on"))` |
| `python/elidedb/scenario.py:168` | `_VERBK = {"open":"open", "opens":"open", "close":"close", "closes":"close", "shut":"close"}` |
| `scripts/match_teacher.py:39`, `scripts/match_events.py:69` | the same prep→verb table, duplicated |
| `python/elidedb/grounding.py:30-31` | `_INWARD`, `_OUTWARD` preposition families |

Defended in-comment as "closed-class English, dictionary knowledge
only". That defence holds for the prepositions; it does **not** hold for
the right-hand side — `put_on`, `put_into`, `take_out` are the S1
vocabulary, so these tables hardwire the task taxonomy into query
parsing. A corpus with different transitions cannot be queried through
them.

**Fix direction:** map query language to the *discovered* transition
space by similarity, not by a lookup table. Duplication across three
files should collapse to one derivation regardless.

---

## S3 — the un-reversal verb list

`python/elidedb/lexicon.py:33` — `_UN_BASES = ["fold","wrap","roll",
"stack","cover","screw","plug","zip","tie","load","lock","pack",
"buckle","hook","fasten","tangle"]`

Sixteen hand-picked English verbs whose opposite is their `un-` form.
The comment claims "generic English, generated inflections, zero dataset
words", and the inflection *is* generated — but the sixteen bases are a
hand-authored list, and `fold`/`stack`/`cover` are conspicuously the
Bridge manipulation vocabulary. The mechanism (swap-contrast to cancel
concept-presence bias) is sound and measured; the list is not derived.

**Fix direction:** derive the un- family morphologically from the corpus
vocabulary rather than enumerating it.

---

## S4 — geometry thresholds, hand-set and inconsistent

| where | value |
|---|---|
| `scripts/extract_events.py:55` | `MIN_BLOB = 120` px |
| `scripts/probe_actors.py:43` | `MIN_BLOB = 40` |
| `scripts/probe_objectfile.py:49` | `MIN_BLOB = 60` |
| `scripts/probe_story.py:44` | `MIN_BLOB = 30` |
| `scripts/track_ingest.py:48` | `MIN_BLOB = 60` |
| `scripts/verb_recompute.py:51,53` | `DISP_MIN = 0.05`, `CAV_MIN = 0.015` |
| `scripts/extract_events.py:56` | `PAD = 0.6` |
| `scripts/track_ingest.py:51` | `PAD = 0.35` |

Five different values for the same concept across five files is the
tell: none was derived, each was tuned where it sat. These are pixel
scales for *this* corpus's resolution and object sizes.

**Fix direction:** express as a fraction of frame area and fit the cut
from the observed blob-size distribution, the way the identity match cut
is now fitted from co-existing-track negatives.

---

## S5 — measured-but-corpus-specific (weakest category, listed for honesty)

| where | what | status |
|---|---|---|
| `python/elidedb/identity.py` | `DET_SZ = 448`, `ADMIT = 0.70`, `MAX_EXEMPLARS = 5` | measured on Bridge; `MATCH` is now correctly fitted from free negatives, these three are not |
| `python/elidedb/store.py` | `min_group_rows` per table | genuinely tuning, measured per table — **not** a violation |
| `scripts/full_write.py` | `GAP_S = 60.0`, `CRF = 26`, `NGEOM = 12` | policy dials, not content priors — **not** violations |
| `scripts/*.py` | `CAM`, `EPOCH_NS`, `FILE_STRIDE_NS`, `FPS = 5.0` | dataset-layout constants; belong in an ingest config, not in six scripts |

---

## Not violations

- Model identifiers (`facebook/vjepa2-...`, `google/siglip2-...`) — naming a model is not a prior about data.
- The SSv2 174-class vocabulary — a **property of the probe**, read from the checkpoint, not authored here.
- `identity.MATCH` — fitted per corpus from disjoint co-existing tracks.
- Row-group sizes, compression, page-index flags — storage tuning, measured.

---

## Fix order

1. **S1** first: it is the root, and it currently contaminates the
   channel benchmark that everything else is now being judged by.
2. **S2** follows automatically once transitions are discovered rather
   than enumerated.
3. **S4** is mechanical and independent — do it any time.
4. **S3** is small and low-impact.
5. **S5** mostly wants a config file, not a redesign.

**Until S1 is fixed, treat `task_auc` on `open`/`close`/`put_into`/
`put_on` as partly circular and rely on `contact`/`release` — the two
labels that come from geometry rather than from a hand-written ladder.**

---

# Deep sweep — the rest of the system, 2026-07-31

The first pass stopped at the geometry. Sweeping every module found
larger violations in the **language path**, including one the code's own
comments admit to. Severities continue from S1.

---

## S0 — the caption prompt is a task prior, and it was tuned on eval labels

**Worse than S1. Promoted above it.**

`python/elidedb/context.py:82` — `MANIPULATION_PROMPT`, the instruction
given to the VLM that produces `context_captions`:

> "the action verb in plain English (whatever it is - picking up,
> putting, opening, closing, pushing, pouring, wiping, pressing...), the
> object acted on with its colour, and where it ends up"

Two separate violations in one string.

**1. It enumerates the task vocabulary.** Eight verbs, hand-picked. The
file's own comment states the consequence outright: *"The caption IS the
index: whatever verbs the prompt teaches are the only verbs lexical
recall can ever match."* A verb absent from that list is unfindable by
construction. The comment even records the previous version failing this
way — 2,348 captions with `picks` x2371, `puts` x1673 and **zero**
close/open/wipe/push, so "close the drawer" could not be retrieved.

**2. It was tuned against the eval labels.** The comment documents the
generic prompt producing *"a robot arm interacts with a wooden box"*
while *"the human label for the same clip was 'put red object in the
drawer'"* — and the prompt was rewritten to close that gap. That is the
truthset's own vocabulary steering the design of the index. The comment
defends itself with "it never names objects that appear in the labels",
which addresses nouns and concedes the verbs and the shape.

It also declares itself: *"the prompt is a per-domain parameter"*. A
per-domain parameter authored by hand is precisely what the rule forbids.

**Fix direction:** captions should not be the index at all — they are a
VLM in the path, already ruled out. If a lexical channel is wanted, its
vocabulary must come from the corpus (attested words), not from a prompt.

---

## S1b — VERB_SWAPS: 81 hand-authored opposite pairs

`python/elidedb/lexicon.py:15` — open/close, picks up/puts down,
lifts/lowers, pushes/pulls, into/out of, onto/off, left/right, up/down,
toward/away from, front/back, and 71 more.

Same family as S3 (`_UN_BASES`) but an order of magnitude larger and
more directly the manipulation task's vocabulary. The swap-contrast
*mechanism* is sound and measured; the table it runs on is authored.

**Fix direction:** derive antonym pairs from corpus co-occurrence or an
embedding-space reflection, not a list.

---

## S1c — sentence TEMPLATES generate the student's training text

`scripts/distill_student.py:60` —

    "put the {a} on the {b}", "pick up the {a}",
    "put the {a} into the {b}", "take the {a} out of the {b}",
    "open the {a}", "close the {a}", ...

Ten hand-written sentence frames used to synthesise the text side of
student training. A student trained on these can only ever have seen
this grammar, so its text tower inherits the task taxonomy directly as
training data — the strongest form of hardwiring in the system, because
it is baked into weights rather than into an `if`.

---

## S2b — the colour table

`python/elidedb/scenario.py:975` — `_HUE`, seven colour words mapped to
OpenCV hue bands.

Defended in-comment as "closed-class colour words... no model, no
metadata, fully explainable", and that defence is the strongest of any
here: the bands are physics, the word list is closed-class English, and
the evidence is pixels. **Listed as borderline, not as a violation to
fix** — but noted because it is a hand-authored vocabulary, and if the
corpus's colour distinctions do not match these seven bands, nothing in
the system can discover that.

---

## What is CORRECT and should not be "fixed"

Recorded so a later pass does not break it:

- **`scripts/bridge_ingest.py` refuses the task strings.** Its docstring:
  the task strings "describe what each clip *is about*, which is
  precisely the thing the database is supposed to work out from the
  pixels. Ingesting them would make every later retrieval number
  meaningless." They are written to `eval/` **outside the store**, and
  nothing on the query path can read them. This is the rule applied
  correctly, and it is why S0 matters — the metadata was kept out of the
  store and then let back in through a prompt.
- Bench scripts reading `t["task"]` — eval side, correct.
- `_STOP` in `sig2.py` — generic English stopwords, no task content.
- `scripts/build_objects.py` `PROMPT = "object"` — deliberately
  vocabulary-free, and commented as such.

---

## Revised fix order

| | what | why first |
|---|---|---|
| **1** | **S0** caption prompt | eval vocabulary reached the index; contaminates the lexical channel and every number computed from captions |
| **2** | **S1** verb set + if/else | root of the element taxonomy; contaminates `chanbench` task labels |
| **3** | **S1c** student templates | hardwiring baked into weights is the hardest to undo later |
| **4** | **S1b / S3** swap and un- tables | mechanism is sound, only the lists need deriving |
| **5** | **S2 / S2b** query-side maps | fall out of 2 |
| **6** | **S4 / S5** thresholds, config | mechanical |

**Standing caution until 1 and 2 are fixed:** any retrieval number that
routes through captions, or any `task_auc` on
`open`/`close`/`put_into`/`put_on`, is partly circular. `contact` and
`release` remain the only labels derived from geometry rather than from
an authored vocabulary.

---

# Round 3 — LOGIC violations, 2026-07-31

Rounds 1 and 2 audited constants and vocabularies. This round audits
CODE LOGIC written for this dataset: procedures that are correct for a
fixed-camera table-top manipulation set and silently wrong elsewhere.
These are harder to see than a literal, because nothing in them looks
like a task word.

Two were found by the user in the space of one message, which is the
measure of how well the first two rounds did.

---

## L0 — THE SYSTEM ONLY RECORDS WHAT MOVES

**The largest violation in the codebase. Everything else is detail.**

The element path is triggered entirely by motion: `agent_track` finds
the largest coherent-motion track, `causal_participants` finds what that
track touched, `cavity_series` measures change. An object that sits
still produces no track, no participant, no event, no label, no row.

So **"all samples where a banana is on the table" is unanswerable by
construction** - not badly answered, structurally absent. The corpus can
only be asked about things that happened, never about things that were.

This is a manipulation-dataset assumption in the deepest possible place:
that the unit of interest is an ACTION. For a driving log the parked
cars matter; for a warehouse the stock on the shelf matters; for a
kitchen the banana matters.

**Fix direction:** state and event are two different indexes and the
store has only one. Idle objects need a presence record - what was
visible, where, for how long - independent of whether anything happened
to it. The identity store is the natural home (it already tracks
objects that persist) but it is currently fed only from the same
motion-gated path.

---

## L1 — `single_cls=True` is a TRAINING flag and does nothing at predict

Measured, same frames, same model:

    single_cls=True   3.10 det/frame  {oven, sink, bowl, person, spoon, wine glass}
    single_cls=False  3.10 det/frame  {oven, sink, bowl, person, spoon, wine glass}

Byte-identical behaviour. The detector still fires only on COCO-shaped
things; the flag hides the label at training time, not at inference.

I described this as "class-agnostic" in identity.py, in the memory
entries, in commit messages and in the earlier rounds of this document.
It is **COCO detection with the label suppressed**, so every structure,
participant proposal and identity region inherits COCO's 80-class prior
- a hardwiring violation introduced while fixing hardwiring violations,
invisible because I believed my own description of it.

Consequence already measured: `_structures` returns 0.7 regions per
episode, and the ones it finds are `oven` and `sink` at ~10% of frame
area. Drawers, tabletops and bins are not COCO classes, so there was
never a container there to find.

**Fix direction:** a genuinely class-agnostic proposer. SAM-family
"everything" mode is now permitted (it was excluded on COST, not
principle) provided it is made fast and then distilled. Alternatives
that need no class list at all: CutLER/MaskCut (unsupervised instance
segmentation), classical region proposals, or self-training YOLO's
objectness on the corpus - though NOT on motion alone, see L0.

---

## L2 — the transition DESCRIPTOR is a hand-authored feature space

`transitions.FIELDS` - disp, dir_x, dir_y, duration, scale_ratio,
enclosure_delta, persists, agent_d0, agent_d1.

Nine numbers *I* chose as what matters about a transition. The verb
ladder was replaced with a feature ladder: less wrong, same kind of
wrong. The corpus is still being told what to measure, only the
quantisation was handed back to it.

**Fix direction:** cluster transitions in a LEARNED visual space - the
trunk's representation of the transition's frames - so the corpus
decides the features and the types together.

---

## L3 — `enclosure` was inherited, not motivated

Carried forward from `inside(c1, abox) -> put_into`, the very ladder
being deleted. Containment is a manipulation primitive; it was never
justified on its own terms. Measured non-zero in **7 of 78** participant
transitions, i.e. carrying no information, and the effort to feed it
better was about to be spent on a detector that cannot see containers
(L1).

**Fix direction:** drop it. Re-derive a spatial-relation signal only if
a corpus demonstrates it needs one.

---

## L4 — `cavity_series` measures DARKNESS as aperture

    thr = 25th percentile of intensity; series = fraction of pixels darker

An open drawer is darker inside than its front panel. That is an optical
fact about *this* furniture under *this* lighting. An opening iris, a
lane gap, a raised barrier or a lit doorway all change aperture without
getting darker - and a shadow falling across the box registers as an
aperture event that never happened.

**Fix direction:** aperture is a geometric change in a structure's free
extent, not a photometric one.

---

## L5 — single-agent, agent-causes-everything

`agent_track` returns `max(tracks, key=len)`: exactly one agent, and it
is whichever coherent motion persisted longest. `causal_participants`
then attributes participants to that agent.

Assumes one actor that causes what happens. Ego-motion in a driving log
is the largest coherent motion in every frame, so the ego vehicle would
be "the agent" and every other car a "participant" it caused. Multi-
agent scenes have no single answer.

**Fix direction:** agents are a discovered set, not a maximum; causal
attribution needs to survive there being several, or none.

---

## L6 — the episode is assumed to exist, and to be the query unit

`gapped()` shifts demos apart by `GAP_S`; `full_write` samples NGEOM
frames uniformly per episode; retrieval returns episodes; the truthset
is keyed by episode; row groups are clustered by `episode_index`.

Bridge ships discrete demos. A continuous driving log, a surveillance
feed or a surgical recording has no episodes - and uniform sampling
within one assumes events are spread evenly through it, which is false
even here (a grasp is brief).

**Fix direction:** the query unit is a TIME WINDOW; episodes are one
possible segmentation of the timeline, supplied by a corpus that has
them, not assumed by the engine.

---

## L7 — `MAX_AREA = 0.5` rejects anything filling half the frame

`identity.py`. Written to reject "the whole scene" detections. In a
driving log the road surface, the sky and the vehicle ahead all exceed
it; in close-up manipulation the manipulated object does. A size prior
about how big things are in this camera's view.

**Fix direction:** fit from the observed area distribution, or drop -
a "too big to be an object" rule needs the corpus's opinion, not mine.

---

## Revised fix order, all rounds

| | what | why |
|---|---|---|
| **1** | **L0** state vs event | the largest gap; idle objects are unqueryable by construction |
| **2** | **L1** real class-agnostic proposer | L0's fix needs one, and every region in the system currently carries a COCO prior |
| **3** | **L3** drop enclosure, **L2** drop the descriptor | both are inherited scaffolding; removing them is subtraction, not work |
| **4** | **S0/S1** already fixed - verify no regression | |
| **5** | **L4, L5, L6, L7** | each needs a corpus that disagrees with Bridge to test against |

**L0 and L1 are the same fix seen twice:** the system cannot record what
it cannot detect, and it cannot detect what COCO does not name.
