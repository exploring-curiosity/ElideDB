# artifacts/ — fitted state, derived from the corpus

Outputs of fitting and scoring runs. **Not code, not weights, not data** —
each file is something a script computed *from* a corpus that a later
script or the read path then consumes.

They live outside `python/elidedb/` because they are regenerable, and
outside `models/` because they are not network parameters. They are
checked in (unlike `lake/` and `models/`) because they are small and
because a benchmark number is not reproducible without the exact
thresholds it was measured under.

| file | written by | read by |
|---|---|---|
| `teacher_base.npz` | `scripts/teacher_eval.py` | `presence_stage.py`, `pairwise_boundary.py` |
| `teacher_labels.npz` | `scripts/distill_student.py` | `distill_student.py` (cache) |
| `itm_scores.npz` | `scripts/itm_*.py` | `itm_fuse.py`, `itm_eval.py` |
| `itm_scores3.npz` | `scripts/itm_full.py` | `teacher_eval.py`, `itm_pool_eval.py` |
| `itm_fullspan.npz` | `scripts/itm_fullspan.py` | `teacher_eval.py` |
| `verbs_v2.json` | `scripts/verb_recompute.py` | `match_events.py`, `desk.py`, `version_teacher.py` |
| `cavity.json` | `scripts/verb_recompute.py` | `desk.py`, `version_teacher.py` |
| `presence_margins.json` | `scripts/presence_stage.py` | `presence_stage.py` |

## Two standing rules apply here

**No dataset priors.** A fitted threshold is legitimate; a hand-written one
is a NO-HARDWIRE violation. Every value in these files must be the output
of a fit over corpus statistics, never a constant someone chose because it
worked on Bridge. See `VIOLATIONS.md`.

**Eval-only truthset.** Nothing here may be fitted on `eval/`. These are
fitted on the corpus; the truthset only ever measures.

## Language neutrality

`.npy`/`.npz` is a documented format readable from Rust (`rust/elide-vec`
already carries an `npy.rs`). `allow_pickle=True` is **not** — it admits
arbitrary Python objects and makes the artifact unreadable outside Python.
Any new artifact here should avoid it.
