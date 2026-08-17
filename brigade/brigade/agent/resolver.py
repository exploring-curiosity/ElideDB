"""Turning what a human said into what the robot will do.

This is the load-bearing module. Everything else in Brigade is machinery around
the claim it makes: **the robot cannot act on an underspecified request without
reading its memory**, and you can switch the memory off and watch that happen.

The problem, concretely. A person says "put the bowl away". In the kitchen the
robot is standing in there are seven things it could physically do, six of them
involving the words in that sentence:

    close the bottom drawer of the cabinet
    put the black bowl in the bottom drawer of the cabinet
    put the black bowl on top of the cabinet
    put the wine bottle in the bottom drawer of the cabinet
    put the wine bottle on the wine rack
    put the black bowl in the bottom drawer of the cabinet and close it

"Away" is not in any of them. Nothing in the request picks one. The scene does
not pick one either — the bowl can go in the drawer or on the cabinet, both are
legal, both are reachable, and a policy handed the raw phrase has no way to know
which was meant. Only history knows: this household puts bowls in the bottom
drawer, and it knows that because the robot did it before and wrote it down.

So resolution is a retrieval, not a lookup:

    1. embed the request                              (MiniLM, 384-d)
    2. nearest past OUTCOMES by cosine                (pgvector HNSW, `<=>`)
    3. group by the instruction that was executed, score = best hit
    4. corroborate with the norm for the subject      (`norms`)
    5. below the floor -> ABSTAIN, do not invent

Step 5 is the honest part. A resolver that always returns something would make
the memory-off comparison meaningless; this one reports that it has never seen
anything like the request and declines, which is the correct behaviour for a
robot with no relevant experience.

Nothing here contains a task list, an object list, or a mapping from words to
LIBERO strings. Every candidate instruction is one the robot previously executed
successfully, recorded by `Pilot` at the time. Delete the rows and this module
resolves nothing — which is exactly the demonstration.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from ..memory.store import Memory, Recalled

log = logging.getLogger("brigade.resolver")

# Locative prepositions, used to split "put X <prep> Y" into subject and place.
# This is ordinary English, not knowledge about kitchens: it holds for "put the
# book in the caddy" as much as "put the bowl in the drawer". Longest first so
# "on top of" wins over "on".
PREPS = (" on top of ", " in front of ", " to the front of ", " to the right of ",
         " to the left of ", " underneath ", " under ", " inside ", " into ",
         " in ", " on ", " at ")

_ARTICLES = re.compile(r"^(the|a|an)\s+", re.I)
_VERBS = re.compile(r"^(put|place|pick up|pick|move|set|stack|push|open|close|turn on|turn off)\s+", re.I)


def _strip(s: str) -> str:
    return _ARTICLES.sub("", s.strip().rstrip(".")).strip()


def split_instruction(text: str) -> tuple[str | None, str | None]:
    """"put the black bowl in the bottom drawer" -> ("black bowl", "bottom drawer").

    Generic verb-object-preposition parsing over the *human's own sentence*. It
    is deliberately not a dictionary of LIBERO objects: the robot learns the
    nouns of its house from the sentences it is given, so a house with a
    samovar in it works the same way.

    Clauses are separated first, because real instructions are compound —
    "open the top drawer AND put the bowl inside" names a fixture in clause one
    and the object in clause two. Splitting on the preposition alone yields the
    subject "top drawer and put the bowl inside", and a norm learned from that
    is garbage that then outranks the good ones.
    """
    clauses = [c for c in re.split(r"\s+and\s+(?=\w)", text.strip()) if c.strip()]
    fallback = None
    for clause in clauses:
        body = _VERBS.sub("", clause.strip())
        for p in PREPS:
            if p in body:
                subj, place = body.split(p, 1)
                subj, place = _strip(subj), _strip(place)
                # A clause naming both an object and a place is the one that
                # teaches a norm; "...inside" alone names no place.
                if subj and place:
                    return subj, place
                fallback = fallback or (subj or None, None)
        fallback = fallback or (_strip(body) or None, None)
    return fallback or (None, None)


def head_noun(phrase: str) -> str:
    """The last word of a noun phrase: "black bowl" -> "bowl".

    Used so a norm learned about a "black bowl" is still found when someone
    says "bowl". Crude, and English-only, but it is a property of the language
    rather than of this dataset.
    """
    words = [w for w in re.split(r"\W+", phrase or "") if w]
    return words[-1].lower() if words else ""


def tokens(phrase: str) -> list[str]:
    return [w.lower() for w in re.split(r"\W+", phrase or "") if w]


@dataclass
class Candidate:
    instruction: str
    score: float
    n_support: int              # how many past successes back this reading
    evidence: list[str] = field(default_factory=list)
    # Which LIBERO goal this instruction was carried out against, remembered
    # from when the robot did it. See Resolution.goal for why this matters.
    suite: str | None = None
    task_id: int | None = None


@dataclass
class Resolution:
    """What the robot decided, and everything needed to audit the decision."""
    request: str
    instruction: str | None     # None => abstained
    confidence: float
    candidates: list[Candidate]
    recalled: list[Recalled]
    norm: dict | None
    rationale: str
    latency_ms: float
    memory_used: bool = True
    # The goal the episode will be judged against. LIBERO's success check comes
    # from the scene's BDDL predicate, NOT from the sentence handed to the
    # policy — verified by reading `LiberoEnv.step`, where
    # `is_success = self._env.check_success()`. So a resolution has to name a
    # goal or the run has no defined notion of having worked. The robot
    # remembers which goal each instruction was carried out against, so this
    # too comes out of memory rather than out of a table in the source.
    suite: str | None = None
    task_id: int | None = None
    # What the request became once pronouns and ellipsis were expanded against
    # memory. Shown to the viewer because "put it back" -> "put the bowl back"
    # is where the memory read is most visible.
    expanded: str | None = None
    rewrite_note: str | None = None

    @property
    def abstained(self) -> bool:
        return self.instruction is None

    @property
    def goal(self) -> tuple[str, int] | None:
        return (self.suite, self.task_id) if self.suite is not None \
            and self.task_id is not None else None

    def to_json(self) -> dict:
        return dict(
            request=self.request, instruction=self.instruction,
            confidence=round(self.confidence, 4), abstained=self.abstained,
            rationale=self.rationale, latency_ms=round(self.latency_ms, 2),
            memory_used=self.memory_used, suite=self.suite, task_id=self.task_id,
            expanded=self.expanded, rewrite_note=self.rewrite_note,
            norm=dict(self.norm) if self.norm else None,
            candidates=[dict(instruction=c.instruction, score=round(c.score, 4),
                             n_support=c.n_support) for c in self.candidates],
            recalled=[dict(id=r.id, text=r.text, score=round(r.score, 4),
                           outcome=r.outcome, subject=r.subject) for r in self.recalled],
        )


class Resolver:
    """Resolves a human request against the robot's own history."""

    def __init__(self, memory: Memory, floor: float = 0.30, k: int = 12):
        self.mem = memory
        self.floor = floor
        self.k = k

    # ------------------------------------------------------- anaphora/ellipsis

    def antecedent(self) -> dict | None:
        """What "it" refers to: the last thing the robot actually handled.

        A pronoun has no referent in the pixels. "Put it back" contains no noun
        at all, so a policy handed those three words has nothing to act on, and
        no amount of looking at the kitchen will supply the missing argument —
        the referent is in the conversation, and the conversation is in the
        database. This is the sharpest form of the whole claim.
        """
        rows = self.mem.db.query(
            """SELECT subject, payload, ts FROM events
               WHERE kitchen_id = %s AND kind = 'outcome' AND subject IS NOT NULL
               ORDER BY ts DESC LIMIT 1""",
            (self.mem.kitchen,),
        )
        return rows[0] if rows else None

    def rewrite(self, request: str) -> tuple[str, str | None]:
        """Expand pronouns and ellipsis against memory. -> (rewritten, note).

        Two ordinary constructions, both resolved from the robot's own history
        rather than from any list written here:

          "put it back"      a pronoun with no noun -> the last subject handled
          "and the bottle too"  a noun with no verb -> the last verb used

        If memory is empty, both are left alone and the request goes through
        untouched — which is exactly what the memory-off arm sees.
        """
        text = request.strip()
        low = text.lower()

        # pronoun: no known noun in the sentence, but a pronoun standing in
        if re.search(r"\b(it|that|them|those|the same)\b", low) and not self._names_something(low):
            ante = self.antecedent()
            if ante and ante["subject"]:
                subj = ante["subject"]
                out = re.sub(r"\b(it|that|them|those|the same)\b", f"the {subj}", text,
                             count=1, flags=re.I)
                return out, f'"{ante["subject"]}" — the last thing handled'

        # ellipsis: names a thing, carries no verb ("and the bottle too")
        if self._names_something(low) and not _VERBS.search(_ARTICLES.sub("", low).strip()) \
                and not low.startswith(("where", "what", "which", "did", "have", "is")):
            rows = self.mem.db.query(
                """SELECT payload FROM events
                   WHERE kitchen_id = %s AND kind = 'outcome'
                   ORDER BY ts DESC LIMIT 1""",
                (self.mem.kitchen,),
            )
            if rows and (rows[0]["payload"] or {}).get("request"):
                prev = rows[0]["payload"]["request"]
                verb = _VERBS.match(_ARTICLES.sub("", prev.strip()))
                if verb:
                    noun = re.sub(r"^\s*(and|also)\s+", "", text, flags=re.I)
                    noun = re.sub(r"\s+(too|as well|also)\s*$", "", noun, flags=re.I)
                    return f"{verb.group(1)} {noun}", f'verb "{verb.group(1)}" carried over from "{prev}"'
        return text, None

    def _names_something(self, low: str) -> bool:
        """Does this sentence mention anything the robot has a name for?"""
        known = {n["label"] for n in self.mem.norms()}
        known |= {b["label"] for b in self.mem.beliefs()}
        return bool(known & set(tokens(low)))

    # ------------------------------------------------------------- questions

    def is_question(self, request: str) -> bool:
        return bool(re.match(r"^\s*(where|what|which|did|have|is|was)\b", request.strip(), re.I))

    def answer_where(self, request: str) -> dict:
        """"where is the bowl?" — answered purely from `object_beliefs`.

        No episode runs. This is the beat that cannot be faked by a policy that
        happens to be looking at the right thing: it is a read of what the robot
        recorded when it put the object down, and with memory off there is
        simply no answer to give.
        """
        t0 = time.perf_counter()
        known = {n["label"] for n in self.mem.norms()} | {b["label"] for b in self.mem.beliefs()}
        hit = next((t for t in tokens(request) if t in known), None)
        beliefs = self.mem.where_is(hit) if hit else []
        ms = (time.perf_counter() - t0) * 1e3
        if not beliefs:
            return dict(answered=False, label=hit, latency_ms=round(ms, 2),
                        text=(f"I have no memory of where the {hit} is" if hit
                              else "I do not know what you are asking about"))
        b = beliefs[0]
        return dict(answered=True, label=hit, place=b["location"],
                    pos=[round(float(v), 4) for v in (b["pos"] or [])],
                    confidence=float(b["confidence"]), stale=bool(b["stale"]),
                    last_seen=str(b["last_seen"]), latency_ms=round(ms, 2),
                    text=f"the {hit} is on the {b['location']}")

    # ---------------------------------------------------------------- resolve

    def resolve(self, request: str, *, use_memory: bool = True) -> Resolution:
        """Decide what to execute for `request`.

        `use_memory=False` is not a simulation of failure — it is the ablation.
        The resolver is handed no recall and can only pass the human's raw words
        through to the policy, because that is genuinely all that is left.
        """
        t0 = time.perf_counter()

        if not use_memory:
            return Resolution(
                request=request, instruction=request, confidence=0.0,
                candidates=[], recalled=[], norm=None, memory_used=False,
                rationale="memory disabled — the raw request goes to the policy "
                          "verbatim, because nothing else is available",
                latency_ms=(time.perf_counter() - t0) * 1e3,
            )

        # 0. expand pronouns and ellipsis first. "put it back" carries no noun,
        # so there is nothing to embed until memory supplies the referent.
        expanded, note = self.rewrite(request)

        # 0b. If the human named BOTH the thing and the place, the request is
        # already an instruction and memory has no business editing it. This
        # keeps the claim honest in the direction that matters: memory is not a
        # filter every command passes through, it is what fills a gap. Without
        # this, "put the bowl on the stove" gets "corrected" to whatever the
        # robot did most recently with a bowl, which is worse than no memory.
        subj, place = split_instruction(expanded)
        if subj and place:
            return Resolution(
                request=request, instruction=expanded, confidence=1.0,
                candidates=[], recalled=[], norm=self._norm_for(expanded),
                rationale=(f"{note + '; ' if note else ''}complete instruction — "
                           f"names both the object ({subj}) and the place ({place}), "
                           f"so nothing needed recalling"),
                latency_ms=(time.perf_counter() - t0) * 1e3,
                expanded=expanded, rewrite_note=note,
            )

        # 1-2. nearest past successes. `floor=0` here so the caller sees the
        # near misses too: a resolution that abstained is much easier to trust
        # when you can see what it *did* find and how far short it fell.
        recalled = self.mem.recall(expanded, k=self.k, floor=0.0)
        successes = [r for r in recalled
                     if r.outcome == "success" and r.payload.get("instruction")]

        # 3. one candidate per distinct instruction, scored by its best hit.
        by_instr: dict[str, Candidate] = {}
        for r in successes:
            instr = r.payload["instruction"]
            c = by_instr.get(instr)
            if c is None:
                by_instr[instr] = Candidate(
                    instr, r.score, 1, [r.id],
                    suite=r.payload.get("suite"), task_id=r.payload.get("task_id"))
            else:
                c.score = max(c.score, r.score)
                c.n_support += 1
                c.evidence.append(r.id)
        candidates = sorted(by_instr.values(), key=lambda c: (-c.score, -c.n_support))

        # 4. the norm for whatever the request is about, as corroboration.
        norm = self._norm_for(expanded)

        latency = (time.perf_counter() - t0) * 1e3
        prefix = f"{note}; " if note else ""

        if not candidates or candidates[0].score < self.floor:
            best = candidates[0].score if candidates else 0.0
            return Resolution(
                request=request, instruction=None, confidence=best,
                candidates=candidates[:5], recalled=recalled[:5], norm=norm,
                rationale=(f"{prefix}nothing in memory resembles this "
                           f"(best {best:.2f} < floor {self.floor:.2f}) — "
                           f"the robot has not done anything like it before"),
                latency_ms=latency, expanded=expanded, rewrite_note=note,
            )

        # A norm is evidence about WHERE, so it can promote a lower-scoring
        # candidate that puts the subject in the right place over a higher one
        # that does not. This is what makes "put the bowl away" and "put the
        # bottle away" resolve differently from one mechanism.
        chosen, why = self._apply_norm(candidates, norm)

        return Resolution(
            request=request, instruction=chosen.instruction, confidence=chosen.score,
            candidates=candidates[:5], recalled=recalled[:5], norm=norm,
            rationale=prefix + why, latency_ms=latency,
            suite=chosen.suite, task_id=chosen.task_id,
            expanded=expanded, rewrite_note=note,
        )

    def _norm_for(self, request: str) -> dict | None:
        """Which of the things the robot has opinions about is this request about?

        Taking the head noun of the parsed subject is not enough: "put the bowl
        away" parses to the subject "bowl away", whose head noun is *away*, and
        the norm lookup then finds nothing — which is how "put the bowl away"
        resolved to the wrong place even with a correct norm sitting in the
        table. Measured, not theorised.

        Rather than keep a list of English particles to strip, the request is
        matched against the labels the robot has actually learned. That list is
        its own experience, so the robot recognises the words it has opinions
        about and ignores the rest — no vocabulary is written down anywhere, and
        a house full of samovars works the same way once it has handled one.
        """
        known = {n["label"]: n for n in self.mem.norms()}
        if not known:
            return None
        # Longest match first, so a "wine bottle" norm beats a "bottle" one.
        for tok in sorted(set(tokens(request)), key=len, reverse=True):
            if tok in known:
                return known[tok]
        return None

    def _apply_norm(self, candidates: list[Candidate], norm: dict | None
                    ) -> tuple[Candidate, str]:
        top = candidates[0]
        if not norm:
            return top, (f"closest past success (cos {top.score:.2f}, "
                         f"{top.n_support} episode(s))")

        home = (norm["home_location"] or "").lower()
        agree = [c for c in candidates if home and home in c.instruction.lower()]
        if not agree:
            return top, (f"closest past success (cos {top.score:.2f}); "
                         f"no candidate matches the norm '{home}'")
        pick = agree[0]
        if pick is top:
            return top, (f"closest past success (cos {top.score:.2f}, "
                         f"{top.n_support} ep) and it agrees with the learned "
                         f"norm '{norm['label']} -> {home}' "
                         f"(confidence {norm['confidence']:.2f}, "
                         f"{norm['n_episodes']} episodes)")
        return pick, (f"norm '{norm['label']} -> {home}' "
                      f"({norm['n_episodes']} episodes, conf {norm['confidence']:.2f}) "
                      f"outranks the nearest text match "
                      f"({top.instruction[:40]}…, cos {top.score:.2f})")

    # ------------------------------------------------------------ after acting

    def learn(self, request: str, instruction: str, ok: bool, seconds: float,
              task_id: str | None = None, *, suite: str | None = None,
              goal_id: int | None = None) -> str:
        """Write the episode down so the next resolution can use it.

        The text is phrased the way somebody would type it into a search box,
        because it is literally what gets embedded and searched. A log line like
        `task=24 status=0` would retrieve nothing.

        `suite`/`goal_id` record which goal the robot was working towards when
        it did this, so a later resolution recovers not just the words but what
        counts as having done it.
        """
        subj, place = split_instruction(instruction)
        verdict = "success" if ok else "failure"
        text = (f"{instruction} — {verdict} in {seconds:.0f}s"
                + (f"; asked as '{request}'" if request != instruction else ""))
        eid = self.mem.remember(
            text, kind="outcome", subject=subj, outcome=verdict, task_id=task_id,
            payload=dict(instruction=instruction, request=request,
                         seconds=round(seconds, 2), place=place,
                         suite=suite, task_id=goal_id),
        )
        if ok and subj and place:
            # Only successes teach a norm. A failed attempt says nothing about
            # where things belong — only that this robot could not get it there.
            self.mem.learn_norm(head_noun(subj), place)
        return eid
