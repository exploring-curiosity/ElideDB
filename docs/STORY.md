# ElideDB

## The story of how we got here

---

### The question that did not change

Every robot, every vehicle, every camera on a factory floor is recording right
now. Almost none of it will ever be watched again.

Not because it is worthless. Because there is no way to ask it anything. A
team that wants to find the six times a robot dropped a plate has two options:
pay people to watch the footage, or pay people to label it first and hope the
labels anticipated the question. Both cost more than the answer is worth, so
the footage sits there, and the fleet keeps making the same mistake.

We set out to fix one thing: let anyone ask a pile of video *when did something
like this happen*, and get timestamps back.

Everything else about this project changed at least twice. That question never
did.

---

### It started as a storage problem

The first version was a database. The premise was that video is expensive to
read, so the win is reading less of it. Index the file so you can decode two
seconds out of an hour without touching the rest. Measure every byte you touch
and report how many you avoided. That is where the name comes from. The best
read is the read you never make.

That version worked. It was also solving the wrong problem.

We could pull any two seconds out of a corpus almost instantly. We just had no
way to know *which* two seconds to ask for. The bottleneck was never the
bytes. It was that nobody could express the question.

So the storage engine became the floor, and the real work moved upstairs.

---

### Words were the obvious answer

Type what you want, get clips back. Everyone builds this, and we built it
well: compositional queries, hybrid ranking, a language model to re-rank the
shortlist. It worked on the easy half of the problem. Ask for a kitchen, get
kitchens.

Then we started asking the questions operators actually ask, and it fell apart.

The failures were not random. They clustered around one thing: **verbs**. Ask
for a drawer being opened and you get drawers. Opened, closed, half open,
being ignored, all of them. To a model trained on captioned photographs, *open*
and *close* are nearly the same word, because they describe nearly the same
picture. The difference between them is not in any frame. It is in the order
the frames arrive.

That is the whole problem in one sentence, and it took us weeks and several
dead systems to see it clearly. **The thing our customers care about is the
thing that language and still images are structurally worst at describing.**
Not a tuning problem. A category error.

We also learned something we did not expect, which was that when we did use a
heavyweight language model, the right move was never to call it when someone
asks a question. It was to call it once, when the video arrives, and bake the
answer into the index. Same quality, thousands of times faster, and the cost
lands on ingest where it can be budgeted instead of on the user where it
cannot. That principle survived every rewrite that followed.

---

### So we stopped using words

If the question is about motion and order, ask it with motion and order. Show
the system a clip. Get back every other time something like it happened.

This is a smaller ask of the user and a much larger ask of the system. It also
happens to be the natural interface for the customer we care most about. A
robot does not type. It already has the clip. It is living in one.

That reframing is when the project became what it is now.

---

### Then we tried to train our way to the top, three times

We built a model to learn what "similar" means. It got better.

We built a world model, one that learns by predicting what happens next, on
the theory that anything that can predict a scene must understand it. It got
better.

We built a self supervised trainer and ran it on everything we had, twice, with
two genuinely different objectives.

Every single one of them showed the same result, and it took us far too long to
admit what it meant. **Each model got better on the data it had seen and worse
on data it had not.** It learned the corpus. Point it at a new customer, a new
building, a new robot, and the gains evaporated or went negative.

For a research project that is a disappointment. For a product it is fatal.
A system that has to be trained on your data before it works is a system that
has an onboarding period, a retraining bill, a drift problem, and a separate
model to version for every customer you sign. We were building the exact thing
we would not want to sell.

So we ran the experiment we should have run at the beginning. We took the
trained model out and measured what was left.

---

### The answer had been sitting there the whole time

What was left was better.

Two off the shelf models that we do not train and do not touch. One watches
what changes from moment to moment. One watches what things look like. We
combine them, normalise them carefully, and match them along time in a way that
tolerates the same action being done faster or slower without pretending a two
second event and a twenty second event are the same thing.

That is the product. **There are no ElideDB weights.** Nothing in the system is
fitted to your data.

We did not want this answer. We had spent weeks earning the right to a trained
model and it would have made a better slide. But it is a far better business.

- There is no onboarding. It works on a new corpus the day it is pointed at it.
- There is nothing to retrain and nothing to drift.
- There is no per customer model to build, host, version, or explain.
- Our cost of adding a customer is storage and compute, not a research project.

The clearest evidence is a number we did not expect to be able to report. On a
sealed test corpus where most of the tasks had never been seen before, the
system performs *slightly better* on the unfamiliar material than on the
familiar. There is no home field advantage, because there is no home field.

The whole thing runs on one machine and answers in about half a second.

---

### Proving it where it actually counts

A retrieval score is an argument. We wanted a demonstration.

So we put the memory behind a real robot policy in a persistent kitchen and
gave it jobs that require remembering something from earlier. With memory
turned on, it completed every one of them. With memory turned off, and nothing
else changed, it completed none of them.

That is the product thesis in one experiment. Not *the model retrieves well*.
**The agent cannot do the job without it.**

We then took the same memory and put it into a shape a company could actually
run: a real database behind it, the corpus in cloud storage, a live console,
and a deployment that survives being restarted. Along the way we learned the
unglamorous things you only learn by shipping, like the fact that a query which
feels instant on a laptop can take half a minute the first time it runs against
cloud storage, for reasons that have nothing to do with the model.

The most interesting behaviour we measured in that build was not accuracy. It
was **refusal**. The agent that knows when it has not found anything, and says
so instead of returning its best bad guess, is worth more than the one that is
slightly more accurate on average. It declines a large fraction of questions
and returns almost no junk on the rest. Buyers of this technology have been
burned by systems that always answer. Knowing when to shut up is a feature.

---

### What we know now that we did not know at the start

Five weeks of building, most of it spent being wrong in useful ways.

**The hard part of video search is not search.** It is that the property people
care about, what happened and in what order, is invisible to the tools the
industry reaches for first.

**Anything you train on a customer's data is a liability.** It buys you
in domain performance and sells your generalisation to do it. We measured this
four separate times across four different architectures before we believed it.

**Expensive intelligence belongs at write time.** Spend it once when the video
arrives, never when a user is waiting.

**A system that abstains is worth more than a system that guesses.**

**And the constraint we are left with is a good one.** After we had exhausted
every model side and query side improvement we could think of, the honest
conclusion was that the remaining ceiling is set by how much and how varied our
video is, not by our algorithm. That is a problem solved with volume and
plumbing. It is the kind of problem money fixes.

---

### Where this goes

Two customers exist for the same store.

**A robot's memory.** The robot asks whether it has been in a situation like
this before and gets its own past back, ranked, fast enough to act on. It never
has to encode anything, because its live experience is already in the store.
This works today.

**An engineering team's archive.** Find every instance of a failure across a
fleet's history so it can be reviewed, or turned into training data for the
next policy. This is where the recall bar is higher, and where the next
phase of work is aimed.

The near term roadmap is narrow on purpose.

1. **Raise recall from the closest matches to all of them.** We are strong at
   *show me the ten most similar moments* and not yet strong enough at *show me
   all thirty five*. We know precisely what closes this, and it is more and
   wider video rather than a cleverer algorithm.

2. **A pipeline that consumes the open internet's video and keeps only what it
   learned.** Video comes in, becomes a compact trace, and the original is
   deleted. The disk never holds the corpus. Designed, not yet running.

3. **Words as a way in, not as the mechanism.** Text will come back as an entry
   point, built on top of the motion representation rather than instead of it,
   so that asking for something being opened does not return everything that
   has ever been shut.

4. **Return the exact moment, not the whole recording.** The alignment that
   makes the match already knows where inside the clip it happened. Surfacing
   it is a product gap, not a research one.

5. **A faster core and a hosted service.** The storage format was deliberately
   kept language neutral so the engine can be rewritten underneath it without
   touching a byte of anyone's data.

---

### The short version

We spent five weeks trying to teach a model what similar means, and the thing
that finally worked was refusing to teach it anything.

That sounds like a retreat. It is the opposite. It means we have a video memory
that works on data it has never seen, costs nothing extra per customer, cannot
drift, and needs no labels from anyone. It answers in half a second on a single
machine, and we have measured an agent that can do its job with it and cannot
do its job without it.

The remaining gap between what we have and what the biggest customers need is
a volume problem, and we know exactly which volume.
