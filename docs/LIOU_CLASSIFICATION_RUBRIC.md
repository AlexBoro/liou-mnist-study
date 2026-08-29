# Liou run classification — criteria

Stated by the user 2026-08-29, alongside the V3 scoring pass
(`liou_fullfield_scores 8_29_26.csv`, 32 runs, blind). **This supersedes the
earlier focal/generalized vocabulary.** The seizure subtype is now defined by
ONSET, not by spatial extent.

The four classes below are exclusive and exhaustive. Text in quotation marks is
the user's own wording and is authoritative; anything outside quotation marks is
commentary and may be corrected.

---

## 1. ENTRAINED, not a seizure

> "Can coexist with spreading tonic activation, but entrained activity continues
> throughout the epoch and fast waves don't appear and tonic activation is
> largely confined to the within the stimulated area."

Three conditions, all required: entrained activity persists for the whole epoch,
no fast waves at any point, and tonic activation stays largely inside the driven
patch. Note that the presence of *some* spreading tonic activation does not by
itself promote a run out of this class.

Reference runs: **V3-09, V3-17**.

---

## 2. TONIC SATURATION

> "Persistent activation extends across the entire heat map. Entrained activity
> may coexist. Fast waves do not appear."

> "Some in some cases tonic saturation begins with <1 to 3 seconds of a fast
> seizure pattern and then evolve to tonic saturation. These run are classified
> as tonic saturation."

> "Tonic saturation can wax and wane."

Whole-sheet persistent activation, no fast waves. Coexisting entrained activity
does not change the call, and the activation need not be continuous — waxing and
waning still counts.

Reference runs: **V3-04, V3-18** (brief fast onset then tonic saturation);
**V3-11, V3-20** (waxing and waning).

---

## 3. SEIZURE, slow-onset subtype
`SEIZURE: early slow tonic wavefront -> fast traveling waves`

> "Slowly advancing tonic front / expanding region precedes (spatially and
> temporally) the appearance of fast moving waves."

The ordering is the criterion, and it is both spatial and temporal: the tonic
front must precede the fast waves, not merely coexist with them.

Reference runs: **V3-08, V3-13**; **V3-23, V3-27** — "beautiful examples of
traveling fast waves".

---

## 4. SEIZURE, fast-onset subtype
`SEIZURE: fast traveling waves at onset`

> "Fast moving waves appear without an antecedent well-developed slowly
> advancing tonic wave front."

> "Sometimes the fast moving activity appears is more of a blob than a wave on
> the movie view."

A blob-like fast propagation still counts as fast onset; the discriminator is
the absence of a well-developed antecedent tonic front, not the morphology of
the fast activity itself.

Reference runs: **V3-10, V3-26** (blob rather than wave); **V3-29** (clear
traveling fast waves).

---

## Precedence rule for mixed sequences

Two rules govern runs that change character during the 80 s, and they resolve in
opposite directions:

| sequence | class |
|---|---|
| "<1 to 3 seconds of a fast seizure pattern and then evolve to tonic saturation" | **TONIC SATURATION** |
| "clear seizures that later develop to tonic saturation" | **SEIZURE** |

So a run that ends in tonic saturation is classified by *what preceded it*: a
brief (<1–3 s) fast pattern is treated as part of the tonic saturation onset,
while a clear, developed seizure keeps the seizure label.

**This boundary is currently qualitative.** "Clear" and "<1 to 3 seconds" are not
yet a single threshold, and the two rules meet somewhere between 1 s and the
duration of a clear seizure. Any automated detector must either reproduce the
distinction or abstain on runs that fall in the gap — it must not silently split
them on an invented cut point. Resolving this needs either a stated duration
threshold or a handful of adjudicated boundary cases.

---

## What an automated detector must be able to see

Every criterion above is about **ordering or morphology within the run**, not
about an end-state scalar:

* fast waves present / absent — needs time-resolved spatial data, and at a frame
  rate that resolves the waves. The stored 80 ms envelope is below Nyquist for
  the ~17 Hz discharge rhythm, so **fast waves cannot be detected from the
  envelope alone**; this needs `rate_full`.
* tonic front *precedes* fast waves — needs the relative timing of two events.
* confined to the driven patch vs whole sheet — the driven/undriven recruitment
  split already computed (`fig_entrained_vs_focal.png`).
* waxing and waning — a single duty-cycle scalar over the run cannot represent
  it; a duty(t) trace can.

A one-scalar end-window rule cannot express classes 3 and 4 at all, and will
confuse class 2 with class 3 whenever a seizure ends in saturation.
