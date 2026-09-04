# Liou run classification — criteria

## Two stipulations, 2026-09-03 (read these first)

Stated by the user and binding on every classification from here on. Both narrow
the rubric below rather than replacing it.

**1. The first three seconds do not count.** Classification is judged on the
**evaluated portion, t = 3 s to the end of the run**. Anything in the first 3 s
of stimulation is disregarded — onset transients, an early fast burst, an early
tonic episode.

*This closes the mixed-sequence boundary that was previously open.* The old rule
said "under 1 to 3 seconds of a fast seizure pattern, then evolving to tonic
saturation" is tonic saturation, while a "clear seizure" that later saturates
stays a seizure — and "clear" never met "<1–3 s" at a stated threshold. Under
the stipulation there is no gap to straddle: a fast pattern confined to the
first 3 s is simply outside the window and cannot promote a run to seizure.
A fast pattern surviving past 3 s is evaluated on its own terms.

*Consequence for every stored statistic — measured, 2026-09-03.*
`unit_duty_median` and `peak_simultaneous_frac` in the run bundles are computed
over the **whole** run, so they no longer match the window the classification is
judged on. Measured over the 112 scored batch-4 runs, using the stored 80 ms
envelope (which tracks the 5 ms statistic at r = 0.97):

* The exclusion is nearly inert. Median duty change **+0.0003**, largest
  **0.037**; no run changes by more than 10% relative. Peak recruitment falls
  for only 3 runs and by at most **0.005**.
* **No run's event lives only in the excluded window.** Zero runs drop from
  substantial peak recruitment to none, so nothing moves from event to
  entrained. This is the check that mattered, and it came back clean.
* Duty moves *up*, not down, because the first 3 s contains the pre-recruitment
  ramp, which dilutes a whole-run average.

That last point has a trap in it. `DUTY_TONIC = 0.80` was calibrated on
whole-run duty. Recompute duty on t ≥ 3 s and keep that threshold, and two runs
scored `fast traveling waves at onset` (`draw102_seq`, `draw77_seq`, adjusted
duty 0.806 and 0.812) cross into tonic saturation — accuracy against the human
calls falls from 101/112 to 99/112. Restoring it needs the threshold at ~0.837.

So: **either evaluate the statistic on the whole run at DUTY_TONIC = 0.80, or on
t ≥ 3 s with the threshold recalibrated — do not mix them.** The whole-run
statistic is what the committed classifier is calibrated against and performs
identically, so there is no reason to change it on this rubric's account. The
0.837 figure was chosen to maximise accuracy on these same 112 runs, so it is
in-sample and would need a fresh batch to confirm.

**2. Tonic already present at t = 3 s, with fast waves later, is the
tonic → fast-wave subtype.** If tonic activation is present at the *beginning*
of the evaluated portion and fast waves appear after it, the run is
`SEIZURE: early slow tonic wavefront -> fast traveling waves`.

The reasoning: excluding the first 3 s can hide the tonic front *developing*,
so a run whose front arose before 3 s would otherwise look like fast waves
arriving without an antecedent front — the other subtype. The stipulation says
the ordering is what matters, not whether the front's growth was observed.
Reference run: **B4-019 = `draw68_seq`**, scored as this subtype.


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
