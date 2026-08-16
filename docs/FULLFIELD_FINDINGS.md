# Full-field re-run: which pilot runs are actually seizures

Twelve pilot runs re-run with the whole 56x56 sheet recorded, then adjudicated
against the sheet rather than against the 30-electrode montage. Every run
reproduced its scored montage bit-for-bit (NaN pattern included), so these
panels describe the same realizations that are in the blind adjudication set --
not new simulations that merely resemble them.

## Headline

**The montage is not just under-sampled; on 5 of 12 runs it is actively
misleading.** The montage samples 30 of 3136 units (0.96%), so a flat montage
bounds only the recorded sites.

* The three runs the automated classifier calls `QUIET_OR_BLOCK` are the
  **most active runs in the set**: 100% of the sheet above threshold, 96-100%
  per-unit duty cycle, 81-87 Hz whole-sheet mean rate. They are in
  **depolarization block** -- tonically saturated, hence no discrete
  discharges at the electrodes, hence a flat montage.
* The two `draw16` runs, classified `ICTAL`, generalize to 100% of the sheet
  within a second and then run **spiral / reentrant waves**. That is a
  generalized non-Liou morphology, not the paper's focal-onset seizure.
* Only `anchor_default_{seq,sim}` show the paper's signature: focal onset,
  gradual territory expansion over tens of seconds, partial recruitment.

This is not a failure of the sweep design. `LIOU_PILOT_SPEC.md` predicted
exactly this for the g_E=220 anchor -- "ZERO discharges, CV 0.00, Cl 42.7 mM"
-- and named the two BLOCK anchors "the discriminating test", warning that any
score built on mean rate or chloride will call them seizure. The sheet data
confirm the anchors behave as designed; it is the montage-based *detector*
that cannot tell block from quiescence.

## Per-run verdicts

| run                   | montage class   | sheet verdict                    | peak recruit   | duty   |   mean Hz |   Liou features |
|:----------------------|:----------------|:---------------------------------|:---------------|:-------|----------:|----------------:|
| anchor_gIlocal600_seq | ENTRAINED       | FOCAL / entrained, no spread     | 5%             | 0%     |       0.5 |               1 |
| anchor_gIlocal600_sim | ENTRAINED       | FOCAL / entrained, no spread     | 17%            | 0%     |       3.6 |               3 |
| anchor_gKmax120_seq   | ENTRAINED       | FOCAL / entrained, no spread     | 7%             | 0%     |       0.8 |               2 |
| draw04_seq            | ACTIVE_NONICTAL | FOCAL / entrained, no spread     | 7%             | 0%     |       0.7 |               2 |
| draw09_sim            | ACTIVE_NONICTAL | FOCAL / entrained, no spread     | 15%            | 0%     |       3.7 |               3 |
| draw16_seq            | ICTAL           | GENERALIZED, non-Liou morphology | 100%           | 43%    |      31.8 |               1 |
| draw16_sim            | ICTAL           | GENERALIZED, non-Liou morphology | 100%           | 46%    |      33   |               1 |
| anchor_default_seq    | ICTAL           | SEIZURE (Liou-like)              | 77%            | 21%    |      11.8 |               4 |
| anchor_default_sim    | ICTAL           | SEIZURE (Liou-like)              | 79%            | 29%    |      17.5 |               4 |
| anchor_gE220_seq      | QUIET_OR_BLOCK  | TONIC SATURATION (not a seizure) | 100%           | 100%   |      81.3 |               1 |
| anchor_gE220_sim      | QUIET_OR_BLOCK  | TONIC SATURATION (not a seizure) | 100%           | 100%   |      81.3 |               1 |
| draw01_seq            | QUIET_OR_BLOCK  | TONIC SATURATION (not a seizure) | 100%           | 96%    |      87.5 |               2 |

Legend: peak recruit = maximum simultaneous fraction of the sheet above 10% of
f_max; duty = median over units of the fraction of time above that threshold;
Liou features counted out of four (focal onset, expands, partial territory,
sustained). Self-termination is excluded from the count -- see below.

## Comparison to Liou et al. 2020

Measured on `anchor_default_seq` (the one run whose raw field was retained),
using the same `wave2d` method validated against the paper in the Figure 6
reproduction:

| quantity | paper | 2-D port (validated) | pilot, default anchor |
|---|---|---|---|
| wavefront / territory expansion | 0.008 | 0.00767 | 0.0021 |
| fast inward wave | 1.36 | 1.364 | 0.86 (IQR 0.67-1.15, 164/280 discharges) |
| ratio | 170 | 178 | 409 |

Speeds are normalised units per second (sheet extent = 1.0).

**What matches.** Focal onset, an expanding recruited territory that saturates
below whole-sheet coverage, fast waves an order of magnitude faster than the
wavefront, and a chloride load that is spatially structured rather than
uniform.

**What differs.**

1. Territory expands ~4x more slowly than the paper (0.0021 vs 0.008), so the
   separation of scales is 2.4x wider than the published 170.
2. **No run self-terminates.** This is expected and not a discrepancy: the
   pilot drives the network continuously for 80 s with no blank intervals
   (`run_mnist` docstring: "no blank intervals and no training"), whereas the
   paper's seizure follows a transient stimulus. Self-termination cannot be
   assessed under this protocol.
3. The pilot's ictal territory saturates near 70-80% and stays there rather
   than covering a fixed subregion.

## Chloride distinguishes block from seizure

| run                   |   [Cl-] mean (mM) |   [Cl-] max (mM) |   [Cl-] spatial SD | sheet verdict                    |
|:----------------------|------------------:|-----------------:|-------------------:|:---------------------------------|
| draw01_seq            |             52    |            52.25 |               0.04 | TONIC SATURATION (not a seizure) |
| anchor_gE220_sim      |             38.99 |            43.91 |               0.76 | TONIC SATURATION (not a seizure) |
| anchor_gE220_seq      |             38.94 |            39.11 |               0.03 | TONIC SATURATION (not a seizure) |
| draw16_sim            |             20.65 |            27.88 |               0.93 | GENERALIZED, non-Liou morphology |
| draw16_seq            |             20.54 |            21    |               0.2  | GENERALIZED, non-Liou morphology |
| anchor_default_sim    |             16.6  |            24.94 |               1.66 | SEIZURE (Liou-like)              |
| anchor_default_seq    |             16.58 |            19.9  |               1.11 | SEIZURE (Liou-like)              |
| anchor_gIlocal600_sim |              9.12 |            25.55 |               3.58 | FOCAL / entrained, no spread     |
| draw09_sim            |              8.02 |            28.19 |               3.51 | FOCAL / entrained, no spread     |
| anchor_gIlocal600_seq |              7.02 |             8.24 |               0.23 | FOCAL / entrained, no spread     |
| anchor_gKmax120_seq   |              6.83 |             8.19 |               0.33 | FOCAL / entrained, no spread     |
| draw04_seq            |              6.69 |             7.53 |               0.15 | FOCAL / entrained, no spread     |

What separates the groups is the **mean** load, not the spatial variance:
depolarization block sits at 39-52 mM, the generalizing `draw16` runs at 20 mM,
the Liou-like seizure at 17 mM, and every focal/entrained run at 7-9 mM.

The spatial SD is a weaker and non-monotonic discriminator, and the table does
not support a simple reading of it:

* Two of the three block runs are loaded almost perfectly uniformly (SD 0.026
  and 0.039 mM), consistent with a sheet in uniform block -- but
  `anchor_gE220_sim` is at 0.762 mM, nearly 30x the group's lowest value
  (0.762 / 0.026 = 28.8), so "uniform" is not a property that holds across all
  three.
* The Liou-like seizure has the highest SD of any *saturating* run (1.11 and
  1.66 mM), which is the focal chloride load that makes the Liou seizure focal.
* Among the focal/entrained runs the SD **splits by presentation
  configuration**, not by class: the two `_sim` runs are the highest in the
  whole set (3.51 and 3.58 mM) while the three `_seq` runs are among the lowest
  (0.15-0.33 mM), overlapping the block group's low end. Whatever drives that
  split is not the block-vs-seizure distinction and is not explained here.

So the defensible statement is the one about the mean, plus duty cycle (see
above). Chloride spatial SD is worth recording per run but should not be used
as a second discriminator until the seq/sim split is understood.

## Human confirmation of the sheet verdicts (2026-08-16)

All 12 sheet verdicts were scored by hand from the full-field viewer, blind: the
automated call, the montage class and the measured metrics were withheld behind
an explicit reveal, and every row of the export records
`revealed_before_scoring = false`.

**Agreement: 12 of 12.**

The exported file recorded `ENTRAINED, not a seizure` for
`anchor_default_seq` and `anchor_default_sim`; the scorer identified both as
mis-clicks -- the calls were unambiguous from the panels, and both landed on the
first two runs scored, before the key positions were familiar. The interface was
at fault: two options began with the same word ("ENTRAINED, not a seizure"
and "FOCAL SEIZURE (Liou-like)") and sat at keys 1 and 3 with an unrelated
option between them. The options are now grouped by the distinction being judged
-- seizure versus not -- so a slip between those two crosses a visible divider,
and `u` undoes the last call. `liou_fullfield_scores_reconciled.csv` keeps both
`recorded_call` (what the page exported, unaltered) and `blind_call` (the
scorer's actual reading), so the correction is auditable rather than silent.

The independent measurement below was worked up while the two runs were still
disputed, and it agrees with the human reading. The discriminator is whether
activity recruits tissue **outside** the 784-unit driven patch (25% of the
sheet); fraction of time above 20 Hz in undriven tissue:

| class | runs | undriven active | peak simultaneous |
|---|---|---|---|
| focal entrained | 5 | 0.7 - 3.7% | 4 - 9% |
| **focal seizure (default anchors)** | 2 | **32 - 50%** | **77 - 78%** |
| generalized seizure | 2 | 73 - 76% | 100% |
| tonic saturation | 3 | 97 - 100% | 100% |

The two default anchors sit in an 8.6x gap with no run inside it
(0.0369 -> 0.3165), and they are the only runs clearly off the
driven-vs-undriven diagonal: the driven core leads and the surround is recruited
but never fills. That is focal onset with spread -- the paper's signature -- and
not entrainment, where undriven tissue stays near baseline.

## Recommendation for the blind adjudication

**Do not discard the scoring in flight, but do not treat it as ground truth for
seizure detection either.** Concretely:

1. The adjudication asks a human whether the *montage* shows a seizure. That
   question is still well-posed and the answers remain valid as a measure of
   what an electrode array reveals -- which is itself the experimentally
   realistic situation.
2. The automated `auto2` classes must not be used as sheet truth. Split
   `QUIET_OR_BLOCK` in downstream analysis: the sheet data show these runs are
   neither quiet nor ambiguous, they are in block.
3. Any detector used for the sweep at scale needs a **duty-cycle term**, not
   just amplitude and discharge statistics. Mean rate alone cannot separate
   block from seizure (block is *higher*); duty cycle separates them cleanly
   (block ~100%, seizure 20-46%, entrained ~0%). Chloride *mean* also separates
   them; chloride spatial SD does not, on these 12 runs -- see the section
   above.
4. Before scaling to 500 runs, record the whole-sheet summary scalars for every
   run -- peak simultaneous recruitment, median duty cycle, chloride spatial SD.
   They cost ~15 MB/run on top of the montage and they are what makes the class
   assignment defensible.

## Reproducing

    PYTHONPATH=. python run_fullfield.py      # 12 runs, ~270 s at 4 workers
    PYTHONPATH=. python render_fullfield.py   # per-run sheet panels

`las_model/pilot_worker.run_and_reduce_field` aborts if a re-run's montage does
not match the stored one, so a silent divergence cannot produce mislabelled
panels.
