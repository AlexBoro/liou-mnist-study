# Liou-MNIST parameter exploration: PILOT SPECIFICATION

Purpose: verify that presence/absence of seizure is captured correctly BEFORE spending
500 draws x 2 configs. The pilot's job is to be judged by eye on LFP plots, not to
produce a boundary.

## Size and cost
20 random draws x 2 configurations (simultaneous, sequential) = 40 runs
+ 7 known-verdict anchors x 2 configurations = 14 runs
= 54 runs. At T_end=80 s that is ~30 min serial, ~6 min on 6 local cores.
Measured basis: 60 s simulated = 25 s wall, n=56, 1 core. The pilot does NOT need Modal;
the full 500 x 2 sweep (9.3 core-hours at 80 s) does.

## Pairing (matches the WC scheme)
Both configurations receive IDENTICAL parameter draws at a given draw index, seeded from
param_seed = 1000 + draw so the draw sequence is independent of the run seed. The
comparison is therefore PAIRED across configurations, differing in dynamics not in
sampled parameters -- this is exactly how the WC sweep was constructed, and it also makes
the 20-draw pilot a free subset of the eventual 500-draw sweep.

## Deliverable per run (this is what gets verified)
1. 30-channel LFP montage over the driven area, full run length, fixed amplitude scale
   across all runs so morphology is comparable by eye
2. Driven-area mean rate trace with the detected discharge times marked
3. Diagnostic strip: Cl_in, rate CV, mean V, g_K -- to separate seizure from block
4. Scored metrics: amplitude (primary), FWHM, IPI, IPI/image_duration ratio, latency,
   n_discharges; secondary spread descriptors; CV and Cl flags
5. A contact sheet of all 54 montages for one-pass eyeball verification

## Ground-truth anchors (must be in every pilot)
| anchor | expected | evidence |
|---|---|---|
| default, scale 17, sim  | ICTAL     | amp 12161, IPI 485, 174 discharges |
| default, scale 17, seq  | ICTAL     | sequential counterpart |
| g_I_local = 120         | ICTAL++   | amp 18527, largest in project; intrinsic 235 ms IPI |
| g_E = 220               | BLOCK     | ZERO discharges, CV 0.00, Cl 42.7 mM |
| phi0 = -52              | BLOCK     | CV 0.00, Cl 26.7 mM, 3 discharges/60 s |
| scale = 4               | NON-ICTAL | weak drive, below nucleation |
| g_I_local = 600         | MARGINAL  | amp 7238, IPI 995 -- low-amplitude boundary case |

The two BLOCK anchors are the discriminating test. They have HIGH mean rate and HIGH
chloride but zero discrete discharges; any score built on mean rate or on chloride level
will label them seizure. If the pilot's classifier calls them ictal, the scoring is wrong
regardless of how well it does on the rest.

## Acceptance criteria before scaling to 500
1. All 7 anchors receive their expected verdict in BOTH configurations
2. User's eyeball verdict on the 54 montages agrees with the automated verdict; every
   disagreement is examined individually rather than tuned away
3. No run's IPI sits within 10% of an integer multiple of image_duration without the
   duration-doubling control having been run on it
4. Ictal fraction is not 0% or 100% -- if it is, the ranges need re-centring, not the
   classifier re-tuning
