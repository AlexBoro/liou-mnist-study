# Liou-MNIST parameter exploration: proposed axes and ranges

Design intent follows the WC-MNIST sweep: parameters drawn from UNIFORM distributions over
the stated ranges, WITHOUT regard to where the seizure boundary is thought to lie. Two
configurations (simultaneous, sequential). No training. Susceptibility scored by DRIVEN-AREA
DISCHARGE MORPHOLOGY (the user's ictality criterion), not by spread.

## Measured cost (n=56, patch projection, this machine, 1 core)

60 s of simulated time = 25 s wall. 500 draws x 2 configs = 7.0 core-hours; on 50 Modal
workers, well under an hour. Cost does not constrain the design, so run length should be
set by the physics (see RUN LENGTH below), not by budget.

## Axis screen: every candidate moves the readout

One run per level per configuration, 25 s, scale 17, seed 0, ranked by total log-deviation
of (amplitude, IPI) from the matched baseline. Full table in liou_axis_sensitivity.csv.

| axis | amp ratio sim | amp ratio seq | IPI ratio sim | IPI ratio seq |
|---|---|---|---|---|
| tau_Cl x3        | 0.23 | 0.34 | 2.06 | 1.47 |
| g_K_max x0.5     | 0.21 | 0.38 | 1.55 | 1.96 |
| scale 8          | 0.81 | 0.36 | 0.48 | 1.96 |
| Cl_in_eq x1.8    | 1.10 | 2.58 | 0.45 | 0.81 |
| tau_syn_I x0.5   | 0.76 | 0.63 | 1.08 | 1.96 |
| tau_Cl x0.5      | 0.81 | 0.76 | 2.06 | 1.02 |
| phi0 depol       | 0.86 | 0.73 | 2.05 | 1.02 |
| tau_K x0.4       | 0.99 | 0.86 | 1.28 | 1.96 |
| g_K_max x2       | 0.80 | 0.76 | 1.58 | 1.06 |
| scale 30         | 1.04 | 1.41 | 1.07 | 0.99 |
| beta x0.6        | 0.96 | 0.73 | 1.18 | 1.02 |

Baselines: simultaneous amp 12161, IPI 485 ms; sequential amp 7201, IPI 255 ms.

### Non-monotonicity -- affects how ranges must be sampled

g_K_max at 60 s: x0.5 -> amp 4734, IPI 1008; x1.0 -> amp 13448, IPI 490; x2.0 -> amp 9675,
IPI 760. Discharge amplitude PEAKS NEAR THE DEFAULT and falls off both ways. Two different
mechanisms: a weak brake lets chloride load until the tissue depolarises into block (few,
weak, slow events), a strong brake prevents recruitment. tau_Cl behaves the same way
(x0.5 and x3 both reduce amplitude). CONSEQUENCE: do not fit a monotonic susceptibility
index to these axes, and do not narrow the range around one tail -- the interesting
structure is a ridge, and uniform sampling across the full range is the right way to find it.

### Three axes act differently in the two configurations

scale (sim 0.81 vs seq 0.36), g_K_max x0.5 (0.21 vs 0.38), Cl_in_eq x1.8 (1.10 vs 2.58).
These are the axes where the two presentation modes genuinely differ, so they are the ones
that will carry a simultaneous-vs-sequential contrast. Do not drop them for being
inconsistent -- the inconsistency IS the finding.

## Proposed axes and ranges

Tier 1 -- the chloride/potassium slow machinery (the mechanism that generates the seizure):

| parameter | default | proposed range | why |
|---|---|---|---|
| tau_Cl   | 5000 ms | U(1500, 20000) ms   | chloride clearance; strongest single axis |
| g_K_max  | 40      | U(10, 120)          | potassium brake; strongest, non-monotonic |
| tau_K    | 5000 ms | U(1500, 20000) ms   | brake timescale |
| Cl_in_eq | 6.0 mM  | U(3, 12) mM         | chloride set-point; largest sequential effect |

Tier 2 -- excitability and synaptic balance (the WC bias analogues).
RANGES BELOW ARE THE POST-SCREEN ONES; the ranges I first proposed here were narrowed
after measurement (see "g_I_global vs phi0" and "Tier 2 gains" sections for the data):

| parameter | default | proposed range | why |
|---|---|---|---|
| g_I_local | 250    | U(120, 400) mV | PRIMARY: 7.2x rate authority, never blocks, highest amplitude in the model |
| phi0      | -45 mV | U(-48, -38) mV | secondary excitability, WC bias_e analogue; report Cl_early with it |
| g_E       | 100    | U(60, 130)     | highest authority (17.7x) but worst chloride confound; >140 is block |
| beta      | 2.5    | U(1.0, 5.0)    | gain/steepness of the transfer function |
| tau_syn_I | 15 ms  | U(7, 30) ms    | inhibitory decay; sets the I/E time ratio |
| g_I_global| 50     | 50 / 300 levels| control arm, NOT a uniform axis: only 2.8x over a 40x span |

Ranges REJECTED after measurement, recorded so they are not reintroduced:
phi0 down to -52 (block: CV 0.00, 3 discharges/60 s), g_E to 200 (zero discharges at 220),
g_I_local to 500-600 (weak, amplitude falls monotonically), g_I_global as U(20,120) (a 6x
parameter change buys 1.4x rate -- less than seed noise).

Tier 3 -- stimulus coupling (analogue of WC mnist_weights + input scale):

| parameter | default | proposed range | why |
|---|---|---|---|
| scale | (pass explicitly) | U(4, 37) | input drive; 37 is just under the 39.7 chloride wall |
| patch weights | rng | resample per draw | matches WC's frozen-but-sampled mnist_weights |

### sigma_I/sigma_E -- CORRECTED (was: "changes the model's identity")

I first said this ratio changes the model's identity rather than its operating point and
should be held fixed. That was asserted without measurement and it is WRONG. Measured:

Geometry (exact, n=56, g_E=100/g_I_local=250/g_I_global=50). The net centre weight is
g_E*K_E(0) - g_I_local*K_I(0) - g_I_global/N, and the ratio controls its SIGN:

| sigma_I/sigma_E | sigma_I (px) | net centre |
|---|---|---|
| 1.00 | 1.12 | -19.87 |
| 1.25 | 1.40 |  -7.52 |
| 1.50 | 1.68 |  -1.06 |
| 2.00 | 2.24 |  +5.08 |
| 2.50 | 2.80 |  +8.07 |
| 3.00 | 3.36 |  +9.62 |

The zero crossing is at ~1.55 and THE PAPER'S DEFAULT (0.03/0.02 = 1.5) SITS ALMOST
EXACTLY ON IT (-1.06). The model ships tuned to the knife edge between a net-inhibited
and a net-excited centre, which is precisely the property an operating-point axis has.
Note also the centre is never a net-excitatory peak at the default -- it is the LEAST
inhibited point, not a positive one, so calling the default profile a "Mexican hat" (as I
did) overstates it.

Dynamics (60 s runs, scale 17, seed 0, driven-area morphology; liou_sigma_ratio.npz):
discharge amplitude is flat in simultaneous (14347 -> 13267 across 1.0-3.0) but collapses
at ratio 1.0 in sequential (5363 vs 15989 at 1.5); IDI shortens 525 -> 377 ms
(simultaneous) and event counts move ~2x. So it moves the readout as much as several
Tier 2 axes.

Diagnostics say the high end is NOT more ictal. Driven-area rate CV falls 0.27 -> 0.09
and late mean V rises -54.3 -> -45.7 mV as the ratio goes 1.0 -> 3.0, with peak Cl_in
13.1 -> 19.5 mM and no chloride-floor hits. High ratio = broad disinhibited centre going
to TONIC SATURATION, which by the driven-area morphology criterion is less ictal, not
more, even though the mean rate is higher. This is the same non-monotone trap as g_K_max
and the reason CV must be logged per run.

RECOMMENDATION (revised): vary it, as sigma_I_frac ~ U(0.020, 0.045) with sigma_E_frac
held at 0.02 (ratio 1.0-2.25), which brackets the sign flip. Ratios above ~2.5 are mostly
tonic-saturation regime and are better covered by a couple of explicit levels than by
spending uniform draws there. IMPLEMENTATION NOTE: run_mnist hardcodes
add_standard_recurrent_connections(topology="circular") with no pass-through, so
sigma_E_frac/sigma_I_frac AND g_E/g_I_local/g_I_global are NOT reachable via run_mnist
kwargs. The sweep driver needs a connectivity-kwargs pass-through in run_mnist (preferred)
rather than the monkeypatch used for this screen.

## What to hold fixed

n=56 (locked configuration), patch projection with compensate='sqrt', cl_floor=0.1,
dt=1 ms, image_duration fixed ACROSS BOTH CONFIGURATIONS (see below), C/g_L/E_L/E_K/E_Esyn/
Cl_ex/T_refractory/f_max (biophysical constants, not operating-point knobs).

## RUN LENGTH -- must be >= 60 s

Nucleation under the digit drive takes ~15 s at scale 17 and longer at weaker drive. A 10 s
run, which is what the WC sweep used, CANNOT see it -- that is exactly how the earlier screen
here missed the seizure entirely. Recommend 60 s minimum, 80 s to match the characterised
runs. This is a deliberate departure from the WC protocol and the one place the two sweeps
should NOT be matched; note it explicitly when comparing.

## Entrainment control -- fix image_duration across configurations

Established discharge IPI (250-265 ms) sits near the 250 ms image refresh and scores phase
concentration R=0.446 there. Varying image_duration 250 -> 400 ms moves discharge IPI only
255 -> 285 ms, so the train is intrinsic with partial entrainment. But sequential
presentation changes the effective refresh by construction (row_duration =
image_duration/28 in the WC scheme), so the two configurations would carry DIFFERENT
entrainment baselines if image_duration were left to follow the mode. Fix it across both,
and run a duration-varying control arm per configuration.

## Scoring each run (driven channels only)

Primary: discharge amplitude, FWHM, inter-discharge interval, discharge rate, and latency
to the first discrete-discharge train. Secondary descriptors, NOT classifiers: spread
beyond the driven band, front speed, fraction of zero-input tissue recruited.
Also log: phase concentration at the refresh period (to flag entrainment), peak Cl_in,
fraction of samples at the chloride floor (to flag depolarisation block vs seizure).


## g_I_global vs phi0 as the excitability axis -- CORRECTED

I speculated that "because g_I_global is a field-mean projection, it may be a cleaner
excitability axis than phi0." Measured, and it is wrong in the way that matters.

STRUCTURE (las_model/model.py, MeanFieldModel.step). phi0 enters exactly one place:
phi_inf(u) = phi0 + u*phi_inf_gain, an additive offset on the dynamic threshold, so it
shifts V - phi in the activation and nothing else. g_I_global enters Input_I, which is
read THREE times in the same step:
  1. gI = Input_I/f_max          -> drives V toward E_Cl (Eq. 1 numerator)
  2. g_sum = g_L + gE + gI + gK  -> shunting: raises total conductance, shortens tau_V
  3. Eq. 3: Cl_in_inf ~ tau_Cl/Vd_Cl/F * Input_I * (V - E_Cl) + Cl_in_eq
Path 3 is decisive: Input_I IS the chloride-loading term. Any inhibitory-conductance
parameter is therefore also a seizure-fuel parameter in this model. There is no
inhibitory knob here that is "just" excitability -- that is a property of Liou's
chloride-accumulation mechanism, not of a particular parameter choice.

MEASURED (60 s, scale 17, seed 0, simultaneous; liou_excitability_axes.npz):

| axis | range | early rate | fold | early Cl_in | CV range | amp |
|---|---|---|---|---|---|---|
| phi0       | -52 .. -38 mV | 0.354 -> 0.035 | 10.1x | 26.7 -> 8.5 mM | 0.00 - 0.36 | 4351 - 13448 |
| g_I_global |  20 .. 800    | 0.102 -> 0.036 |  2.8x | 10.6 -> 9.2 mM | 0.15 - 0.23 | 9103 - 13700 |

Both findings run against my speculation:

(a) g_I_global is WEAK, not clean. Over its default-bracketing range 20-120 (6x) the
early rate moves only 1.4x; it takes 40x (20->800) to get 2.8x. phi0 gets 10x from a
14 mV shift. As a sweep axis, uniform draws on g_I_global mostly produce runs that
differ from each other by less than seed noise.

(b) The chloride coupling I claimed to avoid is WEAKER on g_I_global, not absent, and
phi0's is the strong one: d log(Cl_early)/d log(early_rate) = +0.44 for phi0 vs +0.19 for
g_I_global. So the confound I was worried about is real -- but it attaches to phi0, the
axis I recommended, not to g_I_global. Reason: phi0 changes the RATE, and rate drives
Input_I through recurrent inhibition, so chloride follows. g_I_global raises inhibitory
conductance per unit rate while LOWERING the rate, and the two effects partly cancel in
the Cl_in_inf product.

(c) phi0 crosses into saturation inside a plausible range: at phi0 = -52 mV the driven
area reaches CV 0.00 with Cl_early 26.7 mM and only 3 detected discharges in 60 s -- tonic
block, not seizure. g_I_global never crossed the CV 0.10 saturation floor anywhere in the
40x tested. So g_I_global IS better behaved; it just has too little authority to be the
primary axis.

RECOMMENDATION: keep phi0 as the Tier 2 excitability axis (it is the WC bias-e analogue
and it has the authority), but (i) restrict the range to phi0 ~ U(-48, -38) -- the -52
end is depolarisation block, not extra susceptibility -- and (ii) since phi0 moves
chloride with it, do NOT read a phi0 effect as "excitability alone"; report Cl_early
alongside every phi0 result. Treat g_I_global as a 2-3 level control arm (50 / 300)
rather than a uniform axis: its value is as a rate-matched comparison that changes
chloride much less, which is exactly the control needed to separate "more spikes" from
"more chloride" in any susceptibility claim.


## Tier 2 gains g_E and g_I_local -- SCREENED

Measured 60 s, scale 17, seed 0, simultaneous, n=56, patched connectivity
(liou_tier2_gains.npz / liou_tier2_gain_screen.csv).

| axis | range | early rate | fold | Cl_early | CV range | amp range | blocked levels |
|---|---|---|---|---|---|---|---|
| g_E        |  50 .. 220 | 0.025 -> 0.447 | 17.7x | 7.5 -> 42.7 mM | 0.00 - 0.35 |  4425 - 13448 | 2/5 |
| g_I_local  | 120 .. 600 | 0.201 -> 0.028 |  7.2x | 18.7 -> 8.4 mM | 0.15 - 0.25 |  7238 - 18527 | 0/5 |
| phi0       | -52 .. -38 | 0.354 -> 0.035 | 10.1x | 26.7 -> 8.5 mM | 0.00 - 0.36 |  4351 - 13448 | 2/5 |
| g_I_global |  20 .. 800 | 0.102 -> 0.036 |  2.8x | 10.6 ->  9.2 mM | 0.15 - 0.23 |  9103 - 13700 | 0/7 |

g_E: the highest-authority axis in the model (17.7x rate over 4.4x parameter) and the
worst-confounded (d log Cl / d log rate = +0.598, the largest of the four). Amplitude
peaks exactly at the shipped default 100 and falls off both sides; at 150 CV is 0.02 with
Cl_early 30.1 mM, and at 220 there are ZERO detected discharges in 60 s -- total
depolarisation block. Usable range is therefore narrow: g_E ~ U(60, 130). Above ~140 the
runs are not more seizure-prone, they are silent-and-saturated, and any monotonic score on
rate would rank them as maximal.

g_I_local: the best-behaved strong axis, and the one real surprise of this screen. It has
7.2x rate authority (2.6x more than g_I_global at 1/5 the parameter span), it produced the
LARGEST discharge amplitude anywhere in this project (18527 at g_I_local=120, vs 13448 at
every default), and it never entered block -- CV stayed 0.15-0.25 across the whole range.
Direction is inverted as expected (less local inhibition -> higher rate), and the
amplitude peak is BELOW the default: 120, not the shipped 250. Its chloride coupling
(+0.353) is intermediate -- below phi0, well below g_E.

ENTRAINMENT CONTROL (required here: g_I_local previously produced a refresh false positive
in this project). IMAGE_DURATION_MS = 250 ms, refresh 4.0 Hz. Doubling image duration to
500 ms:

| g_I_local | IPI 250 ms | IPI 500 ms | change | pacing predicts |
|---|---|---|---|---|
| 120 | 235 ms | 220 ms |  -6.4% | +100% |
| 180 | 255 ms | 225 ms | -11.8% | +100% |
| 250 | 490 ms | 255 ms | -48.0% | +100% |

At 120-180 the interval is duration-invariant: the discharges are INTRINSIC, near 230 ms
regardless of refresh, despite sitting close to the 250 ms image period. The default 250
is different -- 490 ms is 2x the image period, i.e. 2:1 locked, skipping every second
image, and it unlocks to 255 ms when duration doubles. So the shipped default is the most
entrained point on this axis, and moving g_I_local DOWN both raises amplitude and reduces
entrainment.

METRIC CHOICE (falls out of the same control): across the three levels, doubling image
duration changed discharge AMPLITUDE by +2.9 / -7.8 / -9.6% while changing IPI by
-6.4 / -11.8 / -48.0%. Amplitude is the entrainment-robust morphology metric; IPI is not.
Score primarily on amplitude; report IPI as a descriptor and always alongside the
IPI/image-duration ratio so n:1 locking is visible.

REVISED TIER 2 (replaces the phi0-only recommendation):
  g_I_local ~ U(120, 400)   PRIMARY -- strong, never blocks, amplitude-dominant, least
                            entrained at the low end
  phi0      ~ U(-48, -38)   secondary excitability, report Cl_early with it
  g_E       ~ U(60, 130)    narrow; exclude >140 (block), highest chloride confound
  g_I_global  50 / 300      2-3 level control arm, not a uniform axis

All four axes collapse onto a single amplitude-vs-rate ridge peaking near early rate 0.1
(figure panel f), which is the operational definition of the boundary this project is
looking for: susceptibility is maximal at intermediate rate, and BOTH tails are less
ictal. Any susceptibility index must be non-monotonic in rate, or must use amplitude
directly rather than rate as its input.
