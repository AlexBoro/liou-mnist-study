# Liou LAS-Model — Python port notes

Python port of the **rate (mean-field) model** from Liou et al. (2020),
*A model for focal seizure onset, propagation, evolution, and progression*,
eLife 9:e50927 (doi:10.7554/eLife.50927). Source of truth is the authors'
MATLAB repository <https://github.com/jyunyouliou/LAS-Model>.

Scope of this port: everything needed to reproduce **Figure 1C** (2D sheet) and
**Figure 2** (1D line). The spiking model, the phenomenological rate model, the
STDP/plasticity machinery, and the recorder/GUI layers are **not** ported.

---

## 1. What was ported

| Python | MATLAB source |
|---|---|
| `las_model/kernels.py` | `StandardRecurrentConnection.m`, `conv2_field.m`, `@Projection/Kernelize` |
| `las_model/params.py` | `gen_param.m`, `MeanField1DStandardTemplate.m`, `MeanField2DStandardTemplate.m`, `Exp1Template.m`, `Exp2Template.m` |
| `las_model/model.py` | `@MeanFieldModel/MeanFieldModel.m` (constructor + `IndividualModelUpdate`), `@NeuralNetwork/Update`, `@Projection/Project.m`, `@ExternalInput` |
| `las_model/run.py` | `Exp1.m` (2D), `Exp2.m` (1D) |
| `las_model/analysis.py` | new — measurement code, no MATLAB counterpart |
| `tests/test_port.py` | new — equivalence tests |

788 lines total.

## 2. State variables and update order

Per unit: membrane potential `V`, intracellular chloride `Cl_in`, sAHP
conductance `g_K`, dynamic threshold `phi`, and the two synaptic filter states
`Input_E` / `Input_I`. One Euler step of `IndividualModelUpdate` computes, in
this order: firing rate from `V` and `phi`; chloride accumulation driven by the
**raw** inhibitory input; the chloride Nernst potential `E_Cl`; sAHP activation;
membrane update using leak, AMPA, GABA-A (conductance-scaled) and sAHP terms;
threshold relaxation; then the synaptic filters decay.

**Statement order is load-bearing.** Several quantities are read at their
pre-update values within a single step — the MATLAB is not a simultaneous
system of equations, and reordering for tidiness changes the trajectory. The
port preserves the original order line for line; the docstrings mark each place
where a value is deliberately stale.

Two further conventions preserved verbatim:

- The **inhibitory input is used raw** in the chloride equation but
  **conductance-scaled** in the membrane equation. This is not a typo in the
  original.
- The synaptic filters decay at the **end** of the step, so the rate computed at
  the top of a step uses the previous step's synaptic drive.

## 3. Connectivity and convolution

Recurrent excitation and the distance-dependent share of inhibition are
Gaussian kernels over the sheet; the rest of the inhibition is
distance-independent (a global mean term). The paper writes the split as a
fraction `gamma`; `StandardRecurrentConnection.m` and this port instead carry
two separate conductances, `g_I_local` (250) and `g_I_global` (50), built as a
`ConvProjection` and a `GlobalProjection` respectively. Three details matter:

1. **`sigma` is a fraction of field extent**, not a pixel count. Converting
   requires the number of units along the dimension — see
   `sigma_pixels_from_fraction`.
2. **Truncate, then renormalise.** The MATLAB builds the kernel on a finite
   support and normalises *after* truncation, so the discrete kernel sums to 1
   even though the underlying Gaussian is clipped. Normalising first gives
   slightly weaker coupling and a measurably slower wavefront.
3. **Zero-padded `same`-mode alignment.** `conv2_field.m`'s linear-topology
   branch pads with zeros rather than wrapping, so units near the boundary
   receive less recurrent drive. This is a physical edge effect of the model,
   not an artefact to correct.

The isotropic Gaussians are exactly separable, so the port convolves with two
1D passes instead of one 2D pass. `tests/test_port.py` asserts this equals both
a direct `convolve2d` reference and a literal transcription of the MATLAB
routine, to machine precision.

## 4. Validation

No MATLAB licence is available here, so validation is against the **published
quantitative values**, not a bit-level diff with the original code. All speeds
are in normalised space units per second (network extent = 1.0 along a
dimension).

| Quantity | Paper | 1D port | 2D port |
|---|---|---|---|
| Ictal wavefront / territory expansion | 0.008 | 0.00826 | 0.00767 |
| Fast inward travelling wave | 1.36 | 1.44 | 1.364 |
| Ratio | 170 | 174 | 178 |

Qualitative features also reproduce: tonic-to-clonic transition (1D, ~18 s),
slowing of the ictal rhythm toward termination, self-termination (1D ~76 s, 2D
~78 s) with the seizure territory covering only part of the sheet (4412 of 7860
units in 2D), and the annular wavefront with waves born at its inner edge
travelling inward.

### Measuring the fast wave speed — a trap

The obvious method (regress firing-rate peak time on position across the
recruited territory) **fails**: once the territory is wider than one
wavelength, the fit aliases across several waves, and fitted speeds sweep
through implausibly large magnitudes and flip sign. `analysis.py` instead
follows individual ridges — start at the outermost recruited position and track
the nearest peak time inward, rejecting jumps larger than a fraction of the
inter-discharge interval. That yields a tight distribution (1D IQR
1.35–1.55 units/s over 23 ridges; 47 ridges in 2D).

Similarly, tonic-to-clonic detection must use the **maximum** rate trace, not
the population mean: the mean is diluted by the growing silent territory and
gives a transition time inconsistent with the raster.

## 5. Reproducing

```python
from las_model.run import run_exp1, run_exp2
res1d = run_exp2(T_end=100_000.0, write_cycle=1.0)   # ~56 s wall
res2d = run_exp1(T_end=100_000.0)                    # ~157 s wall
```

Both are deterministic at fixed seed. Timings are for a 6-core sandbox; the 2D
run peaks around 6.5 GB resident, which is the constraint to watch when this
moves to Modal for the parameter sweeps.

## 6. Implications for Liou-MNIST

- **Cost.** 100 s of 2D simulation takes ~157 s wall and ~6.5 GB. A parameter
  search over 10 s runs is roughly 16 s each, so sweeps are feasible, but memory
  per worker should be sized against the recorded field, not the state.
- **Recording.** The full `(T, n, n)` rate field dominates memory. For sweeps,
  record summary statistics online rather than the whole field.
- **Which parameters move.** The natural seizure-susceptibility axes here are
  the recurrent excitation conductance `g_E`, the local/global inhibitory
  conductances `g_I_local` and `g_I_global` (the model's stand-in for the
  paper's `gamma` split), the kernel widths `sigma_E_frac` / `sigma_I_frac`,
  the chloride time constant `tau_Cl`, and the sAHP strength `g_K_max` — the
  candidates to track during training, analogous to the Wilson-Cowan parameter
  trajectories.
- **Seeds and determinism.** Matching the Wilson-Cowan project's convention, the
  runs are deterministic at fixed seed; the OU noise path in `ExternalInput` is
  present but unused in the Fig-1C/Fig-2 configurations.


---

## Extension: spiking model (Figure 4A/4B)

The Figure-1C port covered only the mean-field rate model.  Figure 4 is a
**spiking**-model result, so this extension adds a second model family.

New modules
: `spiking.py` -- `SpikingModel`, port of `@SpikingModel/SpikingModel.m` plus
  `SpikingModel1DStandardTemplate.m` and `Exp4Template.m`.
: `lfp.py` -- port of `@Recorder/LFP.m` (Mazzoni LFP proxy) and the conductance
  recomputation in `@Recorder/RecordExtraVar.m`.
: `run_spiking.py` -- port of `Exp4.m`.
: `wave_speed.py` -- **not a port**; see below.

### What distinguishes panel A from panel B

Not the template.  `Exp4Template.m` is identical for both panels (beta 2.5 ->
1.5 mV, `E_L` and initial `V` -> -60 mV).  The difference is in `Exp4.m`, which
for panel B overrides both inhibitory projection strengths:

| | local inhibition | global inhibition |
|---|---|---|
| panel A | 250 | 50 |
| panel B | 150 | 150 |

This is the paper's gamma = 1/6 vs gamma = 1/2 -- the *fraction* of inhibition
that is distance-independent, expressed in the code as two conductances.  It is
the same reparameterisation already noted for the mean-field model.

### MATLAB conventions preserved in `spiking.py`

Documented at length in the module docstring.  The load-bearing ones:

- `phi` and `g_K` are driven by the spike vector from the **previous** step
  (`O.S.S` is only overwritten after those two equations run), so their Dirac
  terms lag `V` by one step.
- `E_Cl` is computed once from the pre-update `Cl_in` and reused in both the
  membrane and chloride equations.
- Inhibitory input enters the chloride equation **raw** but conductance-scaled
  (divided by `f_max`) in the membrane equation -- the same asymmetry the
  mean-field model has.
- Refractory suppression is applied **after** the random draw, so refractory
  units still consume a random number.  Required for seed reproducibility.
- `V_reset` is applied to the membrane potential despite the template comment
  calling the argument a threshold.
- Synaptic filters decay at the end of the step, after projection output.

### Two issues found in the released code

**1. The LFP smoothing kernel is ambiguous.**  `LFP.m` builds it as
`exp(0:-1:-N / N / sigma)`.  MATLAB's `:` binds looser than `/`, so the
endpoint is `-(N/N)/sigma = -1/sigma`: the network size **cancels**, the step is
-1 per neuron, and the kernel decays by a factor e per neuron -- an e-folding
length of one neuron, truncated at `1/sigma` = 40 neurons.  In practice 80% of
the weight sits within +/-1 neuron, i.e. almost no spatial averaging.  But the
function's own docstring calls sigma the "contribution of neighboring cells",
implying the intended e-folding length was `sigma*N` = 50 neurons.  Both
readings truncate at the same point but differ by a factor of 50 in smoothing
width.  `lfp_kernel(..., literal=True|False)` selects; figures here use the
literal reading.  **This should be checked against the author's intent before
the LFP is used quantitatively.**

**2. The PSC recorder pairs `g` and `V` one step apart.**

*(Earlier revisions of these notes called this a "double count". That was
wrong -- corrected below after tracing the update cycle. Retained here because
the wrong claim was propagated into the LFP discussion.)*

`RecordExtraVar.m` recomputes conductance as `Input.<T> + sum(Proj.In.Value)/tau`.
The reason is that `Input` alone is *stale* at recording time.  The cycle in
`@NeuralNetwork/Update.m` is:

1. `IndividualModelUpdate` -- folds the pending `Proj.In.Value` into `Input`,
   runs equations 1-4, generates spikes, then decays `Input` by `exp(-dt/tau)`.
2. `Project` -- **overwrites** `Proj.In.Value` with this step's output.
3. The recorder runs.

So by the time the recorder sees the model, `Input` has already been decayed and
the newly written `Proj.In.Value` has not yet been folded in.  Adding it back is
the author's compensation, not an extra count.

**It is not a double count.**  `Project.m` assigns `P(i).Value = ...` rather
than accumulating, and the fold-in at step 1 is the only path into `Input`, so
each arrival is counted exactly once.  Verified numerically: the recorded
conductance equals the conductance the equations use on the *following* step,
to a max absolute difference of **0.0**.

The real defect is subtler -- a **one-step pairing mismatch**.  The recorded `g`
corresponds to step *k+1* while the `V` it is multiplied by is from step *k*:

| comparison | correlation |
|---|---|
| `corr(recorded, consistent)` | 0.9387 |
| `corr(recorded[k], consistent[k+1])` | 0.99999 |
| mean ratio (EPSC / IPSC) | 1.0005 / 0.9993 |

So the PSC traces carry a 1 ms phase error, not an amplitude error, and the
means are unbiased to 0.05%.  For the Figure 4 LFP -- whose features are on the
scale of 50-100 ms discharges -- this is immaterial.  It would matter for
sub-millisecond timing claims or for a driving-force decomposition.

`SpikingModel.psc()` reproduces the author's convention deliberately, since the
published figure was made with it.

### Wave-speed analysis is reconstructed, not ported

The repository ships **no** wave-speed routine.  `ChasePeak.m` walks a 2-D local
maximum; `MeasureSeizureActivity.m` finds phase transitions.  Neither produces
the per-discharge speed-and-direction estimates in Figure 4B's lower subpanel.
`wave_speed.py` implements the method the legend describes: per discharge,
regress local spike time on position, take 1/slope as speed and its sign as
direction, and use the F-test p-value with the legend's 0.001 threshold.
Window sizes are not specified in the paper and are explicit arguments.

**The aliasing trap recurs here.**  In the clonic stage a unit fires at 70-90
Hz, so a regression window wider than half the inter-discharge interval mixes
successive waves and inverts the outward/inward speed ordering.  Using a naive
+/-40 ms window with per-unit spike-time centroids, the measured medians came
out as outward 15.9 and inward 2.8 -- backwards.  `speed_at_discharge` takes
each unit's spike **closest to the discharge centre**, and `measure_wave_speeds`
sets the half-window to 40% of the median inter-discharge interval.

### Figure 4 results

| | this port | paper |
|---|---|---|
| panel A onset | tonic, uniform recruitment, large LFP DC shift | LVFA |
| panel B onset | discrete outward waves, rhythmic LFP | rhythmic |
| outward waves (panel B) | n = 14, median 4.80 | n = 34 |
| inward waves (panel B) | n = 19, median 19.79 | n = 23 |
| outward vs inward | U-test p = 2.6e-05 | U-test p < 0.001 |

Speeds are normalised space units per second; the paper reports the panel-B
speed comparison only as a significance statement, not as absolute values, so
the numbers above are checkable only in ordering and significance.

Cost: 50 s of 1D spiking simulation (2000 units, dt = 1 ms) takes ~46 s wall per
panel and stays under 1 GB with `write_cycle = 1 ms`.

### LFP kernel: the paper resolves the MATLAB ambiguity

Earlier notes flagged `@Recorder/LFP.m`'s kernel construction as ambiguous and
implemented both readings behind `lfp_kernel(literal=...)`. **The paper settles
it.** The Methods subsection on the LFP-like readout states that the synaptic
contribution of neighbouring cells decays exponentially with spatial constant
`sigma_LFP = 0.025` in *normalised spatial units* — the same units Table 2 uses
for `sigma_E = 0.02` and `sigma_I = 0.03`.

For N = 2000 that is an e-folding length of `0.025 * 2000 = 50` neurons.
Measured: `literal=False` gives 51, `literal=True` gives 2. So the literal
MATLAB precedence reading is a **bug in the released code**, not a modelling
choice, and the intended behaviour *is* documented in the paper.

`literal=False` is now the default for both `lfp_kernel` and `compute_lfp`, and
Figure 4 is rendered with it. The two readings are not merely a scale factor:
correlation between the traces they produce is 0.06-0.30 across the four
readout channels, and the resulting figure differs qualitatively (under the
paper kernel, panel A's LFP is dominated by the distal channel rising as the
seizure territory reaches it, rather than by the near channel's early activity).

Consequence for any quantitative LFP claim: results computed before this change
used a kernel with essentially no spatial averaging and should be recomputed.

Full Methods specification, for reference — all three parts are implemented:

| Paper | Implementation |
|---|---|
| `I_E − α·I_I`, α = 1.65 | `alpha=1.65` in `compute_lfp` / `lfp_at` |
| `I_E` delayed by 6 ms | `dT=6` samples at dt = 1 ms |
| neighbour decay, `σ_LFP = 0.025` normalized | `lfp_kernel(literal=False)`, e-folding 51 of 50 neurons |

The sign convention is correct as written: `epsc = gE·(E_syn − V)` is positive
(depolarizing) and `ipsc = gI·(E_Cl − V)` is negative (hyperpolarizing) at rest,
so `I_E − α·I_I` evaluates to `|I_E| + α·|I_I|` — Mazzoni's proxy, which sums
the two synaptic contributions rather than cancelling them.

The input currents themselves are also sound: the `RecordExtraVar.m` convention
was previously described in these notes as double-counting the arriving
projection, but tracing the update cycle shows each arrival is counted exactly
once (see the corrected item 2 above).  The residual is a 1 ms phase offset
between the recorded `g` and `V`, which is immaterial at Figure 4's timescales.
**There is therefore no outstanding correctness blocker on the LFP.**

### Readout-site bug: reversed kernel slice

The single-site helper used for the figure applied the *reversed* kernel slice.
Reversal is a no-op when the window is truncated symmetrically — i.e. when the
readout sits at the array centre — so a centre-location spot check passes.  At
the actual readout sites it mirrors the readout to the far side of the sheet:
requesting unit 300 read unit 1699, and unit 700 read unit 1299.

Symptom: focal onset invisible in the LFP, with signal appearing only once the
seizure had spread across the whole sheet.  The channels were reporting distant
tissue, not the tissue under the electrode.

`lfp_at` now lives in `las_model/lfp.py` (rather than only in figure code) and
is asserted equal to `compute_lfp` to ~1e-13 at interior *and* boundary
locations by `tests/test_port.py::test_lfp_at_matches_compute_lfp`.  After the
fix, LFP1 (x = 0.15) crosses 10 SD at 2.02 s in both panels — stimulus onset —
and panel A's LFP2 (x = 0.35) at 8.60 s, consistent with the wavefront arriving
later at the more distant site.

General lesson for this port: any "cheap single-site" shortcut must be tested
against the array-wide computation at off-centre and boundary positions.  A
symmetric test case cannot see an asymmetry bug.

### Axis convention for the blanked interval

The paper removes a 20 s interval from the Figure 4 time axis for
visualisation.  Here that interval is 12-32 s of simulation time.  **Tick labels
show real simulation time, not the compressed drawing coordinate** — the axis
jumps 10 -> 35 across the break, which is marked with diagonal slashes on both
spines.  The first draft got this wrong: ticks were left at their compressed
positions, so the label "15" sat where real time was 35 s and every post-break
time in the figure was understated by 20 s.  Any future figure using
`blanked()` must set a `FixedLocator` at remapped positions with a
`FuncFormatter` that adds the omitted span back.


---

## Extension: Figure 6 (2D round field, noise vs no noise)

New modules: `las_model/wave2d.py`.  New entry points: `params.exp6_template`,
`run.run_exp6`.  Reuse was high — `MeanFieldModel`, `ExternalInput` and the
kernel/convolution code were already dimension-general.

### Problems found and what was done about them

**1. Noise units are not reconcilable from the sources.**  The paper's Figure 6
legend reports a *diffusion coefficient* `D = 200 pA^2/ms`.
`@ExternalInput/ExternalInput.m` takes an *amplitude* `Random.sigma` and applies
it with no timestep scaling; `Exp6.m` does not set it at all — it reads it from
the caller's workspace if present.  Nothing in the paper or the repository
states the conversion.  `run_exp6` uses `sigma = sqrt(2 D / dt)` (= 20 pA at
`dt = 1 ms`), verified against an ensemble of realisations at three timesteps.
**Consequence: the effective noise magnitude depends on `dt`.**  Passing `D`
directly as an amplitude would inject noise orders of magnitude too strong.
This is the one item that would benefit from the author's confirmation.

**2. OU time-filter used Euler decay where MATLAB uses exact decay.**  Fixed in
`model.py`.  Figure 6 uses white noise (the degenerate case) so this figure is
unaffected, but the spiking-model figures use the filtered path.

**3. Coarse recording destroys the confluence and direction estimates.**  At
`write_cycle = 10 ms` a discharge spans only a handful of frames, so activation
times are massively tied (168 tied maxima in one test) and `chase_peak` wanders
over a plateau.  The direction disc implied by the legend is only ~2.5 units
wide, which compounds it.  `run_exp6` therefore takes a `fine_window` recording
rate at every integration step over a bounded interval.  **Any 2D wave analysis
must use fine recording.**

**4. Discharge detection must be calibrated, not guessed.**  A 40 ms separation
floor over-detected by ~5x, which drove the planar-fit quality down to r2 = 0.54
and left most directions undefined.  The true period is 255 ms (3.8 Hz),
recovered from the autocorrelation and power spectrum of the population-mean
rate.  With `min_sep_ms = 150` the median r2 rose to 0.947.  **Symptom to watch
for: if the measured inter-discharge interval equals your separation setting,
you are at the floor and over-detecting.**

**5. The zero-gradient guard was on the wrong quantity.**  `wave_direction`
originally returned an infinite speed when the fitted gradient vanished.  The
real signature of the noise-free case is different: the planar fit explains
*exactly zero* variance while the gradient is floating-point dust (~1e-14) that
inverts to an absurd finite speed.  The guard is now on `r2` (`min_r2`), which
is the physically meaningful statement — radial convergence is not a travelling
wave and has no direction.  This cleanly separates the panels: 0 directions
defined without noise, 135 with.

**6. Termination is seed-dependent.**  At `D = 200`, seeds 2-5 terminate near
61 s but seeds 0 and 1 run past 150 s.  A scan over amplitude showed termination
is *non-monotonic* in sigma, which rules out the unit conversion as the cause and
identifies this as realisation-level stochasticity.  A terminating seed (2) was
used so the pre-termination stage exists.  The paper reports one realisation.

### Reproduced vs not

Reproduced: noise-free waves converge at a single cell with zero variance and no
defined direction; noise displaces the confluence point and broadens the
direction distribution; direction consistency is higher during the clonic stage
than pre-termination.

Not matched: the directional autocorrelations run below the paper's reported
0.82-0.97 while showing the same qualitative drop.

### Wavefront geometry is square, and that is faithful

The noise-free wavefront is visibly square, not circular.  The Gaussian kernel
itself is radially symmetric; the squareness comes from MATLAB's rectangular
truncation window, `ceil(2.5*sigma)` per dimension, which leaves 11.8% of the
excitatory kernel's mass outside the inscribed circle.  The port preserves this.

Cost: a 100 s 2D run at `dt = 1 ms` on a 50x50 grid takes ~37 s wall.  Fine
recording over a 70 s window is ~1.4 GB, so runs are processed one at a time.


---

## Extension: periodic boundary conditions (torus)

`conv_field` gained a `topology` argument. `'linear'` (default, unchanged) is the
zero-padded convolution behind every published figure; `'circular'` wraps.
`add_standard_recurrent_connections(topology=...)` threads it to both convolution
projections, and `run_exp1_torus` runs the Figure 1C configuration on a 50x50 torus.

**This is a port, not an invention.** `conv2_field.m` and `@Projection` both carry
a `'circular'` branch; no published figure uses it. The MATLAB implements the wrap
by folding the tails of the full convolution back onto the opposite edge; for a
kernel no larger than the field that is exactly `mode='wrap'`, verified against a
literal transcription of the MATLAB to 3.3e-16.

**Three things change together on a torus, and they should not be conflated:**

1. *Periodicity.* Recurrent drive wraps rather than truncating. Every unit then
   receives identical total drive (ratio min/max = 1.000) where the linear
   topology gives edge units 70% and corner units 49% of the centre value. This is
   exactly the deficit the MATLAB's unused `Adjust` flag was written to compensate.
2. *No mask.* `Exp1.m`'s `rr < 0.5` round mask exists to impose a boundary. A torus
   has none, so the mask is dropped: all 2500 units are active, versus 1976 for a
   50x50 round field and 7860 for the published 100x100 one.
3. *Grid size.* Sigma is a fraction of field extent, so halving the grid halves the
   kernel width in pixels (sigma_E 2.0 -> 1.0 px, 9 -> 5 taps). The physical sheet is
   the same; it is sampled half as finely.

**What the torus actually changed here: very little.** Against a matched control
(same 50x50 grid, no mask, same centred stimulus, linear topology) termination moves
62.6 s -> 64.7 s and recruited territory 53.8% -> 51.9%. The reason is visible in the
run: the seizure terminated before its front reached the wrap seam (furthest
min-image distance 21 of 25 units), so periodicity was never exercised along that
trajectory. An off-centre torus run *does* place activity on both sides of the index
seam, so the wrap is functional; it simply did not matter for this stimulus.

The larger gap against the published configuration (77.8 s) is therefore a grid-size
and stimulus-position effect, not a boundary-condition effect.

**Chaotic sensitivity, documented.** On a torus, translating the stimulus must
translate the solution. It does exactly for short runs (0.0 difference at 3 s) but
diverges to O(1) by 40 s in a 100 s run. The convolution is exactly equivariant under
a roll; the culprit is the global inhibition projection, which sums the whole field —
that sum differs by ~2e-13 between a pattern and its roll, and the difference
amplifies. Any claim about a single long trajectory on this model is a claim about
one realization, not about the dynamics. Ensembles, not single runs.

Regression tests: `test_circular_conv_matches_matlab_fold`,
`test_torus_has_uniform_drive_and_conserves_mass`, `test_torus_is_translation_equivariant`.


## Movies of network activity

`las_model/movie.py` renders any 2D result dict (`run_exp1`, `run_exp1_torus`,
`run_exp6`) or a bare `(frames, rows, cols)` array to MP4:

    from las_model.movie import activity_movie, state_movie
    activity_movie(np.load('exp1_torus_50.npz'), 'figs/torus_activity.mp4', stride=17, fps=30)
    state_movie(np.load('exp1_torus_50.npz'), 'figs/torus_state.mp4', stride=17, fps=30)

`activity_movie` gives the firing-rate field with a mean-activity trace and a
moving time cursor; `state_movie` puts rate beside the slow variables
(`Cl_in`, `g_K`, `phi`) on a shared clock, which is where seizure *termination*
is visible -- the rate field alone does not show why activity stops.

Three conventions matter and are enforced in code:

* **The colour scale is fixed across the movie and defaults to the true
  maximum, so nothing clips.** A per-frame autoscale would make a dying seizure
  look as intense as a fully recruited one. The returned dict reports
  `clipped_fraction`, which should be 0.0. (A percentile default was tried and
  rejected: p99.9 clipped the ictal core in 43 of 500 frames.)
* **Every frame carries a clock and the playback speed-up**, so a frame can be
  placed against the paper's time-course figures. At the 10 ms write cycle,
  `stride=17` at 30 fps gives 5.1x real time and a ~20 s movie from a 100 s run.
* **ffmpeg is not on PATH in this environment**; `ensure_writer()` registers the
  `imageio-ffmpeg` binary, falling back to animated GIF if that is unavailable.

**`phi` is the adaptive firing threshold in mV, not a synaptic resource.** It
enters as `activation(V - phi)` with baseline `phi0 = -45 mV`, relaxing toward
`phi0 + 60*(R_f/f_max)` with tau 100 ms -- rising `phi` is spike-frequency
adaptation. Its negative values in a movie are correct.

Regression test: `test_movie_renders_without_clipping`.


## Background noise: D = 200 pA^2/ms (Figures 6 and 7)

**How it enters.** `Exp6.m` sets `O.Ext(1).Random.sigma = noise_sigma` and
leaves `tau_x` and `tau_t` empty.  In `@ExternalInput/ExternalInput.m` that is
the pure-white branch: `z = randn(n)` -- a fresh independent draw for every unit
on every step, added to the deterministic stimulus as a *current* in pA.  No
spatial or temporal OU filtering is applied.  "Uniformly over the whole neural
sheet" means the same sigma everywhere, not a common value shared across units.

**Amplitude vs diffusion coefficient.** The paper reports D = 200 pA^2/ms; the
code takes an amplitude.  `sigma_from_diffusion` uses `sigma = sqrt(2D/dt)`
(20 pA at dt = 1 ms, matching the sigma_s = 20 pA quoted for the Figure 7
spiking model).  Passing 200 directly would inject 100x the intended noise.

**The dt-scaling is what makes the physics dt-invariant** (this CORRECTS an
earlier note in this project, which had it backwards).  Injected current is
low-pass filtered by the membrane -- `I_ext` enters `V_inf`, and `V` relaxes
toward it with `tau_V_eff = C/g_sum` -- so the stationary voltage jitter is

    sd(V) = sqrt(D / (g_L * C)) = sqrt(200 / (4 * 100)) = 0.707 mV

independent of dt.  Verified by simulating a passive membrane at
dt = 0.25/0.5/1/2 ms: simulated sd(V) = 0.7070, 0.7062, 0.7097, 0.7072 mV,
against the analytic 0.7071 at every step.  With sigma = 200 held fixed
instead, sd(V) grows from 3.5 to 14.1 mV over the same range.

**Effect on firing is strongly state-dependent.**  sd(V) = 0.71 mV is 28% of
the activation width beta = 2.5 mV, but `f = f_max/(1+exp(-(V-phi)/beta))` has
very different slope at different operating points:

| operating point | V - phi | rate | induced sd(rate) |
|---|---|---|---|
| rest (V = -58) | -13.0 mV | 0.55% f_max | 0.2% f_max |
| depolarised (V = -46) | -1.0 mV | 40% f_max | 6.8% f_max |
| at threshold | 0 mV | 50% f_max | 7.1% f_max |

A 23x difference in effect from identical injected noise.  This is why the same
D leaves the resting sheet quiescent but makes the ictal wavefront -- where
units sit near threshold -- genuinely stochastic, and it is the mechanism behind
the seed-dependence of Figure 6 termination already recorded here.

**Whiteness verified** in the port over 4000 steps on a 50x50 field: lag-1
spatial correlation -0.0004 (pooled over all adjacent pairs, N = 9.8e6), lag-1
temporal -0.0002 (pooled over all units, N = 1e7), per-unit sd 20.01 pA,
field-mean sd 0.405 pA = sigma/sqrt(2500).

Figure: `figs/noise_explainer.png`.


## The `I_ext` slot: stimulus and noise share one entry point

Source: `@MeanFieldModel/MeanFieldModel.m`, `IndividualModelUpdate`, lines
48-81.  `V_inf` is **Liou's own variable name**, not a paraphrase:

    g_sum = O.param.g_L + O.Input.E./O.param.f_max + ...
                          O.Input.I./O.param.f_max + O.g_K./O.param.f_max;
    E_Cl  = 26.7*log(O.Cl_in./O.param.Cl_ex);
    V_inf = (O.param.g_L.*O.param.E_L + ...
             O.Input.E./O.param.f_max.*O.param.E_Esyn + ...
             O.Input.I./O.param.f_max.*E_Cl + ...
             O.g_K./O.param.f_max.*O.param.E_K + I_ext) ./ g_sum;
    tau_V_eff = O.param.C./g_sum;
    O.V = V_inf + (O.V - V_inf).*(exp(-dt./tau_V_eff));

`las_model/model.py::MeanFieldModel.step` transcribes this line for line.
The same block appears in `@SpikingModel` and `@PhenomenologicalRateModel`.

`I_ext` is accumulated over ALL registered `ExternalInput` objects, and
`Evaluate` returns `deterministic(x1,x2,t) + sigma*z` from one function.  So
the Fig-6b background noise and the Fig-1C stimulus disc occupy the SAME slot;
an MNIST drive would be a third `deterministic` callable summed into the same
term.  There is no separate stimulus pathway.

### Consequence: the slot is shared, the gain is not

`I_ext` is divided by `g_sum`, which grows with the network's own activity.
Measured on the 50x50 torus configuration by logging the model's OWN
`Input_E`, `Input_I`, `g_K` each step (NOT reconstructed from rate):

| state                  | g_sum (nS) | dV_inf/dI_ext (mV/pA) | tau_V_eff (ms) |
|------------------------|-----------:|----------------------:|---------------:|
| rest                   |        4.0 |                 0.250 |          25.0  |
| most-active unit, peak |      154.4 |                 0.0065|           0.65 |

**Swing = 39x** (0.250 -> 0.00648), recovering within ~1 s of termination.
The field MEAN gain moves far less (0.25 -> 0.082 min) because most units stay
quiet.  Data: `exp1_torus_gsum.npz`.

CAUTION -- an earlier revision of this section reported 71x.  That was WRONG:
it reconstructed Input_E as `g_E * local_rate`, but the model's `Input_E` is
the *convolved* rate passed through a synaptic filter (`Input += value/tau_syn`
then `Input *= exp(-dt/tau_syn)`), which is both spatially smoothed and
temporally lagged relative to the local rate.  The reconstruction overstated
g_sum at the active core.  Always log `Input_E`/`Input_I` directly.

This is a divisive, shunting input nonlinearity with no counterpart in the
WC-MNIST model, where `I_ext` is added inside the sigmoid with unit gain
(`input_e = Py2Py - Inh2Py + I_ext + bias_e - alpha_H*H`).  Implications for
Liou-MNIST: (a) stimulus efficacy is history-dependent across sequentially
presented digits; (b) gradients w.r.t. input weights carry the same 1/g_sum
factor, so learning signal is suppressed for units that are firing; (c) a
WC-calibrated MNIST_INPUT_SCALE will not transfer -- calibrate against the
resting gain of 0.25 mV/pA and re-check at the operating point.


## MNIST input on the torus: measured configuration

Design: a 28x28 patch of the 50x50 sheet (31.4% of units), digit embedded
row-major so pixel (r,c) -> WC flat index 28r+c. Implemented in
`las_model/mnist_input.py`.

**Amplitude is referenced to threshold, not to Liou's trigger.** An isolated
unit reaches 0.5*f_max at `I = (phi0 - E_L)*g_L` = **52 pA**. Liou's Fig-1C
disc uses 200 pA = 3.8x threshold, deliberately far suprathreshold. `scale`
in `MNISTStimulus` is in multiples of 52 pA so it survives changes to E_L/g_L.

**Geometry checks out.** MNIST stroke width (2x mean EDT) is 2.35 px median
vs sigma_E = 1.0 unit: strokes are *resolved* by the excitatory kernel, not
smoothed into a blob. ~103 of 784 patch units carry ink per digit. The
centred origin leaves 11 units = 11 sigma_E to the wrap seam, so the digit
cannot interact with its own periodic image.

**The central finding: a two-order-of-magnitude timescale separation.**

| | |
|---|---|
| single MNIST presentation | 250 ms |
| digit held at 200 pA -> 10% territory | **25 s** |
| Liou Fig-1C disc -> 10% territory | 35.7 s |

A digit *can* seize the sheet -- it is not a subthreshold stimulus -- but it
takes ~100x longer than one presentation. Within a presentation the sheet
responds essentially linearly (peak 0.55 f_max at 200 pA, no propagation).

TRAP: my first sweep used a 600 ms stimulus in a 1 s window and found *no
seizure at any amplitude*, including 200 pA. That is a window artefact.
Liou's own Fig-1C stimulus runs 3 s and the seizure develops over tens of
seconds. Always check the run window against the ~30 s nucleation timescale
before concluding a stimulus is subthreshold.

**Cumulative risk is the real constraint.** With state carried across images
(as in WC-MNIST), 250 ms presentations of random digits drive mean g_K from
~0.003 to ~0.55 and seize the sheet within 200 images at every amplitude
tested:

| scale | first territory >5% |
|---|---|
| 1x threshold (52 pA) | image 89 (22 s) |
| 2x (100 pA) | image 32 (8 s) |
| 4x (200 pA) | image 21 (5 s) |
| **undriven control** | **never** |

The undriven control (identical protocol, zero amplitude) never seizes, so
this is digit-driven, not protocol- or noise-driven.

Mitigations tested:
- **Inter-stimulus blank gaps do NOT help** (0/250/750 ms all seize; 750 ms
  was marginally *worse*). So this is not incomplete recovery between images
  -- it is a parameter-regime problem.
- **Hyperpolarising E_L delays but does not prevent it**, and costs little
  separability: -58 -> -65 mV moves onset from image 32 to 81 while mean
  pairwise distance falls only 0.681 -> 0.604.

Separability at 250 ms rises monotonically with amplitude over 0.4-4x
threshold with no saturation, so amplitude is a soft knob -- there is no
sharp value it must take, which leaves it free to trade against seizure risk.

**Open question for the parameter search:** the seizure-free regime for a
long MNIST sequence is not reachable by amplitude or timing alone at the
Fig-1C parameters. Excitability (E_L, g_K recovery, E/I gains) has to be part
of the search grid -- which is convenient, since those are exactly the
parameters whose training-induced drift the project is about.


## Seizure detection in the MNIST-input analysis: what it was, and its correction

**What I used.** `territory = mean(R_f/f_max > 0.2)`, flagged as seizure when
territory > 5%, sampled ONCE per image (every 250 ms). Threshold 0.2 chosen by
eye; 5% chosen by eye. No duration criterion, no contiguity criterion, no
validation against the WC-MNIST classifier or against human labels.

For contrast, `labeling_toolkit/classifier.py` in the WC study requires, for a
Type-5 (ictal) call: >=7 CONTIGUOUS channels AND >=2 s SUSTAINED, with every
threshold traceable to a user ruling and the whole table validated against 100
adjudicated epochs. My detector met neither criterion.

**Two errors found on audit.**

1. *The sequence "seizures" do not survive a sustain test.* Re-logging every
   ms instead of once per image: at 2x threshold the longest CONTINUOUS run
   above 5% territory is **1.06 s** (above 10%: 0.06 s; above 20%: 0.00 s).
   That fails WC's 2 s bar. The activity is intermittent bursting sampled at
   250 ms intervals, not sustained ictal activity. Per-image sampling cannot
   distinguish "seizing at image 32" from "bursting when image 32 happened to
   be on screen".

2. *Instantaneous territory is the wrong metric for a travelling wave.* Liou's
   Fig-1C seizure is a propagating wavefront, thin at any instant. Measured on
   the same 60 s run: peak INSTANTANEOUS territory 15.8% but CUMULATIVE
   recruitment (fraction of units that were ever above threshold) 44.2%. My
   instantaneous metric systematically under-reads propagation while
   over-reading brief synchronous bursts -- errors in opposite directions.

**Corrected metric: cumulative recruitment**, `(ever > 0.2 f_max).mean()`,
with an undriven control. It separates cleanly:

| stimulus | cumulative recruited (60 s) | peak instantaneous |
|---|---|---|
| digit held, 4x threshold | 54.4% | 16.8% |
| Liou Fig-1C disc | 44.2% | 15.8% |
| **undriven control** | **0.0%** | **0.0%** |

The undriven control recruits exactly zero units, so the metric has no
false-positive floor here.

**What survives and what does not.**

- SURVIVES: held digits propagate, on a ~25-40 s timescale, ~100x longer than
  a 250 ms presentation. Confirmed under the corrected metric.
- SURVIVES (weakened): sequences of digits do drive cumulative recruitment
  OUTSIDE the driven patch -- 6.6 / 9.2 / 12.2% at 1x / 2x / 4x threshold
  after 120 images, undriven control 0.0%. So the direction of the earlier
  claim holds and remains digit-driven.
- RETRACTED: "seizes within 200 images at every amplitude", and the onset
  image numbers (89 / 32 / 21). Those were sustain-free per-image threshold
  crossings. Under a sustain criterion they are not seizures. First
  outside-patch recruitment >1% is later and better ordered: image 92 / 85 /
  78 at 1x / 2x / 4x.
- UNAFFECTED: the g_K drift (0.003 -> ~0.35), the gap and E_L results as
  *relative* comparisons, and all separability numbers (which never used the
  detector).

**Standing requirement:** before the parameter search produces any
susceptibility map, the detector must be fixed and validated -- ideally by
running the WC `labeling_toolkit` classifier on Liou LFP traces so both halves
of the project share one seizure definition. A susceptibility boundary is only
as meaningful as the detector that draws it, and the boundary's LOCATION in
parameter space depends on the criterion.


## LFP readout on the 2-D torus: faithful Liou/Mazzoni

**Decision (user, 2026-08-11): stay faithful to Liou/Mazzoni, do NOT adopt the
WC conventions.** Rationale: the models represent different components and make
different assumptions, and the project's question is whether two *different*
models drift toward seizure similarly. Forcing a shared readout would build the
similarity into the instrument.

### What was added

`MeanFieldModel.psc()` -- EPSC/IPSC recorder, identical in form to
`SpikingModel.psc()`. In the MATLAB the recorder lives on `@GeneralModel`, so
it is shared by every model class and the mean-field version must not diverge.
Conductance is recomputed as `Input_<T> + sum(_proj_value)/tau` because
`Input` is stale at recording time. **Verified**: recorded gE equals the
conductance entering the membrane equation on the following step to 2.8e-17.
Carries the same one-step pairing defect as the spiking version (g leads V by
one dt = 1 ms).

`lfp.lfp_kernel_2d` / `lfp.lfp_2d` -- the Mazzoni construction off the line.

### Three decisions that the 1-D original does not force

**1. The delay is in MILLISECONDS, not samples.** `LFP.m` hardcodes `dT = 6`
because Liou's recorder runs at dt = 1 ms, so ms and samples coincide and the
distinction is invisible in the MATLAB. Recording every 2 ms and passing 6
would impose a 12 ms delay. Converted once, tested both ways.

**2. sigma_LFP stays NORMALISED, with a consequence worth stating.** Liou
quotes sigma_LFP = 0.025 in the same normalised units as sigma_E = 0.02 and
sigma_I = 0.03, so the e-folding length is sigma*n = 1.25 units at n=50.
But his own LFP figures come from a 1-D N=2000 line where sigma*N = 50 units,
about 5% of the network within one e-folding. On a 50x50 sheet the same
normalised sigma puts only ~0.2% of the network within one e-folding, because
area grows as r^2. **Holding sigma normalised (faithful) therefore makes the
2-D electrode far more LOCAL, in network-fraction terms, than the 1-D one.**
That is geometry, not a porting choice. The alternative -- inflating sigma to
preserve the sampled fraction -- would break the shared convention with
sigma_E/sigma_I and is not done.

**3. Isotropic vs separable.** `exp(-r/L)` by Euclidean distance is the direct
reading of "decays exponentially with spatial constant". For a GAUSSIAN the
isotropic and separable forms coincide (which is why the connectivity kernels
can be built separably); for an EXPONENTIAL they do not. Default isotropic,
`isotropic=False` available.

### Electrode placement: 30 at stride 83

Checked against the WC aliasing trap, where stride 28 on a 28-wide image put
every electrode in column 0 (no ink). Here `gcd(83, 50) = 1`, so the
electrodes walk diagonally: **30 distinct rows and 30 distinct columns**, mean
toroidal separation 17.1 units (min 17.0, max 17.1 -- essentially uniform).
Test added so this cannot silently regress.

### First result: a complete focal seizure

Fig-1C stimulus (200 pA disc, r = 0.05, 2-5 s) on the 50x50 torus, 80 s,
PSCs recorded every 2 ms.

| quantity | value |
|---|---|
| nucleation | ~25 s |
| peak active fraction | 16.3% at 49.9 s |
| terminated by | ~60 s |
| cumulative recruited | 44.8% |
| LFP s.d. range across channels | 161 to 6939 (**43x**) |
| nearest / farthest electrode | 3.2 / 35.4 units from focus |

Amplitude falls off monotonically with toroidal distance from the focus and
the far channels sit at a flat noise floor (sd ~ 161) -- i.e. the readout
resolves focality rather than reporting a global average. Traces show
large-amplitude rhythmic discharges near the focus with clear onset,
build-up, and termination.

**Positive IPSC samples are not a bug.** `IPSC = gI*(E_Cl - V)/f_max` goes
positive in 2.3% of samples, rising to ~27% of units late in the run. E_Cl at
rest is -77.7 mV against V ~ -58 mV, so this is not depolarizing GABA at rest;
it is units driven BELOW E_Cl -- post-discharge hyperpolarization -- plus
chloride loading raising E_Cl. Where I_I > 0 the `-alpha*I_I` term subtracts
rather than adds. Faithful to the formula; flagged because a sign assumption
("I_I < 0 so the term always adds") would be wrong here.

### Test-runner defect found and fixed

`tests/test_port.py` had its `__main__` block in the MIDDLE of the file with
six explicit calls, so every test defined below it -- all the MNIST-input
tests and all the LFP tests -- was **silently skipped while the suite printed
success**. Replaced with auto-discovery: 21/21 now actually run.


## sigma_LFP raised to 0.11; optional WC-style filter; contrast run

**Decision (user, 2026-08-12).** Raise sigma_LFP from 0.025 to 0.11 so the 2-D
electrode integrates the same NETWORK FRACTION as Liou's 1-D one, and add the
WC-MNIST filter as an OPTION.

| sigma_LFP | e-folding L | units within L | % of 50x50 sheet |
|---|---|---|---|
| 0.025 (normalised, faithful) | 1.25 units | 5 | 0.20% |
| **0.11 (fraction-matched)** | **5.50 units** | **97** | **3.88%** |
| Liou 1-D N=2000 reference | 50 units | 101 | 5.0% |

So 0.11 lands at 3.9% against Liou's 5.0% -- the closest match available while
the kernel stays exponential and isotropic. **This trades convention for
comparability**: sigma_LFP no longer shares the normalised convention with
sigma_E = 0.02 / sigma_I = 0.03, so if the sheet is resampled the LFP kernel no
longer scales with the connectivity kernels and sigma_LFP must be recomputed
by hand. Recorded here because nothing in the code will warn about it.

### Filter is opt-in, unfiltered is the default

`lfp.highpass_eeg` -- zero-phase (forward-backward `filtfilt`) Butterworth
high-pass, 1 Hz, order 4, matching the WC `filter_eeg` convention. Exposed on
`lfp_2d` as `filter_hz=None` (default **unfiltered**), so the raw broadband
Lambda is always available. Two reasons the default is unfiltered: Liou applies
no such filter, and a 1 Hz high-pass ANNIHILATES DC -- in the WC study a
saturated network rendered as a FLAT trace, indistinguishable from quiescent.
A tonic Liou discharge could alias the same way, so classification should be
cross-checked against the unfiltered trace. Panels c/d of the figure are the
unfiltered controls. Zero-phase is enforced by a test (peak lag <= 1 sample).

### The contrast run: finding the intermediate regime

Focal drive was swept to find a non-propagating comparison. The transition is
sharp and sits between 50 and 80 pA, not where a naive guess would put it:

| Ic (pA) | peak active | cumulative recruited |
|---|---|---|
| 40 | 0.00% | 0.00% (silent) |
| 50 | 0.00% | 0.00% (silent) |
| 60 | 0.52% | 0.52% |
| **70** | **0.84%** | **0.84%** (local discharge, no propagation) |
| 80 | 14.72% | 32.04% (seizes) |
| 200 | 16.32% | 44.84% (seizes) |

**70 pA is the contrast run**: units fire locally for ~154 frames but the
wavefront never leaves the stimulated patch. 80 pA already seizes, so the
propagation threshold is between 70 and 80 pA -- a >10x smaller current than
Liou's 200 pA is enough, and 40 pA (my first guess at "subthreshold") was
uselessly silent.

Result at sigma_LFP = 0.11, filtered: seizure per-channel sd 4408..50402
(11x across channels), contrast 345..1860 (5x). Peak channel is **27x** larger
in the seizure than the contrast run, and panel e shows both falling with
distance from focus -- the local run is focal too, just 27x smaller and
non-propagating.

### Positive-IPSC fraction is drive-dependent (and bigger than first measured)

The runtime warning fired on both runs and gave a number I had not measured:
**8.97%** of samples positive on the full 80 s seizure (peak 41.8% of units at
one instant), against 0.03% on the contrast run. The earlier 2.3% figure came
from a 40 s truncated run that cut off before the seizure matured. So the
wrong-sign fraction scales with seizure severity -- another reason not to bake
in a sign assumption.


## The 70 pA discharge morphology (why it is not a small seizure)

Plotted on the seizure's stacking offset the 70 pA run looks flat. On its own
scale (**59x finer**, offset 1127 vs 66445) it has clear structure:

| property | 70 pA (local) | 200 pA (seizure) |
|---|---|---|
| discharges | **6, then silence** | sustained ~35 s |
| rhythm | **2.6 Hz** (IPI 379 ms, range 350-450) | 3.4 Hz (IPI 296 ms) |
| single-discharge FWHM | 50 ms | 50 ms (same shape) |
| active window | 5.31 - 7.26 s (1.95 s) | ~25 - 60 s |
| units that ever fire | **21 = exactly the stimulated disc** | 1121 (44.8%) |
| peak rate | 0.67 f_max | 0.65 f_max |
| LFP sd across channels | 345 - 1860 (5.4x) | 4408 - 50402 (11x) |
| positive-IPSC fraction | 0.03% | 8.97% |

**Interpretation.** The individual discharge is NOT a weaker event -- peak rate
(0.67 vs 0.65 f_max) and waveform FWHM (50 ms) are essentially identical to the
seizure's. What differs is *recruitment*: the 21 units that fire are exactly the
21 that were stimulated, so nothing propagates, and the burst train terminates
after 6 cycles. So 70 pA vs 200 pA separates **local ictal-like discharge** from
**propagating seizure**, which is a more useful contrast than an amplitude
difference would be.

**AMPLITUDE-RATIO CORRECTION.** The "27x" quoted earlier compared the seizure's
*largest* channel against the contrast run's largest channel -- different
electrodes at different distances from focus. On the **same channel** (ch16,
unit 1328, d=3.2) the ratio is **4x**, not 27x. Always compare matched channels.

**Firing starts at 5.31 s, after the 2-5 s pulse ends** -- even the
non-propagating discharge is the network's own activity, not the stimulus
waveform.

**Not a filter edge artifact.** Filtered LFP energy peaks at t=0.04 s, which
looks like a filtfilt edge transient, but the *unfiltered* trace has it too
(max 14619 raw in the first 200 ms): it is the model's initial settling
transient. It is ~4x smaller than the discharge (37339 on ch16) and the sheet is
silent (sd ~0) after 10 s. Checked rather than assumed.


## WC-analogous MNIST drive (`las_model/mnist_drive.py`, `las_model/run_mnist.py`)

Drives a 28x28 patch of the 50x50 sheet the way the WC study drives its network.

**Kept from WC deliberately** (so the two studies stay comparable on stimulus delivery):
- one global gain `MNIST_INPUT_SCALE`, applied to the image;
- a per-pixel learnable weight vector `mnist_weights`, i.i.d. uniform over a
  signed range, so ~45% of pixels *inhibit* their unit;
- elementwise identity pixel->unit map, no receptive field, no input
  connectivity matrix (a pixel spreads only via recurrent connectivity, one
  timestep later);
- the continuous-time row schedule for sequential presentation, computed from
  `(step*dt) % image_duration` rather than a quantised step count.
  `test_mnist_drive_schedule_matches_wc` asserts bit-identity with
  `wc_model.active_row_at` at dt = 0.5/1/2 ms, that all 28 rows are visited,
  and that the global and image-local forms agree over a full 10 s run.

**Changed, and why.** WC's drive is dimensionless; Liou's input is a current in
pA. Transferring WC's bounds literally would put a saturated pixel an order of
magnitude above the amplitude that triggers a seizure in the paper. So the
weight bounds are normalised to max|w| = 1 with WC's min/max *ratio* preserved
exactly (hence the same negative fraction -- `test_mnist_weights_preserve_wc_sign_ratio`),
and `MNIST_INPUT_SCALE` is expressed in multiples of the threshold current
(52.0 pA for `exp1_template`), so scale=1 puts a saturated max-weight pixel
exactly at threshold (`test_drive_is_in_threshold_units`). WC's numbers are
recoverable by rescaling. **These bounds are a starting convention, not a
claim about the right value.**

### Chloride can go non-positive: a failure mode WC does not have

Eq. 1 uses `E_Cl = 26.7*log(Cl_in/Cl_ex)`, which is NaN for `Cl_in <= 0`, and
the `Cl_in` update in `model.py` has **no floor**. Liou never reaches this
because every stimulus in the paper is a *positive* current injection (Fig. 1C
is +200 pA). Our signed weights hyperpolarise ~45% of driven units, which
depletes `Cl_in` without bound. Measured: the lowest-`Cl_in` units are exactly
the ones receiving the most negative drive.

What binds is **contiguous saturated negative drive, not ink coverage** -- a
solid 12x12 block goes non-finite by scale 60 while real digits (same 0.18 ink
fraction) survive to ~70, and all-ones (1.0 ink fraction) also survives 60.
So there is **no image-independent safe scale ceiling**, and none is claimed.

Handled with a counted, disableable clamp (`MeanFieldModel.cl_floor`, default
0.1 mM, `E_Cl = -187 mV` there -- far below physiological but finite):
- `cl_floor=None` restores the unguarded MATLAB behaviour (`test_cl_floor_can_be_disabled`);
- every activation is counted in `model.cl_floor_hits` and surfaced as
  `run_mnist(...)["cl_floor_hits"]`. **A parameter search must reject runs with
  a large count as outside the model's valid range, not treat them as data.**
- Verified non-perturbing: over the full scale sweep to 60 in both configs the
  clamp fires **zero** times and results are bit-identical to the pre-clamp run.
  It only engages where the run previously produced NaN.

### Calibration results (`figs/liou_mnist_calibration.png`, `liou_mnist_scale_sweep.npz`)

10 s runs, 40 random digits at 250 ms each, no training, no reset between images.

| scale | simultaneous mean rate | sequential mean rate |
|---|---|---|
| 1 | 0.0055 | 0.0049 |
| 10 | 0.0155 | 0.0055 |
| 39 | 0.0249 | 0.0065 |
| 60 | 0.0284 | 0.0070 |
| 100 | NaN pre-clamp (1489 hits) | 0.0074 |

- **Simultaneous drives the sheet ~4x harder than sequential** at matched scale,
  because sequential delivers 1/28 of the pixels at any instant with no
  compensating boost (WC behaviour, `test_sequential_masks_one_row_without_rescaling`).
  The two configurations are therefore **not** at comparable drive for a shared
  scale, which matters for any sweep that means to compare them.
- **Activity never leaves the digit patch** at any scale tried: outside-patch
  max rate 0.093 at scale 60 simultaneous, and the fraction of outside samples
  above 0.05 peaks at 3.3e-4. No propagating recruitment -- so at these
  settings the sweep is probing local drive, not seizure spread.

### The user's timing concern is real and quantified

`tau_Cl = tau_K = 5000 ms`, so a 250 ms image is 1/20 of one slow time
constant and a 10 s run spans two.

- **Within-image response is present, not absent**: image-locked mean patch
  rate swings 37% (simultaneous) / 64% (sequential) of its mean. Simultaneous
  peaks 5 ms after onset; **sequential peaks at 195 ms**, i.e. late in the
  image, close to the readout -- worth noting given that WC's readout window
  sits at the bottom rows of the digit.
- **10 s does not equilibrate the slow variables.** On a 40 s run, `g_K`
  reaches 95% of its final level only at ~35.6 s and `Cl_in` at ~32.6 s. At
  t = 10 s, `g_K` is at **42%** of its 40 s level.
- **But the firing rate plateaus by ~15 s** (5 s block means 0.052, 0.101,
  0.122, 0.124, 0.122, 0.120, 0.122, 0.117) -- it *declines* slightly as `g_K`
  accumulates. So a 10 s run measures a **transient**, sampled while adaptation
  is still building.

**Open decision, not settled here:** 10 s runs are what the WC study uses and
what the brief specifies, so they are what the sweep should use for
comparability -- but every statistic from them is an adaptation-transient
statistic, and two runs are comparable to each other only if they share the
window. A 15-20 s window, or a discarded lead-in, would measure the plateau
instead. This trades comparability with WC against stationarity.


## Chloride clamp: what a large hit count does and does not mean (2026-08-13)

An earlier version of these notes said a parameter search "must reject runs with
large clamp counts as outside the model's valid range, not score them as data."
That sentence conflated two different failure modes. Measured, they separate:

**The clamp is load-bearing.** With `cl_floor=None` (unguarded MATLAB
behaviour), scale 80/100/150 all produce non-finite rate. The clamp is what
makes those runs return numbers at all.

**But the numbers it returns are numerically insensitive to the floor.** Varying
`cl_floor` over 0.5 -> 0.001 mM (500x) changes the per-unit mean rate by at most
0.07% of peak, even at scale 150 where 87% of timesteps clamp. Restricted to the
clamped units themselves the difference is 0.6%. So a large hit count does NOT
mean the run's activity statistics are arbitrary or floor-dependent.

**The reason to reject is physiological, not numerical.** Clamping means
`Cl_in` was driven to <= 0 mM, i.e. the cell ran out of intracellular chloride.
At the floor, `E_Cl` = -187 mV (0.1 mM) or -144 mV (0.5 mM), against a resting
-78 mV. Those units are outside the range Eq. 1 was written to describe,
whatever the floor is set to. A seizure-susceptibility search that scores such a
run is scoring a regime the model does not claim to represent.

**Extent, not events.** A raw hit count conflates one unit clamped persistently
with many units clamped once. `run_mnist` now also returns
`cl_floor_frac_units` (fraction of the sheet ever clamped) and
`cl_floor_frac_steps` (fraction of timesteps with any clamping). At scale 150,
only 40 of 2500 units (1.6%) ever clamp while 87% of steps have a clamp event —
a persistent, spatially tiny departure, not a sheet-wide one.

**Recommended rule.** Reject on `cl_floor_frac_units > 0` for anything that will
be reported as a physiological result; the usable range in the calibration
sweep (scale <= 60, both configs) has exactly zero hits, so this costs nothing
there. Runs with hits are still numerically reproducible and can be kept as an
explicitly-labelled out-of-range arm if the search needs to know where the wall
is. What is NOT defensible is mixing them into a susceptibility score
undifferentiated.

Caveat: seed 0, one weight draw. The scale at which clamping begins depends on
the realized negative-weight pattern (see the image-dependence note above).


## 3x3 overlapping-patch projection on a 60x60 torus (2026-08-13)

`las_model/patch_projection.py` + `mnist_drive.PatchDrive`. Pixel (r,c) -> the
3x3 block at network rows 2r..2r+2, cols 2c..2c+2 (mod n), i.e. stride 2, so
consecutive patches share one row/column. Verified against the spec: pixel 1 ->
rows/cols 1-3, pixel 2 -> cols 3-5, pixel 29 -> rows 3-5 (1-indexed). Each of a
pixel's 9 connections has its own weight: (28,28,3,3) = 7056 independent values,
all distinct, normalised max|w| = 1 exactly as the identity drive is, so the WC
sign ratio and the threshold-current scale convention carry over.

**Rejection mechanism (requested).** `run_mnist.physiological_mask(runs)` ->
boolean array, and `assert_physiological(run)` raises. Both reject on
`cl_floor_frac_units > 0` (any unit ever clamped). Use the mask to filter a
sweep; use the assert in analysis paths that must not silently include
out-of-range runs.

### Five consequences of the new geometry, measured

1. **The chloride wall moves from >60 to 26.2** (bisected, simultaneous,
   seed 0). The projection amplifies drive: mean |drive| per driven unit 1.4x,
   peak 3.2x, most-negative 3.1x, because a unit at a 4-fold-covered site sums
   four independent signed weights instead of receiving one. That 3x on the
   negative tail is what moves the wall. **The usable search range shrinks 2.3x.**

2. **Drive is a 4:2:1 checkerboard, independent of the image.** Stride 2 with
   width 3 means even in-patch coordinates are covered twice and odd ones once;
   in 2-D that gives units covered by 4, 2, or 1 pixels (729 / 1620 / 900 on a
   60x60 sheet). Mean |drive| by class: 0.142 / 0.094 / 0.051. **The recurrent
   kernel does not erase it** -- sigma_E = 1.2 units at n=60, comparable to the
   period-2 pattern, so the firing rate still shows a 1.74x ratio between
   4-fold and 1-fold sites. This is a fixed spatial texture on every run, at the
   same scale as the connectivity kernel.

3. **57 != 60: the patches do not tile the torus.** 28 patches at stride 2 span
   2*27+3 = 57, leaving a 3-unit-wide undriven stripe in each direction (a cross
   on the torus), 351 units = 9.8%. **Exact tiling requires n = 56** (= 2*28),
   which also removes multiplicity class 0. If the point of the torus is that
   every unit has an identical neighbourhood, a 3-wide seam of permanently
   undriven tissue partly defeats it.

4. **Almost no unstimulated surround is left.** 90% of the sheet is driven,
   versus 31% under the 28x28-on-50x50 identity map. Earlier calibration found
   activity never leaves the digit patch, so the surround was the natural place
   to look for propagation/recruitment -- the observable that connects to
   seizure spread. At 90% coverage there is barely any "outside" to propagate
   into. If the susceptibility measure depends on recruitment of unstimulated
   tissue, this geometry weakens it.

5. **Sequential mode masks the IMAGE row, not the network row.** A masked
   network row would clip the 3-unit band and destroy the overlap the geometry
   exists for. `PatchDrive._refresh` masks upstream, so active image row 14
   drives network rows 28-30 (asserted in tests).

### Open choices

* **n = 56 vs 60.** 56 tiles exactly, zero undriven units, cleaner torus. 60
  keeps a small unstimulated reservoir. Neither matches the old 50x50, so the
  identity-map calibration numbers do not transfer either way.
* **Whether to compensate the checkerboard.** Dividing each unit's summed drive
  by its multiplicity flattens it, at the cost of making the 9 weights no longer
  independent free parameters. Left uncompensated by default.
* **7056 vs 784 trainable input parameters.** A 9x larger input layer than the
  WC study's. Relevant to the eventual comparison of training trajectories --
  the input parameters will dominate the count.

Caveat: seed 0, one weight draw, 200 images for the drive statistics. The wall
at 26.2 will shift with the realized weight pattern.


## Configuration decisions (2026-08-13): n=56, 1/m compensation, 80 s runs

Three choices were made and each has a consequence worth recording.

**n=56 sheet.** 28 pixels x stride 2 = 56, so the 3x3 patches tile the sheet
exactly: zero undriven units, no periodic wrap-around of the patch grid, and
input multiplicity takes only the values 1, 2, 4 in the 4:2:1 checkerboard set
by patch parity. At n=60 the patch extent (57) fell short of the sheet, leaving
a thin undriven stripe; at n=56 there is no unstimulated surround at all. That
removes the surround as a place to look for recruitment — any propagation study
will need either a larger sheet or a spatially restricted digit.

**Compensation = divide summed drive by multiplicity m.** Implemented as a row
scaling of the projection matrix. This does NOT flatten the drive asymmetry, it
inverts it: because each pixel's 9 connections carry independent signed weights,
the summed drive at an m-fold site grows as sqrt(m), not m, so dividing by m
overcorrects by exactly sqrt(m). Measured relative input drive at m=1/2/4:

    none              1.00 / 1.65 / 2.23
    divide by m       1.00 / 0.83 / 0.56    <- inverted
    divide by sqrt(m) 1.00 / 1.17 / 1.12    <- flat

`compensate` is therefore a mode selector (`None` / `"multiplicity"` /
`"sqrt"`), not a boolean. `"multiplicity"` is the default per the 2026-08-13
decision; `"sqrt"` is available and is the one that actually flattens.

A second consequence: compensation makes a pixel's 9 weights no longer
independent free parameters. The effective weight a unit sees is w / m(unit),
and m depends on the unit's parity, so identical nominal weights act with
different strength at different sites. Training optimizes the nominal weight
and the model responds to the compensated one.

**The rate-level case for compensation is weak.** The earlier 1.74x
high-vs-low-multiplicity firing-rate ratio was measured at 3 s, inside a
transient. Tracked through an 80 s uncompensated run, the ratio decays to ~1.0
by 10 s and stays there. At 80 s the m4/m1 rate ratio is 1.027 (none), 0.993
(1/m), 1.005 (sqrt) — all three flat. The input-drive asymmetry is real; its
rate-level signature is not, once the network equilibrates. If a later analysis
depends on the drive asymmetry rather than the rate, revisit the mode choice.

**80 s runs.** Validated by g_K equilibration: flat to 0.7% over the final
quarter. 10 s reaches only ~50% of the final g_K. Provisional per the user.
Cost: ~90 s wall per run single-threaded, which sets the Modal budget for the
parameter search (a 2-config x N-scale x M-seed grid at 90 s/run).

**Chloride wall under this configuration.** Bisected to MNIST_INPUT_SCALE ~= 37
(between 36.6 and 37.0), up from ~26 at n=56 uncompensated: dividing by m
shrinks the negative tail along with everything else. Usable search range for
physiological runs therefore widens ~1.4x. Reject any run with a nonzero
`cl_floor_units` from physiological analysis (`physiological_mask` /
`assert_physiological` in patch_projection.py).

Single seed, single weight draw — see the standing scope caveat.


## Compensation default changed to sqrt(m) (2026-08-13)

Per your instruction the default is now `compensate="sqrt"` in
`run_mnist`, `PatchDrive` and `build_projection`. Verifying it produced a
correction to what I told you last turn, and found a bug in my own analysis.

**A transposed array invalidated my earlier drive numbers.** The variable I used
for the multiplicity-class measurements was `(784, 200)` — pixels x images —
and I reshaped it as `(200, 784)`, which scrambles pixels across images. Every
drive statistic computed that way is meaningless. The wall sweeps are unaffected
(they pass the image tensor `X` straight to the runner), and so is the earlier
figure's panel b, which used a separately-written loader that happened to be
correct. `tests/test_port.py` now asserts the loader returns `(n, 784)`.

**Corrected scaling.** Drive at an m-fold site grows as m**p with

    p = 0.563  (0.491-0.626 over 5 weight draws, 200 images, n=56)

The spread straddles 0.5, so sqrt(m) is within measurement error of the
flattening exponent. Residual m4/m1 asymmetry:

    none 2.24 | sqrt 1.12 | 1/m 0.56

sqrt(m) is the best of the three and 1/m is the worst — it inverts by almost
exactly the factor sqrt undercorrects by. Your choice is the right one.

**Why 0.5 works despite sparse input.** The random-sign-cancellation argument
assumes all m inputs contribute on every presentation, which sparse input
should violate: MNIST is ~19% nonzero, and synthetic inputs at matched density
give p = 0.89. MNIST does not, because strokes are spatially CORRELATED — the
2-4 pixels feeding one unit are neighbours and are on or off together, so the
unit sees the dense case (p = 0.54). Correlation, not density, is what makes
sqrt right here. This is a property of MNIST; a different input statistic
(random dot patterns, noise) would need a different exponent.

**Chloride wall moves up.** Bisected under the locked configuration
(n=56, 80 s, seed 0, patch projection):

    none          MNIST_INPUT_SCALE 24.4-24.8
    1/m           36.6-36.8   (reproduces the value reported last turn)
    sqrt          39.5-39.8

sqrt(m) therefore gives the widest usable physiological range, ~1.6x the
uncompensated one. Note the ordering is NOT predicted by peak drive magnitude:
sqrt has a 1.35x more negative field than 1/m yet clamps at a HIGHER scale, and
at matched scale it clamps MORE units (5 vs 1 at scale 46). The onset is set by
where the negative drive sits relative to its inhibitory surround, not by the
tail alone. I predicted the wall would fall and it rose.

**`compensate` now accepts an exponent.** `None`, `"sqrt"` (0.5, default),
`"multiplicity"` (1.0), `"measured"` (`MEASURED_EXPONENT` = 0.56), or any
float. `"measured"` flattens to 1.03 but is fitted to one image set and one
sheet size; sqrt is preferred as the principled choice. Unknown strings now
raise `ValueError` instead of silently falling through to sqrt.

Single seed, single weight draw for the wall values — standing scope caveat.


## Seizure transition: the MNIST drive cannot nucleate one (2026-08-13)

You asked for LFP plots showing the transition into paroxysmal activity.  The
short answer is that no setting of the MNIST drive produces one, and the reason
is structural.  A focal probe pulse of Liou's own Exp1 geometry does, so the
runner now takes one.

**What was screened, and what each result was.**  Holding the digit stream as
the drive at n=56 and n=84, 10-80 s runs:

| axis | result |
|---|---|
| `MNIST_INPUT_SCALE` 2-37 | smooth, monotonic; late rate ~ s^0.43, no bifurcation |
| `g_K_max` x0.0-x0.5 | rate rises to 0.27, tonic; no bursting |
| `g_E` x1.3-x1.6 | rate saturates at 0.36, CV *falls* to 0.014 |
| `g_I_local` x0.4 | rate rises, still smooth |
| `g_I_local` x1.6-x3.2 | low-band power rises 0.03 -> 0.35, BUT see below |
| `tau_Cl` x2-x4 | more Cl load, no qualitative change |
| image duration 250-30000 ms | no effect on burstiness |
| rectified (all-positive) weights | no effect; more units hot, still smooth |

**Two false positives I had to discard.**  (1) Raising `g_I_local` looked like
it produced bursting -- CV 0.26, low-band power 0.36 at x2.4.  It does not: the
peak rate is 1.55 Hz against an image refresh of 3.95 Hz, and the phase
concentration of the peaks against the refresh period is R = 0.966.  The
"bursts" are the 253 ms stimulus ripple, standing out more because raising
inhibition *lowers* the mean rate.  The recruitment measure confirms it -- at
n=84 with a naive surround, raising g_I_local monotonically *suppresses*
propagation into the surround (fraction of surround units ever hot: 0.336 at
x1.0, 0.034 at x1.6, 0.019 at x3.2).  (2) In a 30 s run the probe appeared to
raise the LFP s.d. 16-fold at scale 2.  Against a no-probe control at the same
scale the gain is 1.25, and at every other scale 0.96-1.06: the jump was the
network still equilibrating, not the probe.  **Every probe effect below is
reported against a no-probe control run at the same background.**

**Why the drive cannot do it.**  Liou's seizure is a chloride-loading runaway
that needs a compact focus: a small patch driven hard enough and long enough
that [Cl-]in rises locally, E_Cl depolarises, GABA becomes excitatory there,
and the region recruits its neighbours.  The MNIST drive is the opposite of
compact.  At scale 17 it is nonzero on 25% of the sheet (n=56) with per-unit
excursions of +/-1000 pA but a signed field mean of +4 pA, and the global
inhibitory projection is a *mean* over the sheet, so distributed drive raises
global inhibition everywhere it raises excitation.  It saturates individual
units readily -- 21% of units above 0.2 f_max at baseline, versus 121 units
(3.9%) for the reference seizure -- but it never establishes a chloride
gradient.  Duration is not the missing ingredient (a single static image for
30 s changes nothing) and neither is sign balance (rectified weights change
nothing).

**The focal probe.**  `run_mnist(probe=dict(...))` adds Exp1.m's disc -- radius
`r` in normalised units at centre `x`, amplitude `Ic` pA, live during `t`
(seconds) -- to the MNIST field.  At `scale=0` it must reproduce the published
stimulus exactly, and it does: peak unit rate 0.647 and 121 units recruited,
identical to `run_exp1_torus` at n=56, with the rate trace matching to 1e-4
(`test_focal_probe_matches_exp1_reference`).  `probe=None` is bit-identical to
the previous code path (`test_probe_none_leaves_drive_untouched`).

**The transition.**  At n=84 (undriven surround), 80 s, probe 20-23 s:

| background | probe | units ever >0.2 f_max | mean [Cl-]in peak |
|---|---|---|---|
| none | none | 0 | 7.0 mM |
| none | 100 pA | 3463 | 10.9 mM |
| none | 200 pA | 3113 | 10.4 mM |
| none | 400 pA | 3205 | 10.4 mM |
| scale 2 | 200 pA | 5103 (ctl 5163) | 14.1 |
| scale 17 | 200 pA | 6841 (ctl 6852) | 16.1 |

The first two rows are the transition: silence to a propagating seizure with a
~30 s latency after a 3 s pulse.  The threshold is between 0 and 100 pA and has
not been bracketed.  Recruitment is *not* monotonic in Ic (3463 at 100 pA,
3113 at 200) -- consistent with a stronger pulse also loading more chloride
into the inhibitory population, but not investigated.

**The problem this creates for susceptibility measurement.**  With MNIST
running, the probe does nothing measurable (gain 0.96-1.04 on every readout).
The digit stream already drives 18.7% of units above threshold and pushes
[Cl-]in to 16 mM on its own, so the network is far from the quiescent state in
which the focal mechanism is the dominant instability.  A susceptibility
measure that works during training will therefore need either a much smaller
background scale (so the probe is the perturbation, not a rounding error), or a
readout other than "does the probe recruit units" -- e.g. distance to the
Cl-loading boundary in parameter space, measured with the drive briefly off.
This is the same question as the still-open one about comparing WC and Liou
trajectories in a shared reduced coordinate.

**Two runner changes and one drive change.**  `rec=` passes recurrent gains
(`g_E`, `g_I_local`, `g_I_global`, sigmas) through to
`add_standard_recurrent_connections`, which were previously hardcoded.
`image_duration=` overrides `IMAGE_DURATION_MS` for diagnostics only -- the
default is still 250 ms to match WC.  `PatchDrive` now *centres* the patch grid
on the sheet (`offset=`, default centred) instead of anchoring it at the
corner; at n=56 the offset is (0,0) so the locked configuration and every
measurement made at it are unchanged (`test_n56_drive_offset_is_zero`), but at
n=84 the digit sits in the middle with a 14-unit naive border.  Without this,
a larger sheet put the whole surround on two sides.

**Scope.** Single seed, single weight draw, one digit stream ordering.  The
probe threshold is bracketed only to (0, 100] pA; latency, and whether the
seizure terminates, are measured at 80 s and could change with longer runs.

**`n_grid` is dead.** `run_mnist` accepts it and never uses it; the sheet size
comes from `n`.  Left in place but it should be removed or wired up.


## 30-electrode LFP for the transition figure (2026-08-13)

You asked for the 30-electrode version of panel c. Built it, and built the
same view of panel b so the two are comparable.

**Electrode geometry.** 30 electrodes at flat stride 233 on the 84x84 sheet
(units 0, 233, ..., 6757). gcd(233, 84) = 1, so they walk diagonally and
occupy 30 distinct rows AND 30 distinct columns -- the same anti-aliasing
condition as the n=50 ring (stride 83). Minimum pairwise toroidal separation
12.7 units, well beyond the LFP kernel width, so channels are near
independent. Strides 235/239/241 also satisfy gcd=1 but give smaller minimum
separations (10.2-11.4); 233 was chosen on that basis.
14 of the 30 fall inside the driven band (rows/cols 14-69), 16 in the naive
surround.

**Both runs reproduce the stored ring means bit-exactly** (max|diff| = 0
against `LAD['sc0_Ic200']` and `LAD['sc17_Ic200']` in
liou_seizure_transition.npz), so these are the same runs as panels b and c,
not re-simulations with drifted state.

**Panel b (probe, no MNIST) -- what the 30 channels add.** The mean-field
trace shows one onset; the array shows two distinct events:

| event | timing | distance dependence | interpretation |
|---|---|---|---|
| field-wide deflection | 33.1-33.4 s, all 30 ch within 0.3 s | r = 0.26 with distance from focus | global inhibitory projection is a field MEAN -- every unit feels it simultaneously. NOT spread. |
| recruitment (half-max) | 41.4-64.1 s | slope 0.19 s/unit, ~5 units/s | this is the propagating front |

Peak amplitude vs distance from focus r = -0.91; in-band median peak 50438 vs
surround 1638. So the seizure is focal on the LFP as well as in the rate
field, and a single mean-field electrode conflates the global inhibitory
step with the propagating front. **Any future onset-time measurement must use
per-channel amplitude (half-max), not a small absolute threshold** -- an 8x
baseline threshold fires at 33 s on every channel regardless of distance and
would report an apparent spread speed of 1469 units/s, which is not a spread
at all.

Two measurement traps hit on the way, both from thresholding:
- A 5x-median-sd threshold over the whole trace fired at t = 0 s on all 30
  channels -- that is the run's onset transient, not the seizure. Search
  window must start after the probe.
- Per-channel baselines are needed: baseline sd ranges 2.8-122 across the 30
  channels (44x), so one global threshold is meaningless.

**Panel c (MNIST scale 17 + probe) -- confirms the probe does nothing.** Ran
the matched no-probe control on the same electrodes:

| measure | probe | no-probe control | ratio |
|---|---|---|---|
| late (60-80 s) sd, driven ch | 17477 | 17640 | 0.99 |
| late (60-80 s) sd, surround ch | 17846 | 17758 | 1.00 |
| surround onset | 33.1 s | 33.4 s | -- |

The surround channels grow 260-fold (sd 68 -> 17846 from the 10-20 s window
to 60-80 s) while driven channels grow only 2.4-fold, which on its own looks
like invasion of naive tissue. It is not: the control is identical. **The
MNIST stream alone fills the surround by ~60 s.** That is a stronger form of
the susceptibility problem already recorded -- it is not just that the probe
is small relative to the background, it is that by the time the probe's
30 s incubation would elapse there is no quiescent tissue left to invade.

Implication for the training runs: if probe-induced recruitment is the
susceptibility readout, the background must be low enough that the surround
stays quiescent for at least ~40 s (probe latency + front transit). At scale
17 that fails by 60 s. Untested whether a low scale (2-5) leaves a long
enough quiescent window -- the scale-2 control already reached 5163 units.


## CORRECTION: the MNIST background DOES seize (2026-08-13)

You pointed out that panel c shows a focal seizure starting between 10 and 20 s.
You were right, and this reverses the central claim of the two preceding
sections.  What follows supersedes them.

**What I got wrong and why.**  I ran a matched no-probe control at scale 17 and
found the probe had no measurable effect (ratio 0.99-1.00).  That control was
correct and I stand by it -- but it only tested *whether the probe caused the
event*.  I then concluded there was no seizure, which the control never tested.
I attributed the 260-fold growth in the naive-border channels to "the digit
stream filling the sheet", i.e. to spread of driven activity, without checking
whether tissue that receives **zero** input was being recruited.  The lesson is
narrow and worth keeping: a no-probe control identifies the *cause*, never the
*absence*, of an event.

**What the measurement shows.**  At scale 17, n=84, the driven band is the
centred 56x56 patch (3136 units); the remaining 3920 units receive no input at
all.  Active fraction (>0.2 f_max) in the undriven units:

| window | driven band | UNDRIVEN border |
|---|---|---|
| 0-5 s | 4.2% | 0.0000 |
| 10-15 s | 12.2% | 0.0000 |
| 15-20 s | 16.0% | 0.0007 |
| 30-40 s | 20.2% | 0.0425 |
| 60-80 s | 21.2% | 0.1731 |

The undriven fraction is *exactly* zero until 14.8 s and then rises to 19%.
Tissue with no drive on it cannot become active except by recruitment.

**It propagates.**  Binning the undriven units by distance outside the driven
band gives a monotonic outward front: 17.7 s at 1 unit out, 41.7 s at 13 units
out, slope 2.06 s/unit (0.5 units/s), r = 0.99.

**It is not the stimulus-refresh artefact.**  Far-surround activity has a
dominant frequency of 4.30 Hz against a 3.95 Hz image refresh -- close enough
to be the trap logged earlier in this file.  It is not: phase concentration
against the refresh period is R = 0.008 (the genuine refresh artefact scored
0.966).  The activity is intrinsic.

**It is the chloride mechanism.**  Perturbing the two slow variables moves
undriven-tissue recruitment (40-50 s) exactly as Liou's mechanism predicts:

| condition | naive tissue active | peak [Cl-]in |
|---|---|---|
| baseline | 8.3% | 14.7 mM |
| tau_Cl x8 | 100.0% | 36.5 mM |
| g_K_max x3 | 0.002% | 7.4 mM |

Slowing chloride clearance saturates the sheet; strengthening the potassium
brake abolishes recruitment.

**What this changes.**  The claim that the distributed MNIST drive cannot
nucleate a Liou seizure is WITHDRAWN.  It can, at scale 17 on the 84x84 sheet,
with ~15 s latency, and it recruits naive tissue by chloride loading.  The
earlier screen (scale 2-37 etc.) failed to detect this because it scored
burstiness and low-band LFP power on the *driven* units, where the seizure is
hidden under the stimulus; the undriven border is the sensitive readout and I
was not measuring it.

**Consequence for the training runs -- this is now good news.**  We no longer
need an exogenous probe to measure susceptibility.  The digit stream is itself
the provocation, so susceptibility can be read off the drive the network
actually receives during training: latency to undriven-tissue recruitment
(14.8 s here) and the front speed (0.5 units/s) are both graded, both measured
under the training stimulus, and require no perturbation of the protocol.  That
is a closer analogue of the WC readout than the probe would have been.

**Figures corrected.**  liou_mnist_seizes.png is the new evidence figure.
liou_lfp30_mnist_probe.png and panel c of liou_seizure_transition.png were
relabelled -- their traces are unchanged and were always right; only my
annotations were wrong.

**Workspace hazard found while verifying this.** Re-executing a saved figure's
lineage code overwrote ``las_model/mnist_input.py`` with an early 34-line
sketch (the lineage slice includes the cell that first drafted it), silently
breaking 4 tests.  Caught by running the suite from the last SAVED archive
(47/47) against the workspace (44/48) and diffing.  Restored from the archive;
48/48.  The measurement runs above predate the clobber and were re-verified on
the restored module (undriven first-active 14.8 s, frac 0.0023 at 20 s -- both
identical).  Rule: never ``exec`` a full lineage slice into the live workspace;
slice out the plotting tail only, and diff against the saved archive afterwards.


## CRITERION CORRECTION: morphology in the driven area, not spread

User, 2026-08-13: "it's the morphology and evolution of the waveforms in the LFP plot
within the stimulated area that makes the epoch shown in liou_lfp30_probe.png a seizure.
Spread beyond the stimulated area is neither necessary nor sufficient to call the episode
a seizure." And: "activity could spread beyond the stimulated area without necessarily
being ictal"; "activity outlasting the stimulus is not necessarily ictal."

This is a correction to how I justified the previous section, not to its conclusion.
That section leaned on recruitment of zero-input tissue as though it were the definition
of a seizure. It is not. Correct logical structure:

  POSITIVE CRITERION (sufficient on its own): waveform morphology in the DRIVEN area and
  its evolution -- emergence of discrete large-amplitude stereotyped discharges, organised
  into a train that accelerates and grows, converging on the morphology of the canonical
  focal-probe seizure.

  CONFOUND EXCLUSIONS (necessary to trust the trace, never sufficient to call it ictal):
  spread beyond the driven area; activity outlasting the stimulus; absence of phase
  locking to the image refresh. Each rules out an alternative reading. None is evidence
  of ictality by itself. A focal seizure that never propagates is still a seizure;
  propagating non-ictal activity is still not one.

### Measured morphology, driven channels (n=14), scale 17, n=84, 80 s

| epoch | discharge amp | FWHM | IPI | rate |
|---|---|---|---|---|
| early 5-15 s       |  8261 | 50 ms | 495 ms | 1.2-1.3 /s |
| onset 15-25 s      | 34248 | 60 ms | 255 ms | 2.1-2.4 /s |
| established 45-80 s| 39117 | 60 ms | 265 ms | 2.9-3.5 /s |
| REFERENCE focal-probe seizure, NO MNIST, 45-80 s | 46232 | 70 ms | 345 ms | -- |

Amplitude x6.0, FWHM 40 -> 60 ms, IPI 502 -> 250 ms, all converging on the reference
seizure's values. Takeoff 15-20 s. This is the evidence.

### The 250 ms coincidence, and how it was excluded

Established discharge IPI (250-265 ms) sits almost exactly on the 250 ms image refresh,
and phase concentration at 250 ms is R=0.446 (reference seizure 0.112 in the same band).
That is real partial entrainment and cannot be waved away -- my first pass through this
claimed R~0.45 was generic to discharge trains, which the reference run falsifies.

Resolved by changing the stimulus period instead: image_duration 250 -> 400 ms
(x1.60) moves the discharge IPI only 255 -> 285 ms (x1.12), and the R peak to 270 ms,
not 400 ms. The train has its own period, weakly pulled by the stimulus. A stimulus-paced
oscillation would have gone to ~400 ms. This is the right experiment for the question;
phase-concentration-versus-refresh alone was not.

### Consequence for the training-run readout -- SUPERSEDES the previous section's proposal

The previous section proposed measuring susceptibility during training as latency to
naive-tissue recruitment plus front speed. Both are spread measures, so by the criterion
above neither is a susceptibility readout. Replace with morphology measured on the DRIVEN
units: discharge amplitude, FWHM, inter-discharge interval and discharge rate, plus the
latency at which a discrete-discharge train first appears. Keep the spread measures as
secondary descriptors of the event, not as the classifier. Keep image_duration as the
control knob for excluding refresh entrainment in any run where discharge IPI lands near
the refresh period.
