## Local field potential estimation

An LFP-like signal was computed from the recorded synaptic currents of the
integrate-and-fire spiking model following the reference weighted-sum proxy of
Mazzoni et al. (2015), as specified in Liou et al. (2020).

### Synaptic currents

At each timestep the excitatory and inhibitory synaptic currents of every unit
were recorded as driving-force currents,

$$I_E(x,t) = \frac{g_E(x,t)\,\bigl[E_\mathrm{syn} - V(x,t)\bigr]}{f_\mathrm{max}},
\qquad
I_I(x,t) = \frac{g_I(x,t)\,\bigl[E_\mathrm{Cl}(x,t) - V(x,t)\bigr]}{f_\mathrm{max}},$$

where $V$ is the membrane potential, $E_\mathrm{syn}=0$ mV is the excitatory
reversal potential, $f_\mathrm{max}=0.2$ is the maximum firing-rate scale, and
$E_\mathrm{Cl}(x,t) = 26.7\ln\bigl[\mathrm{Cl}^-_\mathrm{in}(x,t)/\mathrm{Cl}^-_\mathrm{ex}\bigr]$
is the instantaneous chloride reversal potential, which evolves with the
activity-dependent intracellular chloride load. Because $V$ rests well below
$E_\mathrm{syn}$ and above $E_\mathrm{Cl}$, $I_E$ is positive (depolarising) and
$I_I$ is negative (hyperpolarising).

The synaptic conductances used for this readout were reconstructed as
$g = g_\mathrm{filt} + \sum_j P_j/\tau_\mathrm{syn}$, where $g_\mathrm{filt}$ is
the exponentially filtered conductance state and $P_j$ are the recurrent
projection values pending at the time of recording. This reconstruction
compensates for the conductance state having already been decayed at the point
in the integration cycle where recording occurs; each synaptic arrival enters
the sum exactly once. It leaves the recorded conductance advanced by one
integration step relative to the membrane potential it multiplies, a
$1$ ms offset that is negligible on the $50$-$100$ ms timescale of the
discharges analysed here (see Validation).

### LFP proxy

The LFP was modelled as a linear combination of the two synaptic currents, with
the excitatory contribution delayed to account for the slower time course of
the associated return currents,

$$\Lambda(x,t) = I_E(x,\,t-\Delta t) - \alpha\, I_I(x,t),$$

with $\alpha = 1.65$ and $\Delta t = 6$ ms, the values recommended by Mazzoni
et al. (2015). Since $I_E \geq 0$ and $I_I < 0$, this difference sums the
magnitudes of the two currents, $\Lambda = |I_E| + \alpha|I_I|$, reflecting that
an extracellular electrode registers excitatory and inhibitory transmembrane
currents as deflections of the same polarity. The first $\Delta t$ samples of
each simulation were discarded.

### Spatial integration

To represent the finite spatial sampling of an extracellular electrode, the
contribution of neighbouring units was weighted by an exponential kernel,

$$\mathrm{LFP}(x_0,t) = \sum_{x} \exp\!\left(-\frac{|x-x_0|}{\sigma_\mathrm{LFP}N}\right)\Lambda(x,t),$$

with $\sigma_\mathrm{LFP} = 0.025$ expressed in normalised spatial units, i.e.
as a fraction of the modelled cortical extent, consistent with the convention
used for the excitatory and inhibitory connectivity widths
($\sigma_E = 0.02$, $\sigma_I = 0.03$). For the $N = 2000$ unit sheet simulated
here this corresponds to an e-folding length of $50$ units; units beyond
$\pm 231$ contribute less than $1\%$ of the peak weight. The kernel is
unnormalised, so the resulting signal is a weighted sum rather than a mean and
is reported in arbitrary units; only relative amplitude and timing are
interpreted.

Signals were read out at two locations, $x_0 = 0.15$ and $x_0 = 0.35$ in
normalised coordinates (units $300$ and $700$), both distal to the focal
stimulus at $x = 0.05$.

### Validation

Two properties of the implementation were verified numerically. First, the
single-site readout used for the figures was confirmed to equal a full
convolution of the entire sheet to within $5\times10^{-13}$ at interior and
boundary locations, including the placement of the leading undefined samples;
this guards against the asymmetric-truncation errors that arise when a kernel
is applied near an array edge. Second, the conductance reconstruction described
above was confirmed to reproduce the conductance entering the membrane equation
on the subsequent integration step exactly (maximum absolute difference $0$),
establishing that no synaptic arrival is counted twice; the resulting
current traces correlate at $r = 0.99999$ with a step-consistent
reconstruction, and their means agree to within $0.05\%$.
