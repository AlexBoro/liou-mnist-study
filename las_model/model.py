"""Mean-field rate model with usage-dependent inhibition exhaustion.

Port of ``@MeanFieldModel/MeanFieldModel.m`` (constructor +
``IndividualModelUpdate``), ``@NeuralNetwork/NeuralNetwork.m`` (``Update``),
``@Projection/Project.m`` and ``StandardRecurrentConnection.m``.

The statement order inside :meth:`MeanFieldModel.step` is a literal
translation of the MATLAB and must not be "tidied up": several quantities are
deliberately read at their pre-update values.  The ones that matter:

* ``phi`` and ``g_K`` are advanced with the *previous* step's firing rate,
  before ``R_f`` is recomputed.
* ``E_Cl`` is computed from the *pre-update* chloride, then used both in the
  ``V`` update and in the chloride update itself.
* The chloride update uses the *post-update* ``V``.
* The membrane-potential equation divides the synaptic inputs by ``f_max`` to
  convert them to conductances, but the chloride equation uses ``Input_I``
  *raw* (i.e. scaled by ``f_max`` relative to a conductance).  This is exactly
  what the MATLAB does; see PORT_NOTES.md.
* Synaptic inputs are consumed at their accumulated (pre-decay) value and
  decayed only at the end of the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .kernels import conv_field, gaussian_kernel_1d, sigma_pixels_from_fraction
from .params import MeanFieldParams

FARADAY = 96500.0  # MATLAB hard-codes 96500 inside IndividualModelUpdate


@dataclass
class ConvProjection:
    """A 'convolution'/'linear' Projection with separable Gaussian kernel."""

    type: str  # 'E' or 'I'
    k1: np.ndarray
    k2: np.ndarray
    w_post: float
    value: np.ndarray = None
    topology: str = "linear"  # 'linear' (paper figures) or 'circular' (torus)

    def project(self, source_value: np.ndarray) -> np.ndarray:
        return conv_field(source_value, self.k1, self.k2, self.topology) * self.w_post


@dataclass
class GlobalProjection:
    """A 'function' Projection: W = @(x) sum(x(:))/prod(n)."""

    type: str
    n_total: int
    w_post: float

    def project(self, source_value: np.ndarray) -> np.ndarray:
        return source_value.sum() / self.n_total * self.w_post


@dataclass
class ExternalInput:
    """Port of ``@ExternalInput/ExternalInput.m`` (deterministic + OU noise).

    ``deterministic(x1, x2, t)`` receives 1-based neuron indices (matching the
    MATLAB ``meshgrid(1:n(2), 1:n(1))`` convention) and time in ms, and returns
    a current in pA.
    """

    deterministic: Optional[Callable] = None
    tau_x: Optional[float] = None
    tau_t: Optional[float] = None
    sigma: float = 0.0
    tmax: Optional[float] = None
    _iz: np.ndarray = None

    def evaluate(self, x1, x2, t, dt, n, rng) -> np.ndarray:
        if self.tmax is not None and t > self.tmax:
            return np.zeros(n)
        out = np.zeros(n)
        if self.deterministic is not None:
            out = out + self.deterministic(x1, x2, t)
        if self.sigma:
            z = rng.standard_normal(n)
            if self.tau_x:
                z = _space_filter(z, self.tau_x)
            if self.tau_t:
                if self._iz is None:
                    self._iz = rng.standard_normal(n)
                # Exact-decay OU step, as ``ExternalInput.m`` writes it:
                #   Iz <- Iz*exp(-dt/tau_t) + sqrt(2*dt/tau_t)*z
                # An Euler decay (1 - dt/tau_t) understates the stationary
                # sd by 1.6% at tau_t = 15 ms, dt = 1 ms.
                self._iz = self._iz * np.exp(-dt / self.tau_t) + np.sqrt(
                    2 * dt / self.tau_t
                ) * z
                z = self._iz
            out = out + self.sigma * z
        return out


def _space_filter(z: np.ndarray, tau_x: float) -> np.ndarray:
    """First-order spatial IIR filter (the OU-in-space of ExternalInput.m)."""
    from scipy import signal

    dx = 1.0
    b = np.sqrt(2 * dx / tau_x)
    a = [1.0, (dx - tau_x) / tau_x]
    zi = (1 - b) * z[0]
    out, _ = signal.lfilter([b], a, z, axis=0, zi=zi[None, :] * (-a[1]))
    return out


class MeanFieldModel:
    """The 2D/1D mean-field rate model of Liou et al. (2020)."""

    def __init__(self, params: MeanFieldParams, seed: int = 0):
        self.p = params
        self.n = tuple(params.n)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        # Dynamical variables (constructor of MeanFieldModel.m)
        self.V = params.V0.copy()
        self.phi = params.phi0.copy()
        self.Cl_in = params.Cl_in_eq.copy()
        # see the clamp in step(); 0.1 mM is ~1.7% of Cl_in_eq (6 mM) and puts
        # E_Cl at 26.7*log(0.1/110) = -187 mV, i.e. far below any physiological
        # reversal but finite, so the run continues and the departure is counted.
        self.cl_floor = 0.1
        self.cl_floor_hits = 0
        # A raw hit count conflates "one unit clamped for 1000 steps" with
        # "1000 units clamped once".  Track both margins so a sweep can tell
        # a localised, brief departure from a pervasive one.
        self.cl_floor_units = np.zeros_like(np.asarray(params.Cl_in_eq, dtype=bool))
        self.cl_floor_steps = 0
        self.g_K = np.zeros(self.n)
        self.R_f = np.zeros(self.n)

        # Synaptic input accumulators
        self.Input_E = np.zeros(self.n)
        self.Input_I = np.zeros(self.n)

        self.proj: List = []
        self.ext: List[ExternalInput] = []

        # 1-based index grids, matching the MATLAB meshgrid convention
        self.x2, self.x1 = np.meshgrid(
            np.arange(1, self.n[1] + 1), np.arange(1, self.n[0] + 1)
        )

        # Projection outputs, indexed like self.proj
        self._proj_value = [np.zeros(self.n) for _ in self.proj]

    # ------------------------------------------------------------------
    def add_standard_recurrent_connections(
        self,
        sigma_E_frac: float = 0.02,
        sigma_I_frac: float = 0.03,
        g_E: float = 100.0,
        g_I_local: float = 250.0,
        g_I_global: float = 50.0,
        topology: str = "linear",
    ) -> None:
        """Port of ``StandardRecurrentConnection.m`` (Adjust=false).

        ``topology='linear'`` reproduces the paper (zero-padded convolution;
        units near the edge receive less recurrent drive, which is exactly the
        boundary effect the MATLAB's unused ``Adjust`` flag was written to
        compensate).  ``topology='circular'`` places the sheet on a torus, so
        every unit has an identical connectivity neighbourhood and no
        ``Adjust`` correction is meaningful.
        """
        sE = sigma_pixels_from_fraction(self.n, sigma_E_frac)
        sI = sigma_pixels_from_fraction(self.n, sigma_I_frac)
        self.proj = [
            ConvProjection("E", gaussian_kernel_1d(sE[0]), gaussian_kernel_1d(sE[1]),
                           g_E, topology=topology),
            ConvProjection("I", gaussian_kernel_1d(sI[0]), gaussian_kernel_1d(sI[1]),
                           g_I_local, topology=topology),
            GlobalProjection("I", int(np.prod(self.n)), g_I_global),
        ]
        self._proj_value = [np.zeros(self.n) for _ in self.proj]

    # ------------------------------------------------------------------
    def psc(self):
        """EPSC / IPSC as ``@Recorder/RecordExtraVar.m`` computes them.

        Identical in form to :meth:`SpikingModel.psc` -- the recorder is
        defined on ``@GeneralModel`` in the MATLAB, so it is shared by every
        model class and the mean-field version must not diverge.

        The conductance is recomputed as ``Input_<T> + sum(Proj.In.Value)/tau``
        rather than read from ``Input_<T>`` alone, because ``Input`` is stale
        at recording time: :meth:`step` decays it at the end of the cycle and
        then writes the *next* step's arrival into ``_proj_value``, which is
        not folded in until the top of the following step.  Adding it back is
        the author's compensation.  It is NOT a double count -- ``_proj_value``
        is overwritten (not accumulated) by ``project``, and the fold-in at the
        top of the next step is the only place it enters ``Input``.

        NOTE the residual one-step pairing defect carried over from the
        spiking model: the returned ``g`` leads ``self.V`` by one ``dt``, a
        1 ms phase error at ``dt = 1``, negligible against the 50-100 ms
        discharge timescale but not zero.

        Returns ``(epsc, ipsc)`` shaped like the field.  ``epsc >= 0`` always
        (depolarising, since ``V < E_Esyn``).

        **``ipsc`` is NOT always negative.**  An earlier version of this
        docstring claimed it was; measurement refutes that.  ``ipsc =
        gI*(E_Cl - V)/f_max`` is positive wherever ``V < E_Cl``, i.e. in units
        driven BELOW the chloride reversal -- post-discharge hyperpolarisation
        -- and increasingly as seizure chloride loading raises ``E_Cl``.
        Measured on the 80 s focal run: positive in 2.3% of all samples,
        rising to ~27% of units late in the event.

        Consequence for anything downstream: in
        ``Lambda = I_E(t-d) - alpha*I_I(t)`` the inhibitory term SUBTRACTS
        where ``ipsc > 0`` instead of adding.  Do not write code that assumes
        a fixed inhibitory sign; see ``lfp.ipsc_sign_report``.
        """
        p = self.p
        gE = self.Input_E.copy()
        gI = self.Input_I.copy()
        for pr, val in zip(self.proj, self._proj_value):
            if pr.type == "E":
                gE = gE + val / p.tau_syn_E
            else:
                gI = gI + val / p.tau_syn_I
        E_Cl = 26.7 * np.log(self.Cl_in / p.Cl_ex)
        epsc = gE * (p.E_Esyn - self.V) / p.f_max
        ipsc = gI * (E_Cl - self.V) / p.f_max
        return epsc, ipsc

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """One ``Update(O, dt)`` cycle: model update, then projection."""
        p = self.p

        # --- Collect external input --------------------------------------
        I_ext = np.zeros(self.n)
        for e in self.ext:
            I_ext = I_ext + e.evaluate(self.x1, self.x2, self.t, dt, self.n, self.rng)

        # --- Collect projections into the synaptic accumulators ----------
        for pr, val in zip(self.proj, self._proj_value):
            if pr.type == "E":
                self.Input_E = self.Input_E + val / p.tau_syn_E
            else:
                self.Input_I = self.Input_I + val / p.tau_syn_I

        # --- Equation 1: membrane potential ------------------------------
        gE = self.Input_E / p.f_max
        gI = self.Input_I / p.f_max
        gK = self.g_K / p.f_max
        g_sum = p.g_L + gE + gI + gK
        E_Cl = 26.7 * np.log(self.Cl_in / p.Cl_ex)
        V_inf = (p.g_L * p.E_L + gE * p.E_Esyn + gI * E_Cl + gK * p.E_K + I_ext) / g_sum
        tau_V_eff = p.C / g_sum
        self.V = V_inf + (self.V - V_inf) * np.exp(-dt / tau_V_eff)

        # --- Equation 2: dynamic threshold (uses previous rate) ----------
        self.phi = self.phi + (p.phi_inf(self.R_f / p.f_max) - self.phi) * (
            1 - np.exp(-dt / p.tau_phi)
        )

        # --- Equation 3: chloride (uses new V, old E_Cl, raw Input_I) ----
        Cl_in_inf = (
            p.tau_Cl / p.Vd_Cl / FARADAY * self.Input_I * (self.V - E_Cl) + p.Cl_in_eq
        )
        self.Cl_in = Cl_in_inf + (self.Cl_in - Cl_in_inf) * np.exp(-dt / p.tau_Cl)
        # PORT ADDITION (not in Liou's MATLAB).  Eq. 1 takes
        # E_Cl = 26.7*log(Cl_in/Cl_ex), which is NaN for Cl_in <= 0.  Liou never
        # reaches that because every stimulus in the paper is a POSITIVE current
        # injection; our WC-analogous MNIST drive uses signed per-pixel weights,
        # so ~45% of driven units get hyperpolarising current, which drives
        # Cl_in down without bound (the equation above has no floor).  A
        # concentration cannot be negative, so clamping is more faithful than
        # propagating NaN -- but it IS a departure, so it is counted and
        # surfaced rather than applied silently.  Set cl_floor=None to restore
        # the unguarded MATLAB behaviour.
        if self.cl_floor is not None:
            n_hit = int(np.count_nonzero(self.Cl_in < self.cl_floor))
            if n_hit:
                self.cl_floor_hits += n_hit
                self.cl_floor_steps += 1
                self.cl_floor_units |= (self.Cl_in < self.cl_floor)
                self.Cl_in = np.maximum(self.Cl_in, self.cl_floor)

        # --- Equation 4: slow AHP (uses previous rate) -------------------
        self.g_K = self.g_K * np.exp(-dt / p.tau_K) + p.g_K_max * self.R_f * dt / p.tau_K

        # --- Generate the new firing rate --------------------------------
        self.R_f = p.activation(self.V - self.phi)

        # --- Emit spikes to outgoing projections -------------------------
        out_value = self.R_f * dt

        # --- Decay the synaptic filters ----------------------------------
        self.Input_E = self.Input_E * np.exp(-dt / p.tau_syn_E)
        self.Input_I = self.Input_I * np.exp(-dt / p.tau_syn_I)

        self.t += dt

        # --- Step 2 of NeuralNetwork.Update: project ---------------------
        self._proj_value = [pr.project(out_value) for pr in self.proj]
