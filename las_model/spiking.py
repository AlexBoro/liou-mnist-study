"""Spiking model with usage-dependent inhibition exhaustion.

Port of ``@SpikingModel/SpikingModel.m`` (constructor + ``IndividualModelUpdate``)
and ``SpikingModel1DStandardTemplate.m`` / ``Exp4Template.m``.

This is the *second* model family in the repo, distinct from the mean-field
model in ``model.py``.  It shares the four state variables (V, phi, Cl_in, g_K)
and the same recurrent connectivity, but replaces the deterministic sigmoid
rate with stochastic spike generation against an exponential intensity, and
adds refractoriness and spike-triggered reset.

MATLAB conventions preserved verbatim (do NOT "tidy" these):

* ``phi`` and ``g_K`` are driven by ``S.S`` -- the spike vector generated at the
  END of the *previous* step.  Their Dirac terms are therefore one step stale
  relative to V, which is updated before spikes are drawn.
* ``E_Cl`` is computed once from the pre-update ``Cl_in`` and reused in both the
  membrane equation and the chloride equation.
* The inhibitory input enters the chloride equation **raw** (``Input.I``) but is
  divided by ``f_max`` (conductance-scaled) in the membrane equation.  Same
  asymmetry as the mean-field model; it is not a typo in the original.
* Synaptic filters decay at the END of the step.
* ``V_reset(x) = x - 20`` is applied to the *membrane potential* of spiking
  units (MATLAB passes ``O.V(Spike)``), despite the template comment calling the
  argument a threshold.
* Refractoriness is enforced by zeroing spikes where ``S.dT < T_refractory``
  AFTER the stochastic draw, so refractory units still consume a random number.
  This matters for exact seed reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kernels import conv_field, gaussian_kernel_1d, sigma_pixels_from_fraction

FARADAY = 96500.0


# ----------------------------------------------------------------------
@dataclass
class SpikingParams:
    """``SpikingModel1DStandardTemplate.m``; ``exp4_template`` applies the
    Figure-4 overrides."""

    n: tuple = (2000, 1)
    # Eq 1 -- membrane potential
    C: float = 100.0
    g_L: float = 4.0
    E_L: float = -57.0
    V0: float = -57.0
    T_refractory: float = 5.0
    f0: float = 0.0
    fs: float = 0.002
    beta: float = 2.5
    dt_AP: float = 0.5
    V_AP: float = 40.0
    V_reset_offset: float = 20.0
    tau_syn_E: float = 15.0
    tau_syn_I: float = 15.0
    E_Esyn: float = 0.0
    # Eq 2 -- dynamic threshold
    phi_0: float = -55.0
    delta_phi_0: float = 2.5
    tau_phi: float = 100.0
    # Eq 3 -- chloride
    Cl_ex: float = 110.0
    Cl_in_eq: float = 6.0
    tau_Cl: float = 5000.0
    # Eq 4 -- sAHP
    tau_K: float = 5000.0
    g_K_max: float = 40.0
    E_K: float = -90.0

    @property
    def f_max(self) -> float:
        return 1.0 / self.T_refractory

    @property
    def Vd_Cl(self) -> float:
        """``0.25*sqrt(2)/12*20^3/1000`` pL, verbatim from the template."""
        return 0.25 * np.sqrt(2) / 12 * 20**3 / 1000

    def f(self, u):
        """Spike intensity, kHz.  ``f(u) = f0 + fs*exp(u/beta)``."""
        return self.f0 + self.fs * np.exp(u / self.beta)


def exp4_template(panel: str = "A") -> SpikingParams:
    """``Exp4Template.m``: beta 2.5 -> 1.5 mV, E_L and V -> -60 mV.

    The panel A/B difference is *not* in this template -- ``Exp4.m`` changes the
    two inhibitory projection strengths instead (see ``run_exp4``).
    """
    p = SpikingParams()
    p.beta = 1.5
    p.E_L = -60.0
    p.V0 = -60.0
    return p


# ----------------------------------------------------------------------
@dataclass
class ConvProjection1D:
    """1D convolution projection; ``Value`` is the post-synaptic drive."""

    type: str
    kernel: np.ndarray
    w_post: float

    def project(self, out_value: np.ndarray) -> np.ndarray:
        return conv_field(out_value, self.kernel, np.ones(1)) * self.w_post


@dataclass
class GlobalProjection1D:
    """``Method='function'`` projection: ``W = @(x) sum(x(:))/prod(n)``."""

    type: str
    n_units: int
    w_post: float

    def project(self, out_value: np.ndarray) -> np.ndarray:
        return np.full_like(out_value, out_value.sum() / self.n_units) * self.w_post


# ----------------------------------------------------------------------
class SpikingModel:
    def __init__(self, p: SpikingParams, seed: int = 0):
        self.p = p
        self.n = tuple(p.n)
        self.rng = np.random.default_rng(seed)
        shape = self.n

        self.V = np.full(shape, p.V0, dtype=np.float64)
        self.phi = np.full(shape, p.phi_0, dtype=np.float64)
        self.Cl_in = np.full(shape, p.Cl_in_eq, dtype=np.float64)
        self.g_K = np.zeros(shape, dtype=np.float64)

        # Spike-associated state (@SpikingNetwork .S)
        self.S = np.zeros(shape, dtype=bool)
        self.dT = np.zeros(shape, dtype=np.float64)

        self.Input_E = np.zeros(shape, dtype=np.float64)
        self.Input_I = np.zeros(shape, dtype=np.float64)

        self.t = 0.0
        self.proj: list = []
        self._proj_value: list = []
        self.ext = []

    # ------------------------------------------------------------------
    def add_standard_recurrent_connections(
        self,
        sigma_E_frac: float = 0.02,
        sigma_I_frac: float = 0.03,
        g_E: float = 100.0,
        g_I_local: float = 250.0,
        g_I_global: float = 50.0,
    ) -> None:
        """``StandardRecurrentConnection.m`` with ``Adjust=false``."""
        sE = sigma_pixels_from_fraction(self.n, sigma_E_frac)
        sI = sigma_pixels_from_fraction(self.n, sigma_I_frac)
        self.proj = [
            ConvProjection1D("E", gaussian_kernel_1d(sE[0]), g_E),
            ConvProjection1D("I", gaussian_kernel_1d(sI[0]), g_I_local),
            GlobalProjection1D("I", int(np.prod(self.n)), g_I_global),
        ]
        self._proj_value = [np.zeros(self.n) for _ in self.proj]

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """One ``Update(O, dt)`` cycle: IndividualModelUpdate, then Project."""
        p = self.p
        f_max = p.f_max

        # --- external input ------------------------------------------------
        I_ext = np.zeros(self.n)
        for e in self.ext:
            I_ext = I_ext + e(self.t, dt, self.n)

        # --- accumulate projections into the synaptic filters ---------------
        for pr, val in zip(self.proj, self._proj_value):
            if pr.type == "E":
                self.Input_E = self.Input_E + val / p.tau_syn_E
            else:
                self.Input_I = self.Input_I + val / p.tau_syn_I

        # --- Eq 1: membrane potential --------------------------------------
        g_sum = p.g_L + self.Input_E / f_max + self.Input_I / f_max + self.g_K / f_max
        E_Cl = 26.7 * np.log(self.Cl_in / p.Cl_ex)      # pre-update Cl_in
        V_inf = (
            p.g_L * p.E_L
            + self.Input_E / f_max * p.E_Esyn
            + self.Input_I / f_max * E_Cl
            + self.g_K / f_max * p.E_K
            + I_ext
        ) / g_sum
        tau_V_eff = p.C / g_sum
        self.V = V_inf + (self.V - V_inf) * np.exp(-dt / tau_V_eff)

        # --- Eq 2: threshold (Dirac term uses PREVIOUS step's spikes) ------
        self.phi = (
            p.phi_0
            + (self.phi - p.phi_0) * np.exp(-dt / p.tau_phi)
            + p.delta_phi_0 * self.S
        )

        # --- Eq 3: chloride ------------------------------------------------
        T_AP = p.dt_AP - self.dT
        T_AP = T_AP * (T_AP > 0)
        T_non_AP = dt - T_AP
        Veff = T_AP * p.V_AP / dt + T_non_AP * self.V / dt
        Cl_in_inf = (
            p.tau_Cl / p.Vd_Cl / FARADAY * self.Input_I * (Veff - E_Cl) + p.Cl_in_eq
        )
        self.Cl_in = Cl_in_inf + (self.Cl_in - Cl_in_inf) * np.exp(-dt / p.tau_Cl)

        # --- Eq 4: sAHP (Dirac term uses PREVIOUS step's spikes) -----------
        self.g_K = self.g_K * np.exp(-dt / p.tau_K) + p.g_K_max * self.S / p.tau_K

        # --- spike generation ----------------------------------------------
        spike = p.f(self.V - self.phi) * dt > self.rng.random(self.n)
        spike[self.dT < p.T_refractory] = False
        self.V[spike] = self.V[spike] - p.V_reset_offset
        self.S = spike

        # --- S-derived variables, same cycle -------------------------------
        self.dT = self.dT + dt
        self.dT[self.S] = 0.0

        out_value = self.S.astype(np.float64)   # flag_STP is false in Exp4

        # --- synaptic filters decay at the END of the step -----------------
        self.Input_E = self.Input_E * np.exp(-dt / p.tau_syn_E)
        self.Input_I = self.Input_I * np.exp(-dt / p.tau_syn_I)

        self.t += dt

        # --- Step 2 of NeuralNetwork.Update: project ------------------------
        self._proj_value = [pr.project(out_value) for pr in self.proj]

    # ------------------------------------------------------------------
    def psc(self):
        """EPSC / IPSC as ``@Recorder/RecordExtraVar.m`` computes them.

        The conductance is recomputed as ``Input.<T> + sum(Proj.In.Value)/tau``
        because ``Input`` alone is stale at recording time: the recorder runs
        after ``Update``, at which point ``Input`` has already been decayed by
        one step and ``Project`` has written the *next* step's arrival into
        ``Proj.In.Value``, which will not be folded in until the top of the
        next step.  Adding it back is the author's compensation for that.

        This is NOT a double count.  ``Proj.In.Value`` is overwritten (not
        accumulated) by ``Project``, and the fold-in at the top of the next
        step is the only place it enters ``Input``, so each arrival is counted
        exactly once.  Verified numerically: the recorded conductance equals
        the conductance the equations use on the FOLLOWING step, to 0.0.

        The residual defect is a one-step pairing mismatch: the recorded ``g``
        leads the recorded ``V`` by one dt.  ``corr(recorded, consistent)``
        = 0.939, rising to 0.99999 once ``g`` is shifted back one step; the
        means agree to 0.05%.  So the effect is a 1 ms phase error in the PSC
        traces, not an amplitude error.
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
