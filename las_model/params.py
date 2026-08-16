"""Model parameter templates.

Port of ``gen_param.m``, ``MeanField1DStandardTemplate.m``,
``MeanField2DStandardTemplate.m``, ``Exp1Template.m`` and ``Exp2Template.m``.

Every parameter is stored as a full array of shape ``n`` exactly as the MATLAB
templates do (via ``gen_param``), so heterogeneous parameter maps can be
introduced later without touching the integrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

FARADAY = 96485.3329  # C/mol  (MATLAB: Faraday constant, see MeanFieldModel.m)


def gen_param(mu: float, sigma: float, n, flag: int = 0, rng=None) -> np.ndarray:
    """Port of ``gen_param.m``.

    flag 0 -> normal distribution, flag 1 -> log-normal.  With ``sigma == 0``
    both collapse to a constant field, which is the only case the published
    templates use.
    """
    n = tuple(n)
    if sigma == 0:
        return np.full(n, float(mu))
    rng = np.random.default_rng() if rng is None else rng
    if flag == 0:
        return mu + sigma * rng.standard_normal(n)
    sigma_log = np.sqrt(np.log1p((sigma / mu) ** 2))
    mu_log = np.log(mu) - 0.5 * sigma_log**2
    return np.exp(mu_log + sigma_log * rng.standard_normal(n))


@dataclass
class MeanFieldParams:
    """Intrinsic parameters of the mean-field rate model."""

    n: tuple

    # --- Equation 1: membrane potential ---------------------------------
    V0: np.ndarray = None
    C: np.ndarray = None
    g_L: np.ndarray = None
    E_L: np.ndarray = None
    T_refractory: np.ndarray = None
    f_max: np.ndarray = None
    beta: np.ndarray = None
    tau_syn_E: np.ndarray = None
    tau_syn_I: np.ndarray = None
    E_Esyn: np.ndarray = None

    # --- Equation 2: dynamic threshold ----------------------------------
    phi0: np.ndarray = None
    phi_inf_gain: float = 60.0  # phi_inf(u) = phi_0 + u * 60
    tau_phi: np.ndarray = None

    # --- Equation 3: chloride -------------------------------------------
    Vd_Cl: np.ndarray = None
    Cl_ex: float = 110.0
    Cl_in_eq: np.ndarray = None
    tau_Cl: np.ndarray = None

    # --- Equation 4: slow AHP -------------------------------------------
    tau_K: np.ndarray = None
    g_K_max: np.ndarray = None
    E_K: np.ndarray = None

    # Optional multiplicative mask on the activation function (Exp1)
    mask: np.ndarray = None

    def activation(self, u: np.ndarray) -> np.ndarray:
        """f(u) = f_max / (1 + exp(-u/beta)), optionally masked (Exp1)."""
        r = self.f_max / (1.0 + np.exp(-u / self.beta))
        if self.mask is not None:
            r = r * self.mask
        return r

    def phi_inf(self, u: np.ndarray) -> np.ndarray:
        return self.phi0 + u * self.phi_inf_gain


def _standard_template(n) -> MeanFieldParams:
    """Shared body of MeanField{1D,2D}StandardTemplate.m.

    The two MATLAB templates are numerically identical apart from network size
    (and the 2D one additionally exposes the same values), so a single builder
    covers both.
    """
    n = tuple(n)
    T_refractory = gen_param(5, 0, n, 1)
    # Vd_Cl = 0.25 * sqrt(2) / 12 * 20^3 / 1000  pL  (~0.2357 pL)
    vd = 0.25 * np.sqrt(2) / 12 * 20**3 / 1000
    return MeanFieldParams(
        n=n,
        V0=gen_param(-58, 0, n, 0),
        C=gen_param(100, 0, n, 1),
        g_L=gen_param(4, 0, n, 1),
        E_L=gen_param(-58, 0, n, 0),
        T_refractory=T_refractory,
        f_max=1.0 / T_refractory,
        beta=gen_param(2.5, 0, n, 1),
        tau_syn_E=gen_param(15, 0, n, 0),
        tau_syn_I=gen_param(15, 0, n, 0),
        E_Esyn=gen_param(0, 0, n, 0),
        phi0=gen_param(-45, 0, n, 0),
        tau_phi=gen_param(100, 0, n, 1),
        Vd_Cl=np.full(n, vd),
        Cl_ex=110.0,
        Cl_in_eq=gen_param(6, 0, n, 1),
        tau_Cl=gen_param(5000, 0, n, 1),
        tau_K=gen_param(5000, 0, n, 1),
        g_K_max=gen_param(40, 0, n, 1),
        E_K=gen_param(-90, 0, n, 0),
    )


def mean_field_1d_standard_template(n=(500, 1)) -> MeanFieldParams:
    """MeanField1DStandardTemplate.m"""
    return _standard_template(n)


def mean_field_2d_standard_template(n=(100, 100)) -> MeanFieldParams:
    """MeanField2DStandardTemplate.m"""
    return _standard_template(n)


def exp1_template(n=(100, 100)) -> MeanFieldParams:
    """Exp1Template.m -- 2D standard template, no parameter overrides.

    Exp1Template.m is ``MeanField2DStandardTemplate;`` verbatim, so E_L and the
    initial V both stay at -58 mV (unlike Exp2Template, which raises them to
    -57.5 mV).
    """
    return mean_field_2d_standard_template(n)


def exp2_template(n=(500, 1)) -> MeanFieldParams:
    """Exp2Template.m -- 1D standard template with V and E_L raised to -57.5 mV."""
    p = mean_field_1d_standard_template(n)
    p.V0 = gen_param(-57.5, 0, n, 0)
    p.E_L = gen_param(-57.5, 0, n, 0)
    return p


def exp6_template(n=(50, 50)) -> MeanFieldParams:
    """Exp6Template.m -- 2D standard template with V and E_L lowered to -58 mV.

    ``MeanField2DStandardTemplate`` sets both to -57 mV; Exp6Template.m
    overrides them to -58 mV, which is what makes the resting state quiescent
    enough for the round-field seizure to terminate.  Network size stays at the
    standard 2D 50x50 (Exp6Template does *not* raise it to 100x100 the way
    Exp1Template does).
    """
    p = mean_field_2d_standard_template(n)
    p.V0 = gen_param(-58, 0, n, 0)
    p.E_L = gen_param(-58, 0, n, 0)
    return p
