"""Experiment drivers -- ports of ``Exp1.m`` (2D, Figure 1) and ``Exp2.m``
(1D, Figure 2)."""

from __future__ import annotations

import numpy as np

from .model import ExternalInput, MeanFieldModel
from .params import exp1_template, exp2_template, exp6_template


def sigma_from_diffusion(D: float, dt: float) -> float:
    """Noise amplitude (pA) giving a current-noise diffusion coefficient ``D``.

    The paper reports noise strength as a diffusion coefficient
    (200 pA^2/ms for Figures 6 and 7), *not* as an amplitude, but
    ``ExternalInput.m`` takes an amplitude in ``Random.sigma``.  For the white
    -noise case (``tau_t`` empty) the injected current is a fresh
    ``sigma * N(0,1)`` each step, so the accumulated charge over time ``T`` has
    variance ``sigma^2 * dt * T``.  Matching ``Var(Q) = 2 D T`` gives

        sigma = sqrt(2 D / dt).

    At ``D = 200`` and ``dt = 1`` ms this is 20 pA -- consistent with the
    sigma_s = 20 pA quoted for the Figure 7 spiking model at the same dt.
    Verified numerically over 4000 realisations: the recovered diffusion
    coefficient is 200.7 at dt=1, 196.0 at dt=0.5 and 198.4 at dt=2.

    The dt-dependence is the point: passing 200 directly as ``sigma`` would
    inject a diffusion coefficient of 20000 pA^2/ms, 100x the intended noise.
    """
    return float(np.sqrt(2.0 * D / dt))


def _round_mask(n) -> np.ndarray:
    """Port of the Exp1.m round boundary mask.

    ``rr < 0.5`` in normalised units, with cell centres offset by half a unit
    so the disc is centred on the grid.
    """
    xx, yy = np.meshgrid(
        np.arange(-(n[1] / 2) + 0.5, n[1] / 2), np.arange(-(n[0] / 2) + 0.5, n[0] / 2)
    )
    xx = xx / n[1]
    yy = yy / n[0]
    return (np.sqrt(xx**2 + yy**2) < 0.5).astype(float)


def run_exp1(
    T_end: float = 99990.0,
    dt: float = 1.0,
    write_cycle: float = 10.0,
    n=(100, 100),
    Ic: float = 200.0,
    stim_t=(2.0, 5.0),
    stim_x=(0.5, 0.1),
    stim_r: float = 0.05,
    seed: int = 0,
    progress_every: float = 20000.0,
    fine_window=None,
):
    """Port of ``Exp1.m`` -- the 2D simulation behind Figure 1C.

    Returns a dict with the recorded fields.  ``fine_window`` optionally names
    a ``(t_start, t_stop)`` interval in ms over which the firing rate is also
    recorded at full ``dt`` resolution, for the fast-timescale panels.
    """
    p = exp1_template(n)
    p.mask = _round_mask(n)
    O = MeanFieldModel(p, seed=seed)
    O.add_standard_recurrent_connections()

    # External input: a disc of radius stim_r at normalised position stim_x,
    # on during stim_t.  Index arithmetic follows Exp1.m literally, including
    # its transposed use of n(2)/n(1) (immaterial for a square grid).
    def det(x1, x2, t):
        if not (stim_t[1] * 1000 > t > stim_t[0] * 1000):
            return 0.0
        r = np.sqrt(
            ((x1 - 1) / (n[1] - 1) - stim_x[0]) ** 2
            + ((x2 - 1) / (n[0] - 1) - stim_x[1]) ** 2
        )
        return (r < stim_r) * Ic

    O.ext = [ExternalInput(deterministic=det, tmax=stim_t[1] * 1000)]

    ts, rates, Vs, Cls, gKs, phis = [], [], [], [], [], []
    fine_ts, fine_rates = [], []
    next_write = 0.0
    while O.t < T_end:
        if O.t >= next_write:
            ts.append(O.t)
            rates.append(O.R_f.copy())
            Vs.append(O.V.copy())
            Cls.append(O.Cl_in.copy())
            gKs.append(O.g_K.copy())
            phis.append(O.phi.copy())
            next_write += write_cycle
        if fine_window is not None and fine_window[0] <= O.t < fine_window[1]:
            fine_ts.append(O.t)
            fine_rates.append(O.R_f.copy())
        O.step(dt)
        if progress_every and O.t % progress_every < dt:
            print(f"t={O.t/1000:.0f}s  max_rate={O.R_f.max()/p.f_max.max():.3f}", flush=True)

    out = dict(
        t=np.array(ts),
        rate=np.array(rates, dtype=np.float32),
        V=np.array(Vs, dtype=np.float32),
        Cl_in=np.array(Cls, dtype=np.float32),
        g_K=np.array(gKs, dtype=np.float32),
        phi=np.array(phis, dtype=np.float32),
        f_max=float(p.f_max.max()),
        mask=p.mask,
    )
    if fine_window is not None:
        out["fine_t"] = np.array(fine_ts)
        out["fine_rate"] = np.array(fine_rates, dtype=np.float32)
    return out


def run_exp2(
    T_end: float = 100000.0,
    dt: float = 1.0,
    write_cycle: float = 1.0,
    n=(500, 1),
    Ic: float = 200.0,
    stim_t=(2.0, 5.0),
    stim_x=(0.1, 0.15),
    seed: int = 0,
    progress_every: float = 20000.0,
):
    """Port of ``Exp2.m`` panel A -- the 1D simulation behind Figure 2."""
    p = exp2_template(n)
    O = MeanFieldModel(p, seed=seed)
    O.add_standard_recurrent_connections()

    def det(x1, x2, t):
        if not (stim_t[1] * 1000 > t > stim_t[0] * 1000):
            return 0.0
        return ((n[0] * stim_x[1] > x1) & (x1 > n[0] * stim_x[0])) * Ic

    O.ext = [ExternalInput(deterministic=det, tmax=stim_t[1] * 1000)]

    ts, rates, Vs, Cls, gKs = [], [], [], [], []
    next_write = 0.0
    while O.t < T_end:
        if O.t >= next_write:
            ts.append(O.t)
            rates.append(O.R_f[:, 0].copy())
            Vs.append(O.V[:, 0].copy())
            Cls.append(O.Cl_in[:, 0].copy())
            gKs.append(O.g_K[:, 0].copy())
            next_write += write_cycle
        O.step(dt)
        if progress_every and O.t % progress_every < dt:
            print(f"t={O.t/1000:.0f}s  max_rate={O.R_f.max()/p.f_max.max():.3f}", flush=True)

    return dict(
        t=np.array(ts),
        rate=np.array(rates, dtype=np.float32),
        V=np.array(Vs, dtype=np.float32),
        Cl_in=np.array(Cls, dtype=np.float32),
        g_K=np.array(gKs, dtype=np.float32),
        f_max=float(p.f_max.max()),
    )


def run_exp6(
    T_end: float = 99999.0,
    dt: float = 1.0,
    write_cycle: float = 10.0,
    n=(50, 50),
    Ic: float = 200.0,
    stim_t=(2.0, 5.0),
    stim_x=(0.5, 0.5),
    stim_r: float = 0.05,
    noise_D: float = 0.0,
    noise_sigma: float = None,
    symmetrise: bool = None,
    seed: int = 0,
    progress_every: float = 20000.0,
    fine_window=None,
):
    """Port of ``Exp6.m`` -- the 2D round-field simulation behind Figure 6.

    Differs from ``run_exp1`` in three ways, all of them from ``Exp6.m``:

    * the stimulus sits at the *centre* of the disc (``stim_x = (0.5, 0.5)``),
      so inward waves converge symmetrically rather than against a boundary;
    * ``Exp6Template`` lowers ``E_L`` and ``V`` to -58 mV on a 50x50 grid;
    * background current noise is available, and in the noise-free case the
      synaptic input fields are explicitly symmetrised each step.

    Noise strength is specified as a diffusion coefficient ``noise_D`` in
    pA^2/ms, matching how the paper reports it; it is converted to the
    amplitude ``ExternalInput`` wants via :func:`sigma_from_diffusion`.  Pass
    ``noise_sigma`` instead to set the amplitude directly.

    ``symmetrise`` defaults to ``True`` exactly when the noise is off, which is
    the condition ``Exp6.m`` tests (``O.Ext(1).Random.sigma == 0``).  This
    combats FFT round-off so that the noise-free panel stays *exactly*
    fourfold-symmetric and the inward waves meet precisely at the centre --
    the qualitative claim of Panel A.
    """
    if noise_sigma is None:
        noise_sigma = sigma_from_diffusion(noise_D, dt) if noise_D else 0.0
    if symmetrise is None:
        symmetrise = noise_sigma == 0.0

    p = exp6_template(n)
    p.mask = _round_mask(n)
    O = MeanFieldModel(p, seed=seed)
    O.add_standard_recurrent_connections()

    # Exp6.m indexes with n(2)/n(1) transposed relative to the meshgrid, as
    # Exp1.m does; immaterial for the square grid used here.
    def det(x1, x2, t):
        if not (stim_t[1] * 1000 > t > stim_t[0] * 1000):
            return 0.0
        r = np.sqrt(
            ((x1 - 1) / (n[1] - 1) - stim_x[0]) ** 2
            + ((x2 - 1) / (n[0] - 1) - stim_x[1]) ** 2
        )
        return (r < stim_r) * Ic

    # The noise has no expiry; only the deterministic stimulus does.  Exp6.m
    # attaches both to a single ExternalInput with no Tmax, so the input is
    # evaluated every step -- keep tmax unset when noise is on.
    O.ext = [
        ExternalInput(
            deterministic=det,
            sigma=noise_sigma,
            tmax=None if noise_sigma else stim_t[1] * 1000,
        )
    ]

    ts, rates, Vs, Cls, gKs = [], [], [], [], []
    fine_ts, fine_rates = [], []
    next_write = 0.0
    while O.t < T_end:
        if O.t >= next_write:
            ts.append(O.t)
            rates.append(O.R_f.astype(np.float32).copy())
            Vs.append(O.V.astype(np.float32).copy())
            Cls.append(O.Cl_in.astype(np.float32).copy())
            gKs.append(O.g_K.astype(np.float32).copy())
            next_write += write_cycle
        # Rate at full dt over ``fine_window``.  The traveling-wave analysis
        # needs this: the direction disc is only 2.5 units across (radius 0.05
        # normalised), so at a 10 ms write cycle the activation times inside it
        # are almost all tied and the planar fit is undefined.
        if fine_window is not None and fine_window[0] <= O.t < fine_window[1]:
            fine_ts.append(O.t)
            fine_rates.append(O.R_f.astype(np.float32).copy())
        O.step(dt)
        if symmetrise:
            # Exp6.m: enforce fourfold symmetry on the synaptic input fields
            # to cancel FFT numerical error.  Applied AFTER Update, as there.
            for name in ("Input_E", "Input_I"):
                a = getattr(O, name)
                a = (a + np.flipud(a)) / 2
                a = (a + np.fliplr(a)) / 2
                setattr(O, name, a)
        if progress_every and O.t % progress_every < dt:
            print("  t = %.0f ms" % O.t, flush=True)

    return {
        "t": np.array(ts),
        "rate": np.array(rates),
        "V": np.array(Vs),
        "Cl_in": np.array(Cls),
        "g_K": np.array(gKs),
        "mask": p.mask,
        "n_units": n,
        "noise_D": noise_D,
        "noise_sigma": noise_sigma,
        "symmetrised": symmetrise,
        "f_max": float(np.mean(p.f_max)),
        "fine_t": np.array(fine_ts),
        "fine_rate": np.array(fine_rates) if fine_rates else None,
    }


def run_exp1_torus(
    T_end: float = 99990.0,
    dt: float = 1.0,
    write_cycle: float = 10.0,
    n=(50, 50),
    Ic: float = 200.0,
    stim_t=(2.0, 5.0),
    stim_x=(0.5, 0.5),
    stim_r: float = 0.05,
    sigma_E_frac: float = 0.02,
    sigma_I_frac: float = 0.03,
    seed: int = 0,
    progress_every: float = 20000.0,
    fine_window=None,
):
    """Figure-1C configuration on a **torus**: periodic boundaries, no mask.

    Differences from ``run_exp1`` (which reproduces the published figure):

    * ``topology='circular'`` in both convolution projections -- the
      ``'circular'`` branch of ``conv2_field.m``.  Recurrent drive wraps
      instead of being truncated at the edge.
    * **No round mask.**  Exp1.m's ``rr < 0.5`` mask exists to impose a
      boundary; a torus has none.  All ``prod(n)`` units are active, versus
      7860/10000 for the published 100x100 round field.
    * The stimulus disc uses the **periodic** (minimum-image) distance, so it
      does not get clipped when placed near an index edge.

    On a torus the network is translation invariant, so ``stim_x`` only sets
    where the seizure nucleates, not its subsequent behaviour; it defaults to
    the centre rather than Exp1.m's off-centre ``(0.5, 0.1)``.

    ``sigma_*_frac`` are fractions of field extent (Liou's convention), so at
    n=50 they give 1.0 and 1.5 pixel widths versus 2.0 and 3.0 at n=100 --
    the same physical sheet sampled half as finely.
    """
    p = exp1_template(n)
    p.mask = None                      # torus: no boundary, so no mask
    O = MeanFieldModel(p, seed=seed)
    O.add_standard_recurrent_connections(
        sigma_E_frac=sigma_E_frac, sigma_I_frac=sigma_I_frac, topology="circular"
    )

    def det(x1, x2, t):
        if not (stim_t[1] * 1000 > t > stim_t[0] * 1000):
            return 0.0
        # Minimum-image displacement in normalised units, both axes periodic.
        d1 = (x1 - 1) / n[1] - stim_x[0]
        d2 = (x2 - 1) / n[0] - stim_x[1]
        d1 = d1 - np.round(d1)
        d2 = d2 - np.round(d2)
        return (np.sqrt(d1**2 + d2**2) < stim_r) * Ic

    O.ext = [ExternalInput(deterministic=det, tmax=stim_t[1] * 1000)]

    ts, rates, Vs, Cls, gKs, phis = [], [], [], [], [], []
    fine_ts, fine_rates = [], []
    next_write = 0.0
    while O.t < T_end:
        if O.t >= next_write:
            ts.append(O.t); rates.append(O.R_f.copy()); Vs.append(O.V.copy())
            Cls.append(O.Cl_in.copy()); gKs.append(O.g_K.copy()); phis.append(O.phi.copy())
            next_write += write_cycle
        if fine_window is not None and fine_window[0] <= O.t < fine_window[1]:
            fine_ts.append(O.t); fine_rates.append(O.R_f.copy())
        O.step(dt)
        if progress_every and O.t % progress_every < dt:
            print(f"t={O.t/1000:.0f}s  max_rate={O.R_f.max()/p.f_max.max():.3f}", flush=True)

    out = dict(
        t=np.array(ts), rate=np.array(rates, dtype=np.float32),
        V=np.array(Vs, dtype=np.float32), Cl_in=np.array(Cls, dtype=np.float32),
        g_K=np.array(gKs, dtype=np.float32), phi=np.array(phis, dtype=np.float32),
        f_max=float(p.f_max.max()), mask=None, topology="circular", n=tuple(n),
    )
    if fine_window is not None:
        out["fine_t"] = np.array(fine_ts)
        out["fine_rate"] = np.array(fine_rates, dtype=np.float32)
    return out
