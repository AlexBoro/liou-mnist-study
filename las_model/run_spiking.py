"""Experiment driver -- port of ``Exp4.m`` (Figure 4A/4B).

Panel A: LVFA onset, standard inhibition (g_I_local 250, g_I_global 50).
Panel B: rhythmic onset -- ``Exp4.m`` sets BOTH inhibitory projection
strengths to 150, which redistributes inhibition toward the non-local
component (the paper's gamma = 1/2 rather than 1/6).
"""

from __future__ import annotations

import numpy as np

from .spiking import SpikingModel, exp4_template


def _focal_stimulus(Ic: float, stim_x, stim_t, n):
    """``O.Ext.Deterministic`` from ``Exp4.m``.

    MATLAB positions are 1-based unit indices; the comparison is strict on both
    sides, so the stimulated band is ``0.05*n < x < 0.1*n``.
    """
    x = np.arange(1, n[0] + 1).reshape(n)
    in_x = (x > stim_x[0] * n[0]) & (x < stim_x[1] * n[0])

    def ext(t, dt, shape):
        in_t = (t > stim_t[0] * 1000) and (t < stim_t[1] * 1000)
        return in_x * Ic if in_t else np.zeros(shape)

    return ext


def run_exp4(
    panel: str = "A",
    T_end: float = 50000.0,
    dt: float = 1.0,
    write_cycle: float = 1.0,
    seed: int = 0,
    progress_every: int = 0,
):
    """Run Exp4 and return recorded fields.

    Returns spike raster (as a firing-rate proxy), EPSC, IPSC and the state
    variables, all ``(n_time, n_units)`` except as noted.
    """
    p = exp4_template(panel)
    O = SpikingModel(p, seed=seed)

    if panel.upper() == "B":
        O.add_standard_recurrent_connections(g_I_local=150.0, g_I_global=150.0)
    else:
        O.add_standard_recurrent_connections()

    O.ext = [_focal_stimulus(80.0, (0.05, 0.10), (2.0, 5.0), O.n)]

    n_steps = int(round(T_end / dt))
    stride = max(1, int(round(write_cycle / dt)))
    n_rec = n_steps // stride + 1
    nu = O.n[0]

    spikes = np.zeros((n_rec, nu), dtype=bool)
    epsc = np.zeros((n_rec, nu), dtype=np.float32)
    ipsc = np.zeros((n_rec, nu), dtype=np.float32)
    Vs = np.zeros((n_rec, nu), dtype=np.float32)
    Cls = np.zeros((n_rec, nu), dtype=np.float32)
    gKs = np.zeros((n_rec, nu), dtype=np.float32)
    ts = np.zeros(n_rec, dtype=np.float32)

    k = 0
    for i in range(n_steps):
        if i % stride == 0 and k < n_rec:
            # Recorder writes BEFORE Update, as in the MATLAB while-loop
            e, ii = O.psc()
            ts[k] = O.t
            spikes[k] = O.S.ravel()
            epsc[k] = e.ravel()
            ipsc[k] = ii.ravel()
            Vs[k] = O.V.ravel()
            Cls[k] = O.Cl_in.ravel()
            gKs[k] = O.g_K.ravel()
            k += 1
        O.step(dt)
        if progress_every and i % progress_every == 0:
            print(f"  t={O.t:.0f} ms  spikes={int(O.S.sum())}", flush=True)

    return dict(
        t=ts[:k], spikes=spikes[:k], EPSC=epsc[:k], IPSC=ipsc[:k],
        V=Vs[:k], Cl_in=Cls[:k], g_K=gKs[:k], panel=panel, dt=dt,
        write_cycle=write_cycle, n_units=nu,
    )
