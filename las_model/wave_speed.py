"""Local traveling-wave speed estimation for Figure 4B (lower subpanel).

NOT a port: the LAS-Model repository ships no such routine.  ``ChasePeak.m``
walks a 2-D local maximum and ``MeasureSeizureActivity.m`` finds phase
transitions; neither estimates per-discharge propagation speed.  This module
implements the method described in the Figure 4 legend and in the paper's
Materials and Methods (the same multiunit spike-time regression used for the
patient data in panel C):

    For each discharge -- identified as a local maximum of the firing rate at
    the readout location -- regress the local peak spike time of each nearby
    unit on that unit's position.  The slope gives 1/speed; its sign gives the
    direction of propagation.  Significance is the F-test on the regression,
    with the legend's threshold of 0.001.

Because the legend fixes only the estimator's structure and not its window
sizes, ``t_half`` and ``x_half`` are explicit arguments; the defaults are set
so that the spatial window is comparable to the recurrent excitation footprint
(sigma_E = 2% of the field) and the temporal window comfortably contains one
discharge.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d


def local_rate(spikes: np.ndarray, loc: int, x_half: int = 20,
               smooth_ms: float = 10.0) -> np.ndarray:
    """Firing-rate proxy at ``loc``: spike count in a spatial window, smoothed."""
    lo, hi = max(0, loc - x_half), min(spikes.shape[1], loc + x_half + 1)
    r = spikes[:, lo:hi].sum(axis=1).astype(float)
    return uniform_filter1d(r, int(smooth_ms), mode="nearest")


def find_discharges(rate: np.ndarray, min_sep_ms: float = 30.0,
                    prominence_frac: float = 0.2) -> np.ndarray:
    from scipy.signal import find_peaks
    if rate.max() <= 0:
        return np.array([], dtype=int)
    pk, _ = find_peaks(rate, distance=min_sep_ms,
                       prominence=prominence_frac * rate.max())
    return pk


def speed_at_discharge(spikes: np.ndarray, t0: int, loc: int,
                       t_half: int = 15, x_half: int = 60,
                       min_units: int = 8, n_units_total: int | None = None):
    """Regress local peak spike time on position around one discharge.

    Returns ``(speed, p_value, n_used)`` with speed in normalised space units
    per second, positive = propagating away from the focus (outward, toward
    increasing index), negative = inward.  ``(nan, nan, n)`` if the window has
    too few active units.

    **Window width is critical.**  Each unit contributes the time of the spike
    *closest to* ``t0``, not the centre of mass of every spike in the window.
    During the clonic stage a unit fires at 70-90 Hz, so a window wider than
    half the inter-discharge interval mixes successive discharges and the
    regression aliases across waves -- the same failure mode documented for the
    mean-field model in ``PORT_NOTES.md``.  ``measure_wave_speeds`` sets
    ``t_half`` from the measured inter-discharge interval by default.
    """
    T, X = spikes.shape
    if n_units_total is None:
        n_units_total = X
    t_lo, t_hi = max(0, t0 - t_half), min(T, t0 + t_half + 1)
    x_lo, x_hi = max(0, loc - x_half), min(X, loc + x_half + 1)
    win = spikes[t_lo:t_hi, x_lo:x_hi]

    pos, tim = [], []
    for j in range(win.shape[1]):
        idx = np.where(win[:, j])[0]
        if idx.size:
            # spike closest to the discharge centre -- one wave only
            pos.append(x_lo + j)
            tim.append(t_lo + idx[np.argmin(np.abs(idx - (t0 - t_lo)))])
    if len(pos) < min_units:
        return np.nan, np.nan, len(pos)

    pos = np.asarray(pos, float)
    tim = np.asarray(tim, float)
    if np.ptp(pos) == 0 or np.ptp(tim) == 0:
        return np.nan, np.nan, len(pos)

    res = stats.linregress(pos, tim)          # tim = a + b*pos, b in ms/unit
    if res.slope == 0:
        return np.nan, np.nan, len(pos)

    # F-test on the regression == two-sided t-test on the slope for simple OLS
    p = res.pvalue
    speed_units_per_ms = 1.0 / res.slope
    speed = speed_units_per_ms / n_units_total * 1000.0   # normalised units / s
    return speed, p, len(pos)


def measure_wave_speeds(spikes: np.ndarray, loc: int, t_ms: np.ndarray,
                        t_half: int | None = None, x_half: int = 60,
                        rate_x_half: int = 20, **kw):
    """All discharges at ``loc``: returns a dict of arrays.

    ``t_half=None`` sets the window to 40% of the median inter-discharge
    interval, which keeps each regression inside a single wave.
    """
    rate = local_rate(spikes, loc, x_half=rate_x_half)
    peaks = find_discharges(rate, **kw)
    if t_half is None:
        if len(peaks) > 1:
            t_half = max(4, int(0.4 * np.median(np.diff(peaks))))
        else:
            t_half = 15
    out = {"t": [], "speed": [], "p": [], "n": [], "t_half": t_half}
    for t0 in peaks:
        s, p, n = speed_at_discharge(spikes, int(t0), loc,
                                     t_half=t_half, x_half=x_half,
                                     n_units_total=spikes.shape[1])
        if np.isfinite(s):
            out["t"].append(t_ms[int(t0)] / 1000.0)
            out["speed"].append(s)
            out["p"].append(p)
            out["n"].append(n)
    return {k: np.asarray(v) for k, v in out.items()}, rate
