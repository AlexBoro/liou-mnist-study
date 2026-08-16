"""Two-dimensional traveling-wave analysis for Figure 6.

``chase_peak`` is a port of ``ChasePeak.m``.  Everything else is
**reconstructed** from the Figure 6 legend: the repository ships no code for
the confluence-point statistics of Panel C or the wave-direction estimate of
Panel D.
"""

from __future__ import annotations

import numpy as np


def chase_peak(T: np.ndarray, X, dT_max: float = 10.0, dX_max: int = 1):
    """Port of ``ChasePeak.m`` -- walk uphill in activation time.

    ``T`` is a map of local activation times and ``X`` a 0-based ``(i, j)``
    start position.  At each iteration the steepest *later*-activating
    neighbour within ``+/- dX_max`` is taken, rejecting steps that jump more
    than ``dT_max`` ahead (which would cross into the next discharge), until no
    neighbour is later.  The terminus is where inward waves meet.

    ``ChasePeak.m`` pads by ``dXmax`` with NaN and relies on MATLAB's ``max``
    ignoring NaN; ``np.nanmax`` is used here for the same effect.  Ties resolve
    to the first index in both languages, but the traversal order differs
    (MATLAB is column-major), so a tie can land on a different neighbour.  Ties
    require exactly equal activation times, which noise makes negligible.
    """
    pad = dX_max
    Tp = np.full((T.shape[0] + 2 * pad, T.shape[1] + 2 * pad), np.nan)
    Tp[pad:-pad, pad:-pad] = T
    x = [int(X[0]) + pad, int(X[1]) + pad]
    trail = []
    for _ in range(T.size):
        trail.append((x[0] - pad, x[1] - pad))
        if not np.isfinite(Tp[x[0], x[1]]):
            return (x[0] - pad, x[1] - pad), trail
        loc = Tp[x[0] - pad : x[0] + pad + 1, x[1] - pad : x[1] + pad + 1] - Tp[x[0], x[1]]
        loc = np.where(loc > dT_max, np.nan, loc)
        if np.all(np.isnan(loc)):
            break
        k = int(np.nanargmax(loc))
        i, j = np.unravel_index(k, loc.shape)
        di, dj = i - pad, j - pad
        if di == 0 and dj == 0:
            break
        x = [x[0] + di, x[1] + dj]
    return (x[0] - pad, x[1] - pad), trail


def discharge_times(rate, mask, f_max, t, height_frac=0.25, min_sep_ms=40.0):
    """Times of population discharges, from the mean rate inside the territory.

    A discharge is a peak in the spatial mean firing rate over active tissue.
    ``height_frac`` is expressed as a fraction of the *observed* peak of that
    mean signal rather than of ``f_max``, because the population mean never
    approaches the single-unit ceiling.
    """
    from scipy.signal import find_peaks

    flat = rate.reshape(len(rate), -1)[:, mask.ravel()]
    m = flat.mean(axis=1)
    dt_frame = float(t[1] - t[0])
    pk, _ = find_peaks(
        m, height=height_frac * m.max(), distance=max(1, int(min_sep_ms / dt_frame))
    )
    return pk, m


def activation_map(rate, k_peak, half_window_frames, mask):
    """Per-unit activation time (frame index) within one discharge.

    For each unit, the frame of its maximum rate inside the window.  Units that
    never exceed a tenth of their own in-window range, and units outside the
    mask, are returned as NaN -- they carry no wave.
    """
    a = max(0, k_peak - half_window_frames)
    b = min(rate.shape[0], k_peak + half_window_frames + 1)
    w = rate[a:b]
    T = np.argmax(w, axis=0).astype(float)
    amp = w.max(axis=0) - w.min(axis=0)
    T[(amp <= 0.1 * np.nanmax(amp)) | (mask <= 0)] = np.nan
    return T + a


def wave_direction(T, centre, radius_units, min_r2: float = 0.25):
    """Planar fit of activation time in a disc: returns (angle, speed, r2, n).

    A direction is reported only if the plane explains at least ``min_r2`` of
    the variance in activation time.  This is the load-bearing guard, not the
    gradient magnitude: in the noise-free field the activation map is radially
    symmetric, so a plane fits *exactly zero* variance (r2 = 0.0) while the
    fitted gradient is float dust of order 1e-14 -- which inverts to an
    apparent speed of 1e13.  Radial convergence is not a traveling wave with a
    direction, and the r2 test is what distinguishes the two.

    The wave travels from early- to late-activating tissue, so the propagation
    direction is that of ``+grad T`` and the speed is ``1/|grad T|``.  Angle is
    measured in the (column, row) = (x, y) frame, counter-clockwise from +x.
    """
    n0, n1 = T.shape
    ii, jj = np.mgrid[0:n0, 0:n1]
    d = np.sqrt((ii - centre[0]) ** 2 + (jj - centre[1]) ** 2)
    sel = (d <= radius_units) & np.isfinite(T)
    if sel.sum() < 4:
        return np.nan, np.nan, np.nan, int(sel.sum())
    A = np.column_stack([jj[sel] - centre[1], ii[sel] - centre[0], np.ones(sel.sum())])
    coef, *_ = np.linalg.lstsq(A, T[sel], rcond=None)
    gx, gy = coef[0], coef[1]
    pred = A @ coef
    obs = T[sel]
    ss = ((obs - obs.mean()) ** 2).sum()
    r2 = 1.0 - ((obs - pred) ** 2).sum() / ss if ss > 0 else np.nan
    g = np.hypot(gx, gy)
    # A perfectly symmetric (noise-free) field activates the whole centre disc
    # on the same frame: the gradient vanishes and direction is undefined
    # rather than arbitrary.  This is the Panel A claim, so report it as NaN
    # instead of letting 1/g overflow into a spurious huge speed.
    if not np.isfinite(r2) or g <= 0 or np.ptp(obs) == 0 or r2 < min_r2:
        return np.nan, np.nan, (float(r2) if np.isfinite(r2) else np.nan), int(sel.sum())
    return float(np.arctan2(gy, gx)), float(1.0 / g), float(r2), int(sel.sum())


def circ_corr(a, b):
    """Jammalamadaka-SenGupta circular correlation coefficient."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return np.nan
    ma = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    mb = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a - ma), np.sin(b - mb)
    den = np.sqrt((sa**2).sum() * (sb**2).sum())
    return float((sa * sb).sum() / den) if den > 0 else np.nan


def circ_autocorr(theta, lag=1, n_perm=2000, seed=0):
    """Circular autocorrelation at ``lag`` with a permutation p-value."""
    th = np.asarray(theta, float)
    th = th[np.isfinite(th)]
    if len(th) < lag + 3:
        return np.nan, np.nan, len(th)
    rho = circ_corr(th[:-lag], th[lag:])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.permutation(th)
        null[i] = circ_corr(s[:-lag], s[lag:])
    p = (np.abs(null) >= abs(rho)).mean()
    return float(rho), float(max(p, 1.0 / n_perm)), len(th)
