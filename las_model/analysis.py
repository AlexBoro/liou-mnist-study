"""Measurement of the two propagation speeds.

The paper reports both in *normalised space units per second*: the network
extent along a dimension is 1.0 regardless of how many units discretise it.
"""

from __future__ import annotations

import numpy as np


def normalised_rate(rate: np.ndarray, f_max: float) -> np.ndarray:
    return rate / f_max


def wavefront_speed_1d(t_ms, r_norm, threshold=0.1, t_min_ms=6000.0):
    """Speed of the recruitment front: how fast the seizure territory grows.

    For each time point the front is the furthest position (in units of index)
    from the stimulus site that has ever been recruited, i.e. has exceeded
    ``threshold`` of f_max.  Speed is the least-squares slope of that
    trajectory, converted to normalised units per second.
    """
    n = r_norm.shape[1]
    recruited = np.cumsum(r_norm > threshold, axis=0) > 0
    idx = np.where(recruited.any(axis=1), recruited.argmax(axis=1) * 0, 0)
    # Leading edge = largest recruited index (the front advances upward in x)
    front = np.array(
        [np.where(row)[0].max() if row.any() else np.nan for row in recruited], dtype=float
    )
    sel = (t_ms >= t_min_ms) & np.isfinite(front)
    # Restrict to the interval where the front is still inside the field
    sel &= front < (n - 2)
    if sel.sum() < 10:
        return np.nan, front
    slope = np.polyfit(t_ms[sel] / 1000.0, front[sel] / n, 1)[0]
    return slope, front


def traveling_wave_speed_1d(
    t_ms, r_norm, x_lo, x_hi, t_lo_ms, t_hi_ms, threshold=0.3, min_gap_ms=50.0
):
    """Speed of the fast inward discharges, by regression on peak timings.

    Within the space-time window, the firing-rate peak time is found at each
    position; a straight line fitted to (position, peak time) gives the inverse
    speed.  Positions are converted to normalised units before fitting.
    """
    n = r_norm.shape[1]
    tsel = (t_ms >= t_lo_ms) & (t_ms <= t_hi_ms)
    sub = r_norm[tsel][:, x_lo:x_hi]
    tt = t_ms[tsel]
    peak_t, xs = [], []
    for j in range(sub.shape[1]):
        col = sub[:, j]
        if col.max() < threshold:
            continue
        peak_t.append(tt[col.argmax()])
        xs.append((x_lo + j) / n)
    if len(xs) < 5:
        return np.nan, np.array(xs), np.array(peak_t)
    xs = np.array(xs)
    peak_t = np.array(peak_t) / 1000.0
    inv_speed = np.polyfit(xs, peak_t, 1)[0]  # seconds per normalised unit
    return 1.0 / inv_speed, xs, peak_t * 1000.0


def territory_radius_2d(rate, mask, f_max, centre_rc, threshold=0.1):
    """Radius of the recruited territory over time (normalised units).

    ``centre_rc`` is the (row, col) of the stimulus site.  Radius is taken as
    the 95th percentile of the distance of ever-recruited units from the
    stimulus, which is robust to isolated units.
    """
    T, n0, n1 = rate.shape
    rr, cc = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")
    dist = np.sqrt(((rr - centre_rc[0]) / n0) ** 2 + ((cc - centre_rc[1]) / n1) ** 2)
    ever = np.zeros((n0, n1), dtype=bool)
    out = np.zeros(T)
    for i in range(T):
        ever |= (rate[i] / f_max) > threshold
        out[i] = np.percentile(dist[ever], 95) if ever.any() else 0.0
    return out


def linear_speed(t_s, radius, t_lo, t_hi):
    sel = (t_s >= t_lo) & (t_s <= t_hi)
    if sel.sum() < 5:
        return np.nan
    return float(np.polyfit(t_s[sel], radius[sel], 1)[0])
