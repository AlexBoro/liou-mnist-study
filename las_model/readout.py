"""Driven-area seizure morphology readout for the Liou-MNIST sweeps.

This is the SAME detector every anchor value in LIOU_PILOT_SPEC.md was measured
with.  It previously existed only as kernel-local defs, which meant the numbers
in the spec could not be reproduced after a kernel reset -- it is a module now
so the pilot and the eventual 500-run sweep score identically.

Criterion (user, 2026-08-13): a seizure is called from the MORPHOLOGY and
EVOLUTION of the LFP waveforms WITHIN the stimulated area.  Spread beyond the
driven area and activity outlasting the stimulus are neither necessary nor
sufficient -- they are confound exclusions, never positive evidence.
"""
import numpy as np
from scipy import signal as sg

from . import lfp as LFP

# 30-electrode montage: flat indices at stride 233.  The stride was chosen on an
# 84x84 sheet, but callers WRAP it (``EL30 % (n*n)``) for whatever sheet they
# use, so the aliasing property must be re-checked per n -- it is NOT inherited.
# Verified at n=56 (the pilot sheet): gcd(233, 56*56)=1 so all 30 wrap to
# distinct units, spanning 30 distinct columns but only 27 distinct ROWS (three
# row collisions), min toroidal separation 5.10 units.  At n=84 (unwrapped):
# 30 distinct rows and columns, min separation 12.7.
STRIDE = 233
EL30 = np.arange(30) * STRIDE

# Reference values for the canonical focal-probe seizure (scale 0, Ic 200),
# near-focus channels -- the positive control the pilot is judged against.
REFERENCE_SEIZURE = {
    (45, 60): dict(n=291, amp=42627, fwhm=65.0, ipi=275.0, cv=0.22, R250=0.132),
    (60, 80): dict(n=609, amp=47281, fwhm=70.0, ipi=365.0, cv=0.30, R250=0.055),
}
IMAGE_PERIOD_S = 0.250


def fwhm(x, p):
    """Full width at half max of the peak at index p, in samples."""
    h = x[p] / 2.0
    i = p
    while i > 0 and x[i] > h:
        i -= 1
    j = p
    while j < len(x) - 1 and x[j] > h:
        j += 1
    return float(j - i)


def morph(Lm, tm, chs, a, b, h=2.0, per_scan=True, dt=1.0):
    """Median discharge morphology over channels `chs` in window [a, b) seconds.

    Peaks are found per channel at h sigma above that channel's own mean with a
    120 ms refractory distance; a channel contributing fewer than 3 peaks is
    dropped so single-transient artifacts cannot set the median.
    """
    A, W, IPI, PT = [], [], [], []
    ds = float(np.median(np.diff(tm)))
    for i in chs:
        # EXCLUDE non-finite samples rather than rejecting the channel.  lfp_2d
        # NaNs the first dT samples by construction (the axonal delay), so a
        # window starting at t=0 contains exactly one NaN -- and an
        # ``isfinite(x).all()`` guard silently dropped EVERY channel, making
        # every early-window metric NaN with no error raised.
        m = (tm >= a) & (tm < b) & np.isfinite(Lm[:, i])
        if m.sum() < 3:
            continue
        x = Lm[m, i] - np.nanmean(Lm[m, i])
        t_ = tm[m]
        s = np.std(x)
        if not np.isfinite(s) or s == 0:
            continue
        pk, _ = sg.find_peaks(x / s, height=h, distance=int(0.12 / ds))
        if len(pk) < 3:
            continue
        A += list(x[pk])
        W += [fwhm(x, p) * ds / dt for p in pk]
        IPI += list(np.diff(t_[pk]))
        PT += list(t_[pk])
    PT = np.array(PT)
    best = (np.nan, np.nan)
    if per_scan and len(PT) > 10:
        Ps = np.linspace(0.18, 0.40, 441)
        Rs = [np.abs(np.mean(np.exp(1j * 2 * np.pi * (PT % Pv) / Pv))) for Pv in Ps]
        best = (Ps[int(np.argmax(Rs))], max(Rs))
    if not A:
        return dict(n=0, amp=np.nan, fwhm=np.nan, ipi=np.nan, cv=np.nan,
                    P_best=np.nan, R_best=np.nan, R250=np.nan)
    return dict(n=len(A), amp=float(np.median(A)), fwhm=float(np.median(W)) * 1e3,
                ipi=float(np.median(IPI)) * 1e3,
                cv=float(np.std(IPI) / np.mean(IPI)) if len(IPI) else np.nan,
                P_best=best[0], R_best=best[1],
                R250=float(np.abs(np.mean(np.exp(1j * 2 * np.pi * (PT % IMAGE_PERIOD_S) / IMAGE_PERIOD_S)))))


def driven_channels(n=56, band=None):
    """Indices into the 30-electrode montage that fall inside the driven area."""
    el = np.asarray(EL30) % (n * n)
    r_, c_ = el // n, el % n
    bd = band if band is not None else (int(round(n * (56 / 84.))) if n != 84 else 56)
    lo, hi = (n - bd) // 2, (n - bd) // 2 + bd
    dmask = (r_ >= lo) & (r_ < hi) & (c_ >= lo) & (c_ < hi)
    return np.where(dmask)[0], dmask


def lfp_montage(z, n=56, sigma=None):
    """(T, 30) LFP montage + time axis in seconds from a run dict.

    ``sigma`` is the LFP spatial constant in NORMALISED units (a fraction of
    sheet extent).  ``None`` defers to :func:`lfp.lfp_2d`'s own default of
    0.025, which is what every run scored before 2026-08-16 used -- so the
    default path stays byte-identical and the 54-run pilot and 12 full-field
    re-runs remain reproducible.

    THIS IS NOT A GAIN KNOB.  The kernel is unnormalised, so widening the
    electrode both scales the output (per-electrode weight sum 12.5 at 0.025
    vs 228.8 at 0.11) AND changes which events clear the h-sigma peak
    criterion in :func:`morph` -- measured -29% discharges on one pilot run
    and +41% on another for the same 0.025 -> 0.11 change.  Any absolute
    amplitude threshold is therefore specific to the sigma it was fit at.

    ``dt_record`` is DERIVED from the run's own time axis and is in
    MILLISECONDS.  It must never be hardcoded: lfp_2d uses it twice, and both
    uses fail silently in the same direction if the unit is wrong --

      dT   = round(delay_ms / dt_record)      -> axonal delay, and
      fs   = 1000 / dt_record                 -> high-pass sample rate.

    Passing 0.005 (seconds) instead of 5.0 (ms) gave a 6 SECOND delay instead
    of 6 ms and told the filter the data were sampled at 200 kHz instead of
    200 Hz, so the 1 Hz high-pass did essentially nothing.  Neither error
    raises; they shifted median amplitude ~23% and discharge count ~37%.
    """
    t_ms = np.asarray(z["t"], dtype=float)
    dtr = float(np.median(np.diff(t_ms)))          # milliseconds
    if not (0.05 <= dtr <= 100.0):
        raise ValueError(
            "dt_record=%.4g ms is outside the plausible range: the time axis is "
            "expected in MILLISECONDS. A value near 0.005 means seconds were "
            "passed." % dtr)
    skw = {} if sigma is None else dict(sigma=float(sigma))
    L, _dT = LFP.lfp_2d(z["EPSC"], z["IPSC"],
                        EL30 % (n * n) if n != 84 else EL30, (n, n),
                        dt_record=dtr, filter_hz=1.0, **skw)
    return L, t_ms / 1000.0


def morph_driven(z, n=56, band=None, a=None, b=None):
    """Driven-area discharge morphology from a run dict.  (metrics, n_channels)."""
    L, t_ = lfp_montage(z, n=n)
    chs, _ = driven_channels(n=n, band=band)
    if len(chs) == 0:
        chs = np.arange(L.shape[1])
    a = a if a is not None else 0.55 * t_[-1]
    b = b if b is not None else t_[-1]
    return morph(L, t_, chs, a, b, per_scan=False), len(chs)
