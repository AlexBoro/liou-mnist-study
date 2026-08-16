"""LFP proxy from recorded synaptic currents.

Port of ``@Recorder/LFP.m`` -- Mazzoni's method (PLoS Comput Biol 2015):
the LFP is the EPSC delayed by ``dT`` samples minus ``alpha`` times the IPSC,
then smoothed across space by an exponential kernel.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal


def lfp_kernel(n_units: int, sigma: float = 0.025, literal: bool = False) -> np.ndarray:
    """Two-sided exponential spatial kernel.

    ``LFP.m`` builds it as::

        kernel = exp(0:-1:-N / N / p.Results.sigma);
        kernel = [kernel(end:-1:2), kernel];

    **Ambiguity, deliberately surfaced.**  MATLAB's ``:`` has lower precedence
    than ``/``, so the endpoint is ``-((N/N)/sigma) = -1/sigma``: the network
    size ``N`` cancels, the step is ``-1`` per neuron, and the kernel decays by
    a factor e for every neuron of distance -- an e-folding length of ONE
    neuron, truncated at ``1/sigma`` neurons.  With ``sigma=0.025`` that is a
    2*40+1 = 81-tap kernel whose weights fall below 1e-17 by tap 40, so in
    practice only the nearest 1-2 neurons contribute.

    **RESOLVED BY THE PAPER.**  Methods, "Local field potential (LFP)-like
    signal readout in the integrate-and-fire spiking model":

        "Synaptic current contributed by neighboring cells decays
        exponentially with spatial constant sigma_LFP = 0.025 (unit:
        normalized spatial scale)."

    ``sigma_LFP`` is quoted in the *same* normalized spatial units as
    ``sigma_E`` (0.02) and ``sigma_I`` (0.03) in Table 2 -- i.e. as a fraction
    of the sheet, not a per-neuron decay.  For N = 2000 that is an e-folding
    length of ``0.025 * 2000 = 50`` neurons, which is exactly what
    ``literal=False`` produces (measured e-folding: 51 taps).  The literal
    MATLAB precedence reading gives 2 neurons and is therefore a **bug in the
    released code**, not a modelling choice.

    ``literal=False`` is the default.  ``literal=True`` is retained only to
    reproduce what the released MATLAB actually computes.
    """
    if literal:
        # exp(0 : -1 : -1/sigma)
        endpoint = -(n_units / n_units) / sigma
        exponents = np.arange(0, endpoint - 1e-12, -1.0)
    else:
        length_units = sigma * n_units
        endpoint = -n_units / length_units
        exponents = np.arange(0, endpoint - 1e-12, -1.0 / length_units)
    k = np.exp(exponents)
    return np.concatenate([k[:0:-1], k])


def lfp_at(
    epsc: np.ndarray,
    ipsc: np.ndarray,
    loc: int,
    alpha: float = 1.65,
    dT: int = 6,
    sigma: float = 0.025,
    literal_kernel: bool = False,
) -> np.ndarray:
    """LFP proxy at a SINGLE location, from ``(n_time, n_units)`` currents.

    Equivalent to ``compute_lfp(...)[loc]`` but only touches the kernel
    footprint around ``loc``, so it is cheap for a couple of readout sites on
    a long run.  Verified against :func:`compute_lfp` to machine precision at
    interior and near-boundary locations.

    The weight applied to network position ``p`` is ``k[centre + (p - loc)]``.
    Do **not** reverse the sliced kernel: reversal is a no-op only when the
    window is truncated symmetrically (``loc`` at the array centre), and near a
    boundary it silently mirrors the readout to the opposite side of the sheet.
    """
    k = lfp_kernel(epsc.shape[1], sigma=sigma, literal=literal_kernel)
    n_units = epsc.shape[1]
    half = (len(k) - 1) // 2
    lo, hi = loc - half, loc + half + 1
    klo = max(0, -lo)
    khi = len(k) - max(0, hi - n_units)
    lo, hi = max(0, lo), min(n_units, hi)
    L = np.roll(epsc[:, lo:hi].astype(np.float64), dT, axis=0)
    L = L - alpha * ipsc[:, lo:hi].astype(np.float64)
    L[:dT] = np.nan
    out = np.nansum(L * k[klo:khi], axis=1)
    out[:dT] = np.nan
    return out


def compute_lfp(
    epsc: np.ndarray,
    ipsc: np.ndarray,
    alpha: float = 1.65,
    dT: int = 6,
    sigma: float = 0.025,
    literal_kernel: bool = False,
) -> np.ndarray:
    """LFP proxy for a 1D model.

    ``epsc``/``ipsc`` are ``(n_units, n_time)``.  ``dT`` is the EPSC delay in
    samples (6 at dt = 1 ms).  The first ``dT`` columns are NaN, matching
    ``L(:,1:dT) = nan`` in the MATLAB; ``circshift`` wraps, so those columns
    would otherwise hold end-of-run data.
    """
    L = np.roll(epsc, dT, axis=1)
    L = L - alpha * ipsc
    L[:, :dT] = np.nan

    kernel = lfp_kernel(epsc.shape[0], sigma=sigma, literal=literal_kernel)
    nanmask = np.isnan(L)
    L = np.where(nanmask, 0.0, L)
    L = signal.convolve2d(L, kernel[:, None], mode="same", boundary="fill", fillvalue=0.0)
    L[nanmask] = np.nan
    return L


def lfp_kernel_2d(shape, sigma: float = 0.025, isotropic: bool = True,
                  periodic: bool = True):
    """Per-unit weights for every electrode position on a 2-D sheet.

    Returns a callable ``w(loc_rc) -> (n_units,)`` giving the weight of each
    flattened unit for an electrode at ``loc_rc = (row, col)``.

    **Spatial constant.**  ``sigma_LFP`` is quoted by Liou in NORMALISED units
    -- a fraction of field extent -- exactly as ``sigma_E`` (0.02) and
    ``sigma_I`` (0.03) are.  So on an ``n x n`` sheet the e-folding length is
    ``sigma * n`` units: 1.25 units at n=50, 2.5 at n=100.  This is the same
    convention ``add_standard_recurrent_connections`` uses, so the LFP kernel
    and the connectivity kernels scale together when the sheet is resampled.

    **Consequence worth stating.**  Liou's own LFP figures come from a 1-D
    N=2000 line, where ``sigma*N = 50`` units and roughly 5% of the network
    lies within one e-folding.  On a 50x50 sheet the same NORMALISED sigma
    gives 1.25 units, and because area grows as r^2 only ~0.2% of the network
    lies within one e-folding.  Holding sigma normalised (faithful) therefore
    makes the 2-D electrode far more LOCAL, in network-fraction terms, than
    the 1-D one.  That is a property of the geometry, not a porting choice;
    the alternative -- inflating sigma to preserve the sampled fraction --
    would break the shared convention with sigma_E/sigma_I and is not done.

    **isotropic** (default) weights by Euclidean distance, ``exp(-r/L)``,
    which is the direct reading of "decays exponentially with spatial
    constant".  ``isotropic=False`` gives the separable form
    ``exp(-|dr|/L)exp(-|dc|/L)``.  For a GAUSSIAN these coincide (which is why
    the connectivity kernels can be built separably); for an EXPONENTIAL they
    do not, so the choice is explicit rather than inherited.

    **periodic** uses minimum-image distance, matching ``topology='circular'``.
    """
    nr, nc = shape
    L = sigma * max(nr, nc)
    rr = np.arange(nr)[:, None]
    cc = np.arange(nc)[None, :]

    def w(loc_rc):
        dr = np.abs(rr - loc_rc[0]).astype(float)
        dc = np.abs(cc - loc_rc[1]).astype(float)
        if periodic:
            dr = np.minimum(dr, nr - dr)
            dc = np.minimum(dc, nc - dc)
        if isotropic:
            k = np.exp(-np.hypot(dr, dc) / L)
        else:
            k = np.exp(-dr / L) * np.exp(-dc / L)
        return k.ravel()

    return w


def lfp_2d(epsc, ipsc, electrodes, shape, dt_record: float, alpha: float = 1.65,
           delay_ms: float = 6.0, sigma: float = 0.025, isotropic: bool = True,
           periodic: bool = True, weight_floor: float = 1e-8,
           filter_hz: float = None, filter_order: int = 4, report_ipsc_sign: bool = True):
    """LFP proxy at 2-D electrode sites, from ``(n_time, n_units)`` currents.

    Same Mazzoni construction as :func:`compute_lfp`, generalised off the line:

        Lambda(x,t) = I_E(x, t - delay) - alpha * I_I(x,t)
        LFP(x0, t)  = sum_x  w(|x - x0|) Lambda(x,t)

    ``electrodes`` are FLAT row-major indices into the ``shape`` sheet.

    **The delay is given in MILLISECONDS, not samples.**  Liou's ``LFP.m``
    hardcodes ``dT = 6`` because his recorder samples at dt = 1 ms, so the
    two coincide there and the distinction is invisible in the MATLAB.  It is
    not invisible here: recording every 2 ms and passing 6 would impose a
    12 ms delay.  The conversion is done once, below, and the number of
    samples is rounded.

    The kernel is UNNORMALISED (Liou's is too), so the output is a weighted
    sum in arbitrary units; only relative amplitude and timing are
    interpretable.  The first ``delay`` samples are NaN, matching
    ``L(:,1:dT) = nan``.
    """
    epsc = np.asarray(epsc); ipsc = np.asarray(ipsc)
    dT = int(round(delay_ms / dt_record))
    lam = np.roll(epsc.astype(np.float64), dT, axis=0) - alpha * ipsc.astype(np.float64)
    lam[:dT] = np.nan

    wfun = lfp_kernel_2d(shape, sigma=sigma, isotropic=isotropic, periodic=periodic)
    nc = shape[1]
    out = np.empty((lam.shape[0], len(electrodes)), dtype=np.float64)
    for j, u in enumerate(electrodes):
        wts = wfun((int(u) // nc, int(u) % nc))
        keep = wts > weight_floor
        out[:, j] = lam[:, keep] @ wts[keep]

    if report_ipsc_sign:
        rep = ipsc_sign_report(ipsc)
        if rep["frac_positive"] > 0:
            warnings.warn(
                "IPSC is positive in %.2f%% of samples (peak %.1f%% of units at one "
                "time): the -alpha*I_I term SUBTRACTS there rather than adding. This is "
                "physiology (V < E_Cl), not a bug -- see MeanFieldModel.psc." %
                (100 * rep["frac_positive"], 100 * rep["max_frac_units_at_one_time"]),
                RuntimeWarning, stacklevel=2)

    if filter_hz is not None:
        out = highpass_eeg(out, fs_hz=1000.0 / dt_record,
                           cutoff_hz=filter_hz, order=filter_order)
    return out, dT


def ipsc_sign_report(ipsc):
    """Fraction of IPSC samples with the WRONG sign for the Lambda formula.

    ``Lambda = I_E(t-d) - alpha*I_I(t)`` is written on the assumption that
    ``I_I < 0``, so that ``-alpha*I_I`` ADDS to the excitatory term.  That
    assumption does not hold in this model: ``ipsc = gI*(E_Cl - V)/f_max`` goes
    positive wherever ``V < E_Cl`` (post-discharge hyperpolarisation, and
    increasingly as seizure chloride loading raises ``E_Cl``).  Where it does,
    the inhibitory term subtracts instead.

    This is PHYSIOLOGY, not a bug, and the LFP stays faithful to the formula
    regardless -- but a reader who assumes a fixed inhibitory sign will be
    wrong, so every LFP computation reports the fraction rather than leaving
    it to be rediscovered.  Measured on the 80 s focal run: 0.023 overall.

    Returns ``{'frac_positive', 'max_frac_units_at_one_time'}``.
    """
    ipsc = np.asarray(ipsc)
    pos = ipsc > 0
    return {
        "frac_positive": float(pos.mean()),
        "max_frac_units_at_one_time": float(pos.mean(axis=1).max()) if ipsc.ndim == 2 else float(pos.mean()),
    }


def highpass_eeg(x, fs_hz: float, cutoff_hz: float = 1.0, order: int = 4):
    """Zero-phase (forward-backward) Butterworth high-pass, WC-compatible.

    Matches the WC-MNIST ``filter_eeg`` convention -- Butterworth high-pass,
    1 Hz, order 4, applied forward then backward so the result is zero-phase
    (no group delay, at the cost of an effectively doubled filter order).

    Applied along axis 0 (time), NaN-safe: leading NaN rows from the Lambda
    delay are preserved and excluded from the filtered span rather than
    poisoning the whole trace.

    **This is a display choice, not part of the model.**  ``lfp_2d`` returns
    UNFILTERED Lambda by default; filtering is opt-in via ``filter_hz`` so the
    raw broadband signal remains available.  Two reasons that matters here:
    Liou's Lambda is broadband and the paper applies no such filter, and a
    1 Hz high-pass ANNIHILATES DC -- in the WC study a saturated network
    rendered as a flat trace, indistinguishable from quiescent.  A tonic
    Liou discharge could alias the same way, so any classification should be
    cross-checked against the unfiltered trace.
    """
    from scipy import signal as _sig

    x = np.asarray(x, dtype=np.float64)
    b, a = _sig.butter(order, cutoff_hz / (0.5 * fs_hz), btype="highpass")
    out = np.full_like(x, np.nan)
    good = np.isfinite(x).all(axis=tuple(range(1, x.ndim))) if x.ndim > 1 else np.isfinite(x)
    seg = x[good]
    if seg.shape[0] > 3 * (order + 1):
        out[good] = _sig.filtfilt(b, a, seg, axis=0)
    return out
