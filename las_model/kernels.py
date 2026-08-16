"""Gaussian projection kernels and the field convolution.

Port of ``StandardRecurrentConnection.m`` (kernel construction) and
``conv2_field.m`` (the 'linear' topology branch) from Liou's LAS-Model.

MATLAB conventions preserved here
---------------------------------
* Kernel half-width is ``ceil(2.5 * sigma_pixels)``; the sampled grid runs
  ``-(K-1) .. (K-1)`` so the kernel has ``2K-1`` taps per dimension.  This
  truncates the Gaussian at ~2.3-2.5 sigma depending on rounding, and the
  kernel is renormalised to unit sum *after* truncation.
* ``Sigma_E = diag(n) * 0.02`` means sigma is expressed as a fraction of the
  network extent along each dimension, converted to pixels by multiplying by
  the number of units along that dimension.
* ``mvnpdf(x, 0, Sigma.^2)`` takes a *covariance* matrix, so ``Sigma`` itself
  holds standard deviations in pixels.
* ``conv2_field(..., 'linear')`` performs a full convolution then shifts by
  ``(n_full - n_target)/2``, which for an odd-length kernel is exactly a
  zero-padded 'same'-mode convolution.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def gaussian_kernel_1d(sigma_pixels: float) -> np.ndarray:
    """One separable factor of the MATLAB kernel, normalised to unit sum.

    ``sigma_pixels`` <= 0 (a singleton dimension) yields the length-1 kernel
    ``[1.0]``, which is what ``mvnpdf`` sampled at a single point reduces to
    after normalisation.
    """
    if sigma_pixels <= 0:
        return np.ones(1)
    half = int(np.ceil(2.5 * sigma_pixels))
    half = max(half, 1)
    x = np.arange(-(half - 1), half, dtype=float)
    k = np.exp(-0.5 * (x / sigma_pixels) ** 2)
    return k / k.sum()


def gaussian_kernel_2d(sigma_pixels) -> np.ndarray:
    """Full 2-D kernel exactly as ``StandardRecurrentConnection.m`` builds it."""
    s1, s2 = sigma_pixels
    k1 = gaussian_kernel_1d(s1)
    k2 = gaussian_kernel_1d(s2)
    return np.outer(k1, k2)


def sigma_pixels_from_fraction(n, fraction: float):
    """``diag(n) * fraction`` -> per-dimension sigma in units of neurons."""
    return (n[0] * fraction, n[1] * fraction)


def conv_field(field: np.ndarray, k1: np.ndarray, k2: np.ndarray,
               topology: str = "linear") -> np.ndarray:
    """Separable 'same' convolution in either of Liou's two topologies.

    ``topology='linear'`` (default) is the zero-padded convolution used by
    every figure in the paper: equivalent to ``scipy.signal.convolve2d(field,
    np.outer(k1, k2), mode='same')`` but O(n*K) instead of O(n*K^2).

    ``topology='circular'`` wraps instead of truncating, i.e. the network
    lives on a torus.  This is the ``'circular'`` branch of ``conv2_field.m``
    and ``@Projection/Projection.m``, which fold the tails of the full
    convolution back onto the opposite edge.  For a kernel no larger than the
    field that fold is exactly ``mode='wrap'``.

    ``tests/test_port.py`` asserts both against direct reference
    implementations to machine precision.
    """
    if topology not in ("linear", "circular"):
        raise ValueError("topology must be 'linear' or 'circular', got %r" % topology)
    mode = "constant" if topology == "linear" else "wrap"
    out = field
    if k1.size > 1:
        out = ndimage.convolve1d(out, k1, axis=0, mode=mode, cval=0.0)
    if k2.size > 1:
        out = ndimage.convolve1d(out, k2, axis=1, mode=mode, cval=0.0)
    return out


def conv_field_reference(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Direct (slow) reference implementation used only by the tests."""
    return signal.convolve2d(field, kernel, mode="same", boundary="fill", fillvalue=0.0)
