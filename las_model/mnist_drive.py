"""WC-analogous MNIST drive for the Liou mean-field sheet.

Mirrors the WC-MNIST stimulus scheme (``wc_model.compute_mnist_input`` +
``wc_model.active_row_at``) so the two studies differ in DYNAMICS, not in how
the digit is delivered.  What is deliberately kept, and what necessarily
changes:

KEPT from WC-MNIST
------------------
* A single global gain constant (``MNIST_INPUT_SCALE``) applied to the images.
* A per-pixel vector of signed random weights (``mnist_weights``, length 784)
  drawn i.i.d. uniform from ``[MNIST_WEIGHT_MIN, MNIST_WEIGHT_MAX]``.  Sign
  included: in WC ~48% of pixels INHIBIT their unit, and that is retained.
* Elementwise, identity pixel->unit mapping.  No input connectivity matrix, no
  receptive field: pixel (r, c) drives exactly one unit and spreads only
  through the recurrent kernels on later steps.
* The continuous-time row schedule for sequential presentation, ported so that
  rows dither 5/4 steps and all 28 rows are visited (see ``active_row_at``).

CHANGED, and why
----------------
* WC's drive is dimensionless; Liou's ``I_ext`` is in pA.  Transferring WC's
  literal bounds U(-12, 13) as multiples of threshold current would put a
  saturated pixel at 3.0 * 13 * 52 pA = 2028 pA, i.e. 10x Liou's own
  seizure-triggering amplitude.  So the weight bounds here are NORMALISED to
  max|w| = 1 -- U(-0.923, 1.0), which preserves WC's -12/13 sign ratio and
  hence its 48% negative fraction exactly -- and ``MNIST_INPUT_SCALE`` is
  expressed in multiples of ``threshold_current(params)``.  WC's literal
  numbers are recovered by scaling: WC's effective peak gain 3.0*13 = 39
  corresponds to ``MNIST_INPUT_SCALE = 39`` here.  Calibrating the scale is
  the point of ``sweep_scale``; these bounds are a starting convention, not a
  claim about the right value.

Total drive is therefore, in pA:

    I_ext[unit(r,c)] = image[r, c] * MNIST_INPUT_SCALE * I_THRESHOLD * w[28r + c]

TIMESCALE WARNING
-----------------
``tau_Cl = tau_K = 5000 ms``.  A 10 s run is TWO time constants of the slow (measured 2026-08-13: 80 s
equilibrates g_K to 0.7% over the final quarter; 10 s does not)
variables that gate Liou's seizure transition, and a 250 ms image is 1/20th of
one.  WC's slowest variable is ``tau_H = 1000 ms``, so relative to the image
the Liou sheet is 5x slower.  ``run_mnist`` records the slow states so this is
measurable rather than assumed; see ``PORT_NOTES.md``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .mnist_input import embed, patch_origin, patch_mask, threshold_current

__all__ = [
    "MNIST_INPUT_SCALE", "MNIST_WEIGHT_MIN", "MNIST_WEIGHT_MAX",
    "IMAGE_DURATION_MS", "N_ROWS", "WC_WEIGHT_MIN", "WC_WEIGHT_MAX",
    "WC_INPUT_SCALE",
    "sample_mnist_weights", "active_row_at", "drive_field", "MNISTDrive",
]

# ---------------------------------------------------------------------------
# Constants.  MNIST_INPUT_SCALE is in multiples of threshold current; the
# weight bounds are normalised (see module docstring).
# ---------------------------------------------------------------------------
MNIST_INPUT_SCALE = 3.0
MNIST_WEIGHT_MIN = -12.0 / 13.0      # = -0.9231, preserves WC's sign ratio
MNIST_WEIGHT_MAX = 1.0

IMAGE_DURATION_MS = 250.0            # matches WC IMAGE_DURATION = 0.25 s
N_ROWS = 28

# WC's literal values, for conversion only -- not used in the drive.
WC_INPUT_SCALE, WC_WEIGHT_MIN, WC_WEIGHT_MAX = 3.0, -12.0, 13.0


def sample_mnist_weights(rng, side: int = 28,
                         lo: float = MNIST_WEIGHT_MIN,
                         hi: float = MNIST_WEIGHT_MAX) -> np.ndarray:
    """(side, side) signed per-pixel gains, i.i.d. uniform on [lo, hi].

    WC draws a flat (784,) vector and indexes it as ``28*r + c``; ``embed``
    uses the same row-major convention, so ``sample_mnist_weights(...).ravel()``
    is directly comparable to WC's ``mnist_weights`` and either can be
    reshaped into the other without reordering.
    """
    return np.asarray(rng.uniform(lo, hi, size=side * side),
                      dtype=float).reshape(side, side)


def active_row_at(step: int, dt: float, image_duration: float,
                  row_duration: Optional[float] = None,
                  n_rows: int = N_ROWS) -> int:
    """Which image row is live at Euler ``step``.  Port of ``wc_model.active_row_at``.

    Recomputes the row from CONTINUOUS time rather than a quantised step count.
    That is load-bearing, not stylistic: ``row_duration/dt`` is 8.9286 at the
    250 ms / 1 ms defaults, and quantising it to ``int()`` drops the remainder
    on every row until the ``min(..., n_rows-1)`` clamp parks the leftover
    steps on the near-empty bottom row.  On the WC side that cost 11.7% of the
    delivered drive per image (measured, 10k test set).  Computing against the
    exact row_duration makes rows take 8 or 9 steps as needed, summing to
    exactly one image.

    ``step`` may be image-local or a global counter; the modulo handles both,
    so a probe run and a training run get identical schedules.
    """
    if row_duration is None:
        row_duration = image_duration / n_rows
    time_in_image = (step * dt) % image_duration
    return min(int(time_in_image / row_duration), n_rows - 1)


def drive_field(image: np.ndarray, weights: Optional[np.ndarray],
                I_threshold: float, scale: float = MNIST_INPUT_SCALE,
                n: int = 56, origin=None, active_row: Optional[int] = None
                ) -> np.ndarray:
    """(n, n) pA current field for one image, WC-style.

    ``active_row=None`` gives simultaneous presentation (all 784 pixels live).
    An int masks to that row alone -- 28 of 784 units -- which is WC's
    sequential configuration.  Masking happens BEFORE the weights are applied
    but the weights are unchanged between modes, exactly as WC does it (the
    same ``mnist_weights`` is sliced, not rescaled), so per-step sequential
    drive is 1/28 of simultaneous and no per-row boost is applied.
    """
    img = np.asarray(image, dtype=float)
    if img.max() > 1.0 + 1e-9:
        raise ValueError("image must be scaled to [0, 1] before use")
    if active_row is not None:
        masked = np.zeros_like(img)
        masked[active_row] = img[active_row]
        img = masked
    if weights is not None:
        img = img * weights
    return embed(img * (scale * I_threshold), n, origin)


class MNISTDrive:
    """Swappable WC-analogous MNIST drive registered once on the model.

    Register with ``model.ext.append(drive.as_input())``.  The callable closes
    over a mutable field, so ``set_image``/``set_step`` swap the stimulus
    without re-registering and without disturbing the model's own ``I_ext``
    accumulation (see PORT_NOTES.md "The I_ext slot").
    """

    def __init__(self, params, rng=None, scale: float = MNIST_INPUT_SCALE,
                 n: int = 56, side: int = N_ROWS, origin=None,
                 weights: Optional[np.ndarray] = None,
                 sequential: bool = False,
                 weight_bounds: Tuple[float, float] = (MNIST_WEIGHT_MIN,
                                                       MNIST_WEIGHT_MAX)):
        self.params = params
        self.n, self.side = n, side
        self.origin = patch_origin(n, side) if origin is None else origin
        self.I_threshold = threshold_current(params)
        self.scale = float(scale)
        self.sequential = bool(sequential)
        self.mask = patch_mask(n, side, self.origin)
        if weights is None and rng is not None:
            weights = sample_mnist_weights(rng, side, *weight_bounds)
        self.weights = weights
        self._image: Optional[np.ndarray] = None
        self._field = np.zeros((n, n), dtype=float)
        self._row: Optional[int] = None

    # -- stimulus control -------------------------------------------------
    def set_image(self, image: Optional[np.ndarray]) -> None:
        self._image = None if image is None else np.asarray(image, dtype=float)
        self._refresh()

    def set_row(self, row: Optional[int]) -> None:
        self._row = row
        self._refresh()

    def set_step(self, step: int, dt: float) -> None:
        """Advance the sequential schedule to ``step`` (no-op if simultaneous)."""
        if self.sequential:
            self.set_row(active_row_at(step, dt, IMAGE_DURATION_MS))

    def _refresh(self) -> None:
        if self._image is None:
            self._field = np.zeros((self.n, self.n), dtype=float)
        else:
            self._field = drive_field(
                self._image, self.weights, self.I_threshold, self.scale,
                self.n, self.origin,
                active_row=self._row if self.sequential else None)

    # -- diagnostics ------------------------------------------------------
    def drive_stats(self) -> dict:
        """pA statistics of the CURRENT field, over stimulated units only."""
        f = self._field[self.mask]
        return dict(peak_pA=float(np.abs(f).max()) if f.size else 0.0,
                    mean_pA=float(f.mean()) if f.size else 0.0,
                    frac_negative=float((f < 0).mean()) if f.size else 0.0,
                    frac_suprathreshold=float((f > self.I_threshold).mean())
                    if f.size else 0.0)

    def as_input(self):
        from .model import ExternalInput
        return ExternalInput(deterministic=lambda x1, x2, t: self._field)


class PatchDrive(MNISTDrive):
    """MNISTDrive with the 3x3 overlapping-patch projection replacing identity.

    Sequential mode masks the IMAGE row before projection (so an active row
    still drives its 3-unit-tall band of network rows, correctly overlapping
    its neighbours), rather than masking network rows -- masking downstream
    would clip the patch and break the overlap the geometry is for.
    """

    def __init__(self, params, rng=None, scale: float = MNIST_INPUT_SCALE,
                 n: int = 56, side: int = N_ROWS, sequential: bool = False,
                 weights=None, compensate: object = "sqrt", offset=None, **kw):
        from .patch_projection import build_projection, sample_patch_weights, STRIDE
        if rng is None:
            rng = np.random.default_rng(0)
        if weights is None:
            weights = sample_patch_weights(rng, side)
        if offset is None:
            pad = (n - STRIDE * side) // 2
            offset = (max(pad, 0), max(pad, 0))
        self.offset = tuple(offset)
        self.W = build_projection((n, n), offset=self.offset, weights=weights, n_pix_side=side,
                                  compensate=compensate)
        self.compensate = compensate
        self.patch_weights = weights
        super().__init__(params, rng=rng, scale=scale, n=n, side=side,
                         sequential=sequential, weights=weights, **kw)
        self.mask = (np.asarray(self.W.sum(axis=1)).ravel() != 0).reshape(n, n)

    def _refresh(self) -> None:
        if self._image is None:
            self._field = np.zeros((self.n, self.n), dtype=float)
            return
        img = np.asarray(self._image, dtype=float)
        if self.sequential:
            m = np.zeros_like(img)
            if self._row is not None:
                m[self._row] = img[self._row]
            img = m
        self._field = (self.scale * self.I_threshold
                       * (self.W @ img.ravel()).reshape(self.n, self.n))
