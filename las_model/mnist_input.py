"""MNIST stimulus for the Liou mean-field sheet.

Design notes (all numbers measured, see PORT_NOTES.md "MNIST input on the torus"):

* The digit occupies a ``side x side`` patch of an ``n x n`` torus.  At the
  default 28 in 50 the patch is 31.4% of the sheet and leaves an 11-unit
  margin to the wrap seam -- 11 sigma_E, so the digit cannot interact with
  its own periodic image through the excitatory kernel.
* Current is referenced to THRESHOLD, not to Liou's Fig-1C trigger.  A unit
  needs ``(phi0 - E_L) * g_L`` pA of injected current to sit at half of
  ``f_max`` in the absence of recurrence: 52 pA at the standard parameters.
  ``I_THRESHOLD_PA`` records this, and ``scale`` is expressed in multiples of
  it, so the same scale means the same thing if E_L or g_L change.
* The stimulus enters the SAME ``I_ext`` slot as the background noise -- it is
  just another ``ExternalInput`` with a ``deterministic`` callable.  See
  PORT_NOTES.md "The I_ext slot".
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .model import ExternalInput

__all__ = [
    "threshold_current",
    "patch_origin",
    "patch_mask",
    "embed",
    "MNISTStimulus",
]


def threshold_current(params) -> float:
    """pA of injected current putting an isolated unit at 0.5 * f_max.

    From ``V_inf = (g_L*E_L + I_ext)/g_L`` with no synaptic drive: the unit
    reaches ``phi`` (where ``activation`` is half-maximal) at
    ``I = (phi0 - E_L) * g_L``.  Recurrence lowers this; it is a reference
    scale, not a prediction.
    """
    g_L = float(np.mean(params.g_L))
    E_L = float(np.mean(params.E_L))
    phi0 = float(np.mean(params.phi0))
    return (phi0 - E_L) * g_L


def patch_origin(n: int = 50, side: int = 28) -> Tuple[int, int]:
    """Centred origin.  On a torus every origin is equivalent by translation
    invariance (verified: tests/test_port.py::test_torus_is_translation_
    equivariant), so this is a convention, not a tuning choice."""
    return ((n - side) // 2, (n - side) // 2)


def patch_mask(n: int = 50, side: int = 28, origin=None) -> np.ndarray:
    """Boolean (n, n) mask selecting the stimulated patch."""
    r0, c0 = patch_origin(n, side) if origin is None else origin
    m = np.zeros((n, n), dtype=bool)
    m[r0:r0 + side, c0:c0 + side] = True
    return m


def embed(image: np.ndarray, n: int = 50, origin=None) -> np.ndarray:
    """Place a ``side x side`` image into an (n, n) field, zero elsewhere.

    Row-major, matching the WC-MNIST convention where pixel (r, c) maps to
    flat index ``28*r + c`` -- so an input-weight vector can be reshaped
    between the two studies without reordering.

    Placement WRAPS: the sheet is a torus, so an origin near the seam
    carries the digit around it rather than truncating it.  (A plain
    ``F[r0:r0+side, c0:c0+side] = image`` silently drops the overhanging
    rows/columns -- that was a bug here, caught by the translation-
    equivariance test.)
    """
    side = image.shape[0]
    r0, c0 = patch_origin(n, side) if origin is None else origin
    F = np.zeros((n, n), dtype=float)
    rows = (np.arange(side) + r0) % n
    cols = (np.arange(side) + c0) % n
    F[np.ix_(rows, cols)] = image
    return F


class MNISTStimulus:
    """A swappable MNIST drive for ``MeanFieldModel``.

    Register once with ``model.ext.append(stim.as_input())``; then call
    ``stim.set(image)`` between presentations.  The registered callable reads
    the current image each step, so no re-registration is needed and the
    model's own ``I_ext`` accumulation is untouched.

    Parameters
    ----------
    params
        The model's ``MeanFieldParams`` -- used only for the threshold scale.
    scale
        Peak drive for a saturated pixel, in multiples of threshold current.
        1.0 puts a full-intensity pixel exactly at the isolated-unit
        threshold.  Measured digit separability at 250 ms rises monotonically
        over 0.4-4.0 with no saturation, so this is a soft knob.
    weights
        Optional per-unit multiplicative gain of shape ``(side, side)``, the
        analogue of WC-MNIST's learnable ``mnist_weights``.  Pass an array to
        make the input trainable; leave ``None`` for a fixed stimulus.
    row_schedule
        ``None`` for simultaneous presentation.  For sequential presentation
        pass the number of steps per row; ``set_time`` then gates rows.
    """

    def __init__(self, params, scale: float = 2.0, n: int = 50, side: int = 28,
                 origin=None, weights: Optional[np.ndarray] = None):
        self.params = params
        self.n = n
        self.side = side
        self.origin = patch_origin(n, side) if origin is None else origin
        self.I_thresh = threshold_current(params)
        self.scale = scale
        self.weights = weights
        self.mask = patch_mask(n, side, self.origin)
        self._field = np.zeros((n, n), dtype=float)

    # -- stimulus construction -------------------------------------------
    def field_for(self, image: np.ndarray) -> np.ndarray:
        """(n, n) current field in pA for a [0, 1] image."""
        img = np.asarray(image, dtype=float)
        if img.max() > 1.0 + 1e-9:
            raise ValueError("image must be scaled to [0, 1] before use")
        if self.weights is not None:
            img = img * self.weights
        return embed(img * (self.scale * self.I_thresh), self.n, self.origin)

    def set(self, image: Optional[np.ndarray]) -> None:
        """Swap in a new image, or ``None`` for a blank interval."""
        self._field = (np.zeros((self.n, self.n)) if image is None
                       else self.field_for(image))

    def set_row(self, image: np.ndarray, row: int) -> None:
        """Sequential presentation: drive only one image row."""
        masked = np.zeros_like(np.asarray(image, dtype=float))
        masked[row] = np.asarray(image, dtype=float)[row]
        self.set(masked)

    def as_input(self) -> ExternalInput:
        return ExternalInput(deterministic=lambda x1, x2, t: self._field)
