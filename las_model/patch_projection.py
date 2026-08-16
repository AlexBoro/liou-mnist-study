"""3x3 overlapping-patch projection from a 28x28 MNIST image onto a torus sheet.

Geometry (as specified 2026-08-13)
----------------------------------
Pixel ``(r, c)`` (0-indexed, raster order, 28 per row) projects to the 3x3
block of network units::

    rows  offset + 2*r .. offset + 2*r + 2   (mod n)
    cols  offset + 2*c .. offset + 2*c + 2   (mod n)

i.e. **stride 2**, so consecutive patches share exactly one row/column --
"overlap by one unit" in both directions.  In 1-indexed terms this is the
requested map: pixel 1 -> rows 1-3 / cols 1-3, pixel 2 -> cols 3-5,
pixel 29 -> rows 3-5.

Each of a pixel's 9 connections carries its **own** weight, so the full weight
tensor is (28, 28, 3, 3) = 7056 independent values, not 784.

Two facts about this geometry that the caller must know
-------------------------------------------------------
1. **Extent is 57, not 60.**  28 patches at stride 2 with width 3 span
   ``2*27 + 3 = 57`` units.  On a 60x60 sheet that leaves a 3-unit-wide
   undriven stripe in each direction (a cross, given the torus).  Exact
   periodic tiling requires ``n = 2 * 28 = 56``; see ``coverage_report``.
2. **Input multiplicity is a 4:2:1 checkerboard.**  A unit at an even
   in-patch coordinate is covered by two consecutive patches, an odd one by
   one.  In 2-D that gives units covered by 4, 2 or 1 pixels.  The drive is
   therefore spatially inhomogeneous at period 2 units, independent of the
   image.  ``multiplicity_map`` returns it.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .mnist_drive import WC_WEIGHT_MIN, WC_WEIGHT_MAX

PATCH = 3
STRIDE = 2
N_PIX_SIDE = 28
EXTENT = STRIDE * (N_PIX_SIDE - 1) + PATCH  # 57


def sample_patch_weights(rng, n_pix_side: int = N_PIX_SIDE) -> np.ndarray:
    """(n, n, 3, 3) signed weights, normalised exactly as ``mnist_drive`` does.

    Same signed-uniform draw and same max|w| = 1 normalisation, so the WC
    negative fraction and min/max *ratio* are preserved and the resulting
    scale is still in multiples of threshold current.
    """
    w = rng.uniform(WC_WEIGHT_MIN, WC_WEIGHT_MAX, size=(n_pix_side, n_pix_side, PATCH, PATCH))
    return w / np.abs(w).max()


MEASURED_EXPONENT = 0.56
"""Empirical exponent p in drive ~ m**p for MNIST inputs.

Measured as mean |drive| per unit over 200 images, 5 weight draws, n=56:
p = 0.49-0.63, mean 0.563.  The seed-to-seed spread straddles 0.5, so
``compensate="sqrt"`` (the default) is within measurement error of the
flattening exponent and is preferred over this fitted value: it is exact for
independent signed weights rather than tuned to one image set.

Why 0.5 works here.  The sqrt(m) argument assumes each of a unit's m inputs
contributes on every presentation.  MNIST is only ~19% nonzero, which would
push p towards 1.0 if pixels switched on independently -- synthetic inputs at
matched density give p = 0.89.  But MNIST strokes are spatially CORRELATED:
the 2-4 pixels projecting to one unit are neighbours and tend to be on or off
together, so the unit behaves like the dense case (p = 0.54) rather than the
independent-sparse case.  Correlation, not density, is what makes sqrt right.

Residual m4/m1 asymmetry at p_true = 0.56: none 2.18, sqrt 1.09, 1/m 0.55.
"""


def build_projection(n_grid=(56, 56), offset=(0, 0), weights=None, rng=None,
                     n_pix_side: int = N_PIX_SIDE,
                     compensate="sqrt") -> sparse.csr_matrix:
    """Sparse (n_units, n_pixels) map so that ``drive = W @ image.ravel()``.

    Wrapping is modular in both axes, matching ``topology='circular'`` in the
    recurrent connectivity.  At the default 56x56 the 28 patches at stride 2
    span exactly 56, so the sheet tiles with no wrap and no undriven units.

    ``compensate`` selects how a unit's summed drive is scaled by its input
    multiplicity m (the number of pixels projecting to it, 1/2/4 here):

      ``None``            raw sums; drive grows ~m**0.56 across classes
      ``"sqrt"``          divide by sqrt(m) -- FLATTENS to 1.09 (default)
      ``"multiplicity"``  divide by m -- OVERcorrects to 0.55
      ``"measured"``      divide by m**MEASURED_EXPONENT (fitted, 0.56)
      float               divide by m**float

    Because each pixel's 9 connections carry INDEPENDENT SIGNED weights, the
    summed drive at an m-fold site grows roughly as sqrt(m) (random-sign
    cancellation), not m.  Dividing by m overshoots and inverts the asymmetry.
    Measured relative drive at m=1/2/4 (200 MNIST images, seed 0):
      none 1.00/1.65/2.24 | sqrt 1.00/1.17/1.09 | 1/m 1.00/0.83/0.55
    """
    if weights is None:
        weights = sample_patch_weights(rng if rng is not None else np.random.default_rng(0), n_pix_side)
    ny, nx = n_grid
    oy, ox = offset
    rows, cols, vals = [], [], []
    for r in range(n_pix_side):
        for c in range(n_pix_side):
            pix = r * n_pix_side + c
            for dy in range(PATCH):
                for dx in range(PATCH):
                    uy = (oy + STRIDE * r + dy) % ny
                    ux = (ox + STRIDE * c + dx) % nx
                    rows.append(uy * nx + ux)
                    cols.append(pix)
                    vals.append(weights[r, c, dy, dx])
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(ny * nx, n_pix_side ** 2))
    W.weights_tensor = weights
    if compensate:
        # 'sqrt' divides by sqrt(m) and FLATTENS the multiplicity classes to
        # within ~9%; 'multiplicity' divides by m and OVER-corrects to 0.55.
        # Each pixel's 9 connections carry independent SIGNED weights, so the
        # summed drive at an m-fold site grows as m**p with p ~ 0.56 measured
        # (0.5 predicted by random-sign cancellation).  Measured m=1/2/4 --
        #   none 1.00/1.65/2.24 | sqrt 1.00/1.17/1.09 | 1/m 1.00/0.83/0.55
        # See PORT_NOTES.md.  NOTE any mode makes a pixel's 9 weights no
        # longer independent free parameters: the effective weight a unit sees
        # is w / m(unit)**expo and m depends on the unit's parity, so the same
        # nominal weight acts with different strength at different sites.
        # Training sees the nominal weight; the model responds to the scaled one.
        m = np.asarray(sparse.csr_matrix(
            (np.ones(len(vals)), (rows, cols)),
            shape=(ny * nx, n_pix_side ** 2)).sum(axis=1)).ravel()
        mm = np.maximum(m, 1)
        if compensate in (True, "multiplicity"):
            expo = 1.0
        elif compensate == "sqrt":
            expo = 0.5
        elif compensate == "measured":
            expo = MEASURED_EXPONENT
        elif isinstance(compensate, (int, float)) and not isinstance(compensate, bool):
            expo = float(compensate)
        else:
            raise ValueError(
                "compensate must be None, 'sqrt', 'multiplicity', 'measured', "
                "or a float exponent; got %r" % (compensate,))
        denom = mm ** expo
        inv = np.where(m > 0, 1.0 / denom, 0.0)
        W = sparse.diags(inv) @ W
        W = W.tocsr()
        W.weights_tensor = weights
        W.multiplicity = m.reshape(ny, nx)
        W.compensated = compensate
    else:
        W.compensated = False
    return W


def multiplicity_map(n_grid=(56, 56), offset=(0, 0), n_pix_side: int = N_PIX_SIDE) -> np.ndarray:
    """How many pixels project to each unit (the 4:2:1 checkerboard)."""
    W = build_projection(n_grid, offset, weights=np.ones((n_pix_side, n_pix_side, PATCH, PATCH)),
                         n_pix_side=n_pix_side, compensate=False)
    return np.asarray(W.sum(axis=1)).reshape(n_grid)


def coverage_report(n_grid=(56, 56), offset=(0, 0), n_pix_side: int = N_PIX_SIDE) -> dict:
    m = multiplicity_map(n_grid, offset, n_pix_side)
    counts = {int(k): int(v) for k, v in zip(*np.unique(m, return_counts=True))}
    return {
        "n_grid": tuple(n_grid),
        "extent": STRIDE * (n_pix_side - 1) + PATCH,
        "tiles_exactly": (STRIDE * n_pix_side) == n_grid[0] == n_grid[1],
        "n_units": int(m.size),
        "n_undriven": int((m == 0).sum()),
        "frac_undriven": float((m == 0).mean()),
        "multiplicity_counts": counts,
        "n_weights": int(n_pix_side ** 2 * PATCH * PATCH),
    }
