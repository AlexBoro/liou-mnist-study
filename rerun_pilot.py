"""MNIST image pool for the Liou pilot runs.

RECONSTRUCTED 2026-08-16 after the original rerun_pilot.py was lost in a
workspace cleanup (it was never saved as an artifact).

THE POOL IS THE FIRST 400 MNIST TRAINING IMAGES as (400, 28, 28) float32,
sha256[:16] = 6f1f96ff627f7593.  That is checked at load time below, so a
wrong file or a wrong slice fails immediately instead of producing a run that
merely looks plausible.

TWO PROPERTIES THAT ARE EASY TO GET WRONG, both silent:

  * SHAPE MUST BE (n, 28, 28), NEVER (n, 784).  PatchDrive._refresh masks the
    live row with ``m[self._row] = img[self._row]``; on a flat image that
    selects a single PIXEL, so sequential runs receive essentially no drive.
    Simultaneous runs are unaffected because _refresh ravels anyway -- which is
    why the bug hides: half the runs reproduce and only the _seq half collapses.
  * POOL SIZE IS PART OF THE RNG STREAM.  image_stream draws
    ``rng.integers(0, len(images), size=n_images)`` from the SAME generator that
    later feeds the noise, so a different pool size changes both the digit
    sequence and every subsequent random draw.  X[:200] instead of X[:400]
    shifts amplitudes ~3%.

A NOTE ON PROBING THIS, learned the hard way: do NOT try to identify the digit
by running a single-image pool.  ``rng.integers(0, 1, ...)`` consumes a
different number of words than ``rng.integers(0, 400, ...)``, so a one-image
pool silently changes the whole downstream noise trajectory and the comparison
is meaningless -- it "identified" train index 250 when the run actually draws
340.  Likewise do not compare truncated runs: the LFP high-pass filter edge
differs, so a short re-run mismatches even when it is correct.  Verify only
end-to-end at full T_end, against pilot_out_v2, via
``pilot_worker.run_and_reduce_field(verify_against=...)``.
"""
from __future__ import annotations

import glob
import hashlib
import struct

import numpy as np

# Candidate locations for the raw MNIST training images, in preference order.
IDX_PATHS = (
    "mnist_data/MNIST/raw/train-images-idx3-ubyte",
    "~/dev/wc-mnist-study/mnist_data/MNIST/raw/train-images-idx3-ubyte",
)

N_POOL = 400          # the pilot pool; see the module docstring
POOL_SHA16 = "6f1f96ff627f7593"


def _idx_path():
    for p in IDX_PATHS:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    raise SystemExit("MNIST idx file not found in %s" % (IDX_PATHS,))


def mnist_images(n=N_POOL):
    """First ``n`` MNIST training images as (n, 784) float64 in [0, 1].

    Row-major (n, 784), NOT transposed: patch_projection expects one image per
    row, and a transposed array once made the drive-compensation exponent read
    as 0.74 instead of 0.56 without any error being raised.
    """
    with open(_idx_path(), "rb") as fh:
        _, cnt, r, c = struct.unpack(">IIII", fh.read(16))
        if n > cnt:
            raise SystemExit("asked for %d images, file has %d" % (n, cnt))
        buf = fh.read(n * r * c)
    a = (np.frombuffer(buf, dtype=np.uint8).reshape(n, r, c).astype(np.float32) / 255.0)
    # (n, 28, 28), NOT (n, 784): drive_field does ``img * weights`` with weights
    # shaped (28, 28).  A flat pool makes MNISTDrive raise inside _refresh, and
    # in the pilot_sweep path the net effect was that the digit contributed
    # nothing at all -- a run with real images was bit-identical to one with
    # zeros.  Silent, and it invalidates the whole run.
    assert a.shape == (n, 28, 28), a.shape
    if n == N_POOL:
        _check_pool(a)
    return a


def _check_pool(a):
    """Fail loudly if the reconstructed pool is not the pilot's."""
    got = hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]
    if got != POOL_SHA16:
        raise SystemExit("MNIST pool sha256[:16]=%s, expected %s -- wrong file, "
                         "wrong slice, or wrong dtype" % (got, POOL_SHA16))
    return a
