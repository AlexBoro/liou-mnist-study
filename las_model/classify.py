"""Seizure classification for the Liou-MNIST sweep, from SHEET metrics.

Two rules live here.  ``auto2`` is the montage-based rule the pilot used,
preserved verbatim so its 54 labels stay reproducible and auditable.  ``sheet``
is its replacement, fitted on 54 human-scored runs and used for new work.

WHY THE MONTAGE RULE IS RETIRED
-------------------------------
Scored blind against 42 human calls (2026-08-16, sigma 0.11 panels), ``auto2``
reaches 100% specificity and **29% sensitivity**: it misses 12 of 17 runs the
user calls a seizure.  The misses have three causes, all structural rather than
threshold-tuning:

  * 5 runs fail ``amp >= 1.2e4``.  The bar was fitted on sigma-0.025 montages;
    amplitude is not invariant to the spatial constant.
  * 2 runs fail ``cv >= 1.15``.  A regular, high-amplitude discharge train is
    still a seizure; interval variability is not a requirement.
  * 5 runs are captured by the entrainment arm first, because their discharge
    IPI happens to land within 5% of 2x or 3x the 250 ms image period.  This is
    the worst failure mode: the arm exists to keep stimulus-following out of the
    ictal class, and it is instead vetoing seizures whose own rhythm coincides
    with a harmonic.  ``PORT_NOTES.md`` gives the correct control for that
    ambiguity -- change ``image_duration`` and see whether the discharge IPI
    follows -- which a single run's scalars cannot answer.

THE SHEET RULE
--------------
``unit_duty_median`` -- the median over units of the fraction of time that unit
spends above 20 Hz -- separates the user's four classes on all 54 scored runs
with no overlap:

    class            duty_median range (n)
    ENTRAINED        0.0000 - 0.0016  (25)
    seizure          0.0490 - 0.6750  (21)   focal and generalized pooled
    TONIC_SATURATION 0.8673 - 0.9996   (8)

Thresholds are placed in the gaps, not on the data: 0.02 sits between 0.0016 and
0.0490 (a 30x gap), and 0.80 between 0.6750 and 0.8673.  The lower boundary is
wide; **the upper one is not** -- 1.28x, on 8 tonic runs and 21 seizures -- so
treat a duty near 0.7-0.9 as unresolved rather than classified, which is what
``margin`` reports.

Two properties matter for the sweep.  It is scale-free: duty is a fraction of
time above a rate threshold, so unlike the amplitude bars it does not move when
sigma_LFP or the channel set changes.  And it reads the whole sheet, so it does
not inherit the montage's blind spot -- ``FULLFIELD_FINDINGS.md`` showed the
30-electrode montage cannot separate depolarization block from quiescence, and
under this rule the block runs are the highest-duty class rather than the
lowest-amplitude one.

WHAT IT DOES NOT DO
-------------------
It does not separate the two seizure subtypes, which are defined by ONSET
DYNAMICS (renamed 2026-08-19 from focal/generalized, which wrongly implied
spatial extent):

  * "early slow tonic wavefront -> fast traveling waves" -- the pattern closer
    to Liou et al. (2020);
  * "fast traveling waves at onset".

Every stored scalar is measured over the last 45% of the run, i.e. after onset,
so the information is not present.  Of nine end-window features tested against
the pass-2 split (13 slow-onset / 8 fast-onset), none separates them cleanly;
rise/decay asymmetry did on the smaller pass-1 split and does not survive the
larger one.  ``sheet()`` therefore returns SEIZURE and leaves the subtype to the
human or to a future onset metric -- a wavefront-speed or time-resolved duty
measurement, both computable from the stored 80 ms envelope without a re-run.
"""
from __future__ import annotations

import numpy as np

IMAGE_MS = 250.0

# --- sheet rule -----------------------------------------------------------
DUTY_SEIZURE = 0.02     # below: not a seizure.  Gap in data: 0.0016 .. 0.0490
DUTY_TONIC = 0.80       # above: tonic saturation.  Gap: 0.6750 .. 0.8673
_NARROW = 0.10          # relative margin under which a call is "unresolved"

CLASSES = ("NOT_SEIZURE", "SEIZURE", "TONIC_SATURATION")


def sheet(unit_duty_median):
    """Class from the sheet duty metric.  See module docstring for provenance.

    Returns (label, margin) where margin is the fractional distance to the
    nearer threshold; < 0.10 means the run sits close enough to a boundary that
    it should be looked at rather than trusted.
    """
    d = float(unit_duty_median)
    if not np.isfinite(d):
        return "NOT_SEIZURE", 0.0
    if d >= DUTY_TONIC:
        lab, edge = "TONIC_SATURATION", DUTY_TONIC
    elif d >= DUTY_SEIZURE:
        lab = "SEIZURE"
        edge = DUTY_TONIC if (DUTY_TONIC - d) < (d - DUTY_SEIZURE) else DUTY_SEIZURE
    else:
        lab, edge = "NOT_SEIZURE", DUTY_SEIZURE
    return lab, abs(d - edge) / max(edge, 1e-12)


def is_unresolved(unit_duty_median):
    """True when the run is too near a boundary to classify without eyes."""
    return sheet(unit_duty_median)[1] < _NARROW


def auto2(amp, cv, ipi, asym, image_ms=IMAGE_MS):
    """The pilot's montage rule, PRESERVED FOR REPRODUCIBILITY -- do not use it
    for new classification (29% sensitivity; see module docstring).

    Reproduces all 54 rows of liou_pilot_classes.csv exactly.
    """
    if not np.isfinite(amp) or amp < 2000:
        return "QUIET_OR_BLOCK"
    k = ipi / image_ms
    if abs(k - round(k)) < 0.05 and round(k) >= 2:
        return "ENTRAINED"
    if cv >= 1.15 and amp >= 1.2e4 and asym < 1.3:
        return "ICTAL"
    return "ACTIVE_NONICTAL"
