"""Batch 4: 100 unbiased sequential draws, plus 6 replicates and the anchors.

WHY SEQUENTIAL ONLY.  Presentation configuration has never moved the coarse
class: across 31 paired draws scored by hand it agreed 31/31, and the one fine
disagreement (draw32) was a subtype, both seizure.  Pairing therefore costs half
the parameter coverage per run scored, and scoring -- not compute -- is the
binding constraint at ~54 s per run.  Sequential is also the cleaner arm: 0 of
39 sequential runs clamp chloride against 19 of 41 simultaneous.

WHY NO TARGETED ARM.  A duty surrogate fitted on the 35 scored draws reaches
cross-validated R2 0.38 (MAE 0.17 against an overlap band 0.49 wide).  Selecting
draws predicted to land in the tonic/seizure overlap is right 55% of the time
against a 34% base rate, so 40 targeted runs would have bought about 8 extra
in-band runs for 40 unbiased parameter points.  Measured, rejected, and recorded
in batch4_surrogate_diagnostics.csv rather than assumed either way.

DRAW SET.  The 15 draws of the original 0-49 sequence that were run but never
hand-scored, plus 85 new draws k=50..134.  draw(k) is seeded 1000+k, so the new
draws continue the same deterministic sequence and the 15 reproduce exactly.

REPLICATES.  Two parameter sets at three weight seeds each.  The corpus has
never had a label-noise estimate: every conclusion about class boundaries so far
assumes a draw's class is a property of its parameters, and nothing has tested
how often the same parameters give a different call.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import las_model.pilot_sweep as PS
import las_model.pilot_worker as PW
import rerun_pilot as RP

OUT = "batch4_out"
SIGMA = 0.11
T_END = 80000.0
WORKERS = 4

# the 15 unscored draws of 0..49, then 85 new
UNSCORED_0_49 = [23, 24, 26, 27, 29, 30, 31, 35, 36, 37, 38, 41, 43, 44, 46]
DRAWS = UNSCORED_0_49 + list(range(50, 135))

# Replicate arm: two draws at three weight seeds each.  The standard run of
# draw k uses seed k, so the two extra seeds are offsets that cannot collide
# with any draw index in this batch (max 134).
REPLICATE_DRAWS = (50, 51)
REPLICATE_SEEDS = (50, 51, 900, 901)


def rows():
    """Every run in the batch: draws, replicates, then the anchors.

    Anchors are included because check_anchor() is the detector's own sanity
    check -- a batch whose anchors disagree has every draw label in doubt.
    """
    out = {}
    for k in DRAWS:
        d = PS.draw(k)
        # SEED = DRAW INDEX.  This is PS.manifest's convention, not an arbitrary
        # choice, and it is load-bearing: the seed sets the MNIST weight draw, so
        # running draw k at seed 0 gives a DIFFERENT run with the same
        # parameters.  Caught 2026-08-29 when re-running draw23 returned duty
        # 0.000250 against the 0.000312 in sweep50_labels.csv -- one time sample
        # in 16000, deterministic on repeat, and entirely down to the seed.  Had
        # it gone unnoticed, all 100 runs would have been incomparable with the
        # existing corpus while looking perfectly reasonable.
        out["draw%02d_seq" % k] = dict(d, tag="draw%02d" % k, base_tag="draw%02d" % k,
                                       kind="draw", sequential=True, seed=k,
                                       expected="")
    for k in REPLICATE_DRAWS:
        d = PS.draw(k)
        for s in REPLICATE_SEEDS:
            if s == k:
                continue                      # that is the standard draw run
            # A REPLICATE NEEDS ITS OWN TAG, not just its own dict key.
            # pilot_worker.run_key(row) is "<tag>_<seq|sim>" and ignores the
            # seed, so three replicates sharing tag "draw50" all wrote to
            # draw50_seq.npz and silently overwrote the standard run.  Found
            # 2026-08-29 after the fact: 112 runs executed, yet the directory
            # held 107 bundles of the expected 113 -- 106 written by the batch
            # plus draw23_seq from an earlier check -- and draw50_seq.npz held
            # seed 900.  The stored `seed` in each summary made it detectable.
            tag = "draw%02dr%d" % (k, s)
            out["%s_seq" % tag] = dict(
                d, tag=tag, base_tag="draw%02d" % k, kind="replicate",
                sequential=True, seed=s, expected="")
    for tag, ov, exp, _sub in PS.ANCHORS:
        d = dict(PS.BASE, **ov)
        out["%s_seq" % tag] = dict(d, tag=tag, base_tag=tag, kind="anchor",
                                   sequential=True, seed=0,
                                   expected=PS.expected_for(exp, True))
    _check_keys(out)
    return out


def _check_keys(out):
    """Every dict key must equal the filename the worker will actually write.

    pilot_worker derives the bundle name from tag+config alone, so two rows
    differing only in seed collide and the later one wins -- with no error,
    because overwriting an npz is a normal operation.  This is the guard that
    turns that into an exception at manifest time instead of a corrupted draw
    discovered later.
    """
    seen = {}
    for k, r in out.items():
        got = PW.run_key(r)
        if got != k:
            raise SystemExit("row %r would be written as %r" % (k, got))
        if got in seen:
            raise SystemExit("rows %r and %r both write %r" % (seen[got], k, got))
        seen[got] = k
    return out


def existing():
    """run_key -> path for bundles already on disk, checked for sigma."""
    have = {}
    for p in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        k = os.path.splitext(os.path.basename(p))[0]
        with np.load(p, allow_pickle=True) as z:
            s = json.loads(str(z["summary"]))
        if s.get("lfp_sigma") != SIGMA:
            raise SystemExit(
                "%s was produced at lfp_sigma=%r, not %r -- amplitudes are not "
                "comparable across sigmas. Re-run it or move the file aside."
                % (p, s.get("lfp_sigma"), SIGMA))
        have[k] = p
    return have


def one(key):
    r = rows()[key]
    imgs = RP.mnist_images()
    t0 = time.time()
    s = PW.run_and_reduce_field(r, imgs, n=56, T_end=T_END, outdir=OUT,
                                keep_full=False, sigma=SIGMA)
    return dict(s, key=key, kind=r["kind"], seed=r["seed"],
                wall_s=round(time.time() - t0, 1))


def main(keys=None):
    os.makedirs(OUT, exist_ok=True)
    man = rows()
    want = list(keys) if keys else sorted(man)
    todo = [k for k in want if k not in existing()]
    print("batch4: %d runs requested, %d already on disk, %d to run"
          % (len(want), len(want) - len(todo), len(todo)), flush=True)
    t0 = time.time()
    done = []
    if todo:
        with ProcessPoolExecutor(WORKERS) as ex:
            for i, s in enumerate(ex.map(one, todo), 1):
                done.append(s)
                print("  %3d/%3d %-22s duty %.4f  peak %.3f  %5.1fs"
                      % (i, len(todo), s["key"], s["unit_duty_median"],
                         s["peak_simultaneous_frac"], s["wall_s"]), flush=True)
    wall = time.time() - t0
    mb = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "*.npz"))) / 1e6
    print("\n%d runs in %.0f s (%.1f s effective each at %d workers) | %.0f MB on disk"
          % (len(todo), wall, wall / max(len(todo), 1), WORKERS, mb))
    json.dump(done, open(os.path.join(OUT, "_batch4_runs.json"), "w"), indent=1)
    return done


if __name__ == "__main__":
    main(sys.argv[1:] or None)
