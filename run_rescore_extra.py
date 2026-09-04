"""Re-run the four discrepant runs whose bundles were never archived.

draw14 and draw32, both configurations.  They come from the 50-draw sweep, whose
bundles were swept without being saved -- the runs are deterministic, so this
reproduces them rather than approximating them.

SEED = DRAW INDEX, and both configurations of a draw SHARE it: that is the
paired design pilot_sweep.manifest implements, and it is why a config pair is a
meaningful reliability check (same parameters, same wiring, different
presentation).  Getting this wrong produces a valid run of the right parameters
at the wrong wiring, which is indistinguishable from a real one by eye.

    PYTHONPATH=. python run_rescore_extra.py
"""
from __future__ import annotations

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import las_model.pilot_sweep as PS
import las_model.pilot_worker as PW
import rerun_pilot as RP

OUT = "rescore_out"
SIGMA = 0.11
T_END = 80000.0
WORKERS = 4
DRAWS = (14, 32)


def rows():
    out = {}
    for k in DRAWS:
        d = PS.draw(k)
        for seq in (True, False):
            r = dict(d, tag="draw%02d" % k, base_tag="draw%02d" % k, kind="draw",
                     sequential=seq, seed=k, expected="")
            key = PW.run_key(r)
            if key in out:
                raise SystemExit("run_key collision on %r" % key)
            out[key] = r
    for key, r in out.items():
        if PW.run_key(r) != key:
            raise SystemExit("row %r would be written as %r" % (key, PW.run_key(r)))
    return out


def one(key):
    r = rows()[key]
    t0 = time.time()
    s = PW.run_and_reduce_field(r, RP.mnist_images(), n=56, T_end=T_END,
                                outdir=OUT, keep_full=False, sigma=SIGMA)
    return dict(s, key=key, wall_s=round(time.time() - t0, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    man = rows()
    todo = [k for k in sorted(man)
            if not os.path.exists(os.path.join(OUT, k + ".npz"))]
    print("rescore extras: %d of %d to run" % (len(todo), len(man)), flush=True)
    done = []
    if todo:
        with ProcessPoolExecutor(WORKERS) as ex:
            for s in ex.map(one, todo):
                done.append(s)
                print("  %-14s duty %.4f  peak %.3f  %5.1fs"
                      % (s["key"], s["unit_duty_median"],
                         s["peak_simultaneous_frac"], s["wall_s"]), flush=True)
    json.dump(done, open(os.path.join(OUT, "_runs.json"), "w"), indent=1)
    return done


if __name__ == "__main__":
    main()
