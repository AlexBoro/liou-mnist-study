"""Re-run the 12 already-scored runs at LFP sigma 0.11, for a uniform re-read.

The 12 full-field runs were scored from sigma-0.025 panels; the 42 verification
runs from sigma-0.11 panels.  A reliability check across all 54 has to hold the
instrument fixed, or a change of call between passes could be the electrode
rather than the scorer -- and at 0.11 the montage is a near-uniform whole-sheet
average where at 0.025 it is a sparse, 625x-uneven sampling, so that is not a
hypothetical difference.

Same settings as run_verify42.py otherwise: 80 s, n=56, keep_full=False,
verify_against off (a 0.11 montage cannot be bit-compared to a 0.025 reference).

    PYTHONPATH=. python run_confirmed12_sigma011.py
"""
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.getcwd())
import las_model.pilot_sweep as PS      # noqa: E402
import las_model.pilot_worker as PW     # noqa: E402
import rerun_pilot as RP                # noqa: E402
from run_verify42 import CONFIRMED, rows_by_key, SIGMA, T_END, N_PROC  # noqa: E402

OUT = "confirmed12_out"


def one(key):
    row = rows_by_key()[key]
    imgs = RP.mnist_images()
    t0 = time.time()
    s = PW.run_and_reduce_field(row, imgs, n=56, T_end=T_END, outdir=OUT,
                               keep_full=False, sigma=SIGMA)
    return dict(s, key=key, wall_measured_s=round(time.time() - t0, 1))


if __name__ == "__main__":
    keys = sys.argv[1:] or list(CONFIRMED)
    missing = [k for k in keys if k not in rows_by_key()]
    if missing:
        raise SystemExit("not in manifest: %s" % missing)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(N_PROC) as p:
        res = p.map(one, keys)
    for r in sorted(res, key=lambda d: d["key"]):
        print("%-24s %6.1f s %6.1f MB  duty %.3f  clfloor %.4f"
              % (r["key"], r["wall_measured_s"], r.get("MB", float("nan")),
                 r["unit_duty_median"], r["cl_floor_frac_units"]), flush=True)
    json.dump(res, open("confirmed12_manifest.json", "w"), indent=1)
    print("TOTAL %.0f s for %d runs at sigma %.3f" % (time.time() - t0, len(res), SIGMA))
