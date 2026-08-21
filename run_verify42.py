"""Full-field re-run of the 42 pilot runs whose scoring is NOT yet confirmed.

Only 12 of the 54 pilot runs have a confirmed score: the ones adjudicated in
liou_fullfield.html (12/12 blind agreement, liou_fullfield_scores_reconciled.csv).
The other 42 carry automated calls that no human has checked.  This driver
re-runs those 42 with the whole 56x56 sheet recorded, so they can be scored from
sheet panels rather than from a 30-electrode montage -- which the full-field
re-run showed misreads 5 of 12 runs, including calling three depolarization-block
runs "quiet".

TWO DELIBERATE DEPARTURES from run_fullfield.py, both required:

  * SIGMA.  Rendered at LFP sigma 0.11 (the sweep decision), not the 0.025
    default every existing scored artifact used.  Amplitudes are ~6x larger and
    the discharge COUNT moves run-specifically, so these panels are NOT
    comparable to the 12 already-scored ones and must not be pooled with them.
  * NO verify_against.  Bit-comparison against pilot_out_v2 is meaningless at a
    different sigma; run_and_reduce_field now raises if both are given.  The
    simulation itself is unchanged and still deterministic -- only the LFP
    readout differs -- so realization identity is guaranteed by the seed, not by
    the montage comparison.

Cost measured on this machine: ~90 s wall per run at 4 workers, so ~16 min for
42, and ~13 MB per bundle, so ~0.55 GB.

    PYTHONPATH=. python run_verify42.py            # all 42
    PYTHONPATH=. python run_verify42.py draw07_seq # named subset
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

OUT = "verify42_out"
SIGMA = 0.11
N_PROC = 4
T_END = 80000.0

# The 12 whose scoring IS confirmed -- excluded here by definition.
CONFIRMED = [
    "anchor_default_seq", "anchor_default_sim",
    "draw16_seq", "draw16_sim",
    "anchor_gIlocal600_seq", "anchor_gIlocal600_sim", "anchor_gKmax120_seq",
    "anchor_gE220_seq", "anchor_gE220_sim", "draw01_seq",
    "draw04_seq", "draw09_sim",
]


def rows_by_key():
    return {PW.run_key(r): r for r in PS.manifest(20)}


def unconfirmed_keys():
    keys = [k for k in sorted(rows_by_key()) if k not in CONFIRMED]
    # Fail loudly on a stale CONFIRMED list rather than silently scoring 43.
    missing = [k for k in CONFIRMED if k not in rows_by_key()]
    if missing:
        raise SystemExit("CONFIRMED names not in the manifest: %s" % missing)
    if len(keys) != 42:
        raise SystemExit("expected 42 unconfirmed runs, got %d" % len(keys))
    return keys


def one(key):
    row = rows_by_key()[key]
    imgs = RP.mnist_images()
    t0 = time.time()
    s = PW.run_and_reduce_field(row, imgs, n=56, T_end=T_END, outdir=OUT,
                                keep_full=False, sigma=SIGMA)
    return dict(s, key=key, wall_measured_s=round(time.time() - t0, 1))


if __name__ == "__main__":
    keys = sys.argv[1:] or unconfirmed_keys()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(N_PROC) as p:
        res = p.map(one, keys)
    for r in sorted(res, key=lambda d: d["key"]):
        print("%-24s %6.1f s %6.1f MB  peak %.3f duty %.3f  clfloor %.4f %s"
              % (r["key"], r["wall_measured_s"], r.get("MB", float("nan")),
                 r["peak_simultaneous_frac"], r["unit_duty_median"],
                 r["cl_floor_frac_units"], "cached" if r.get("cached") else ""),
              flush=True)
    json.dump(res, open("verify42_manifest.json", "w"), indent=1)
    print("TOTAL %.0f s wall for %d runs at sigma %.3f"
          % (time.time() - t0, len(res), SIGMA))
