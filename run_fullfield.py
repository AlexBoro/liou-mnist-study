"""Re-run a subset of the pilot with FULL 56x56 field recording.

Selection (12 runs, all four automated classes, both presentation configs):

  ICTAL            anchor_default_{seq,sim}, draw16_{seq,sim}
  ENTRAINED        anchor_gIlocal600_{seq,sim}, anchor_gKmax120_seq
  QUIET_OR_BLOCK   anchor_gE220_{seq,sim}, draw01_seq
  ACTIVE_NONICTAL  draw04_seq, draw09_sim

The anchors carry the parameter contrasts (the four ANCHORS that bracket the
seizure boundary); the draws are the cases the montage reads as extreme --
draw16 saturates the colour ceiling, draw01 reads as near-flat.

Every run is verified against the already-scored montage in pilot_out_v2 and
ABORTS on mismatch: if the re-run were a different realization, the full-field
panels would not describe the data that was adjudicated.

Workers: 4.  Each holds the (16000, 3136) rate field plus EPSC/IPSC of the same
shape (~2.4 GB total measured), against ~14 GB available.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.getcwd())
import las_model.pilot_sweep as PS
import las_model.pilot_worker as PW
import rerun_pilot as RP

OUT = "fullfield_out"
REF = "pilot_out_v2"
N_PROC = 4

SELECTED = [
    "anchor_default_seq", "anchor_default_sim",
    "draw16_seq", "draw16_sim",
    "anchor_gIlocal600_seq", "anchor_gIlocal600_sim", "anchor_gKmax120_seq",
    "anchor_gE220_seq", "anchor_gE220_sim", "draw01_seq",
    "draw04_seq", "draw09_sim",
]
# ALL runs keep their raw (16000, 3136) field as of 2026-08-16.
#
# It was one run (anchor_default_seq) when the raw field was needed only to audit
# the envelope reduction and to give wave-speed analysis full temporal resolution
# on a single seizure.  The user then asked for discharge-resolving movies of
# every run, and that is not possible from the 80 ms envelope: the discharge
# rhythm is ~16.7 Hz, so the envelope samples it at 0.75 frames per cycle, below
# Nyquist.  A detail movie needs <=30 ms sampling, which only the raw field has.
#
# Cost measured on anchor_default_seq: ~94 MB compressed per run (not the ~200 MB
# the raw array occupies in memory), so 12 runs is ~1.1 GB rather than 227 MB.
#
# NOTE ON RE-RUNNING: run_and_reduce_field short-circuits to cached=True if the
# output npz already exists, so an existing envelope-only file will NEVER gain a
# raw field in place.  The old files must be moved aside first (they are kept in
# fullfield_out_envonly/ rather than deleted).  verify_against is left ON, so a
# re-run that does not reproduce the scored montage bit-for-bit aborts instead of
# silently substituting a different realization.
KEEP_FULL = set(SELECTED)


def rows_by_key():
    return {PW.run_key(r): r for r in PS.manifest(20)}


def one(key):
    row = rows_by_key()[key]
    imgs = RP.mnist_images()
    t0 = time.time()
    s = PW.run_and_reduce_field(row, imgs, n=56, T_end=80000.0, outdir=OUT,
                                keep_full=(key in KEEP_FULL), verify_against=REF)
    return dict(s, key=key, wall_measured_s=round(time.time() - t0, 1))


if __name__ == "__main__":
    keys = sys.argv[1:] or SELECTED
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(N_PROC) as p:
        res = p.map(one, keys)
    for r in sorted(res, key=lambda d: d["key"]):
        print("%-24s %6.1f s  %6.1f MB  peak_simult %.3f  duty %.3f  %s"
              % (r["key"], r["wall_measured_s"], r.get("MB", float("nan")),
                 r["peak_simultaneous_frac"], r["unit_duty_median"],
                 "cached" if r.get("cached") else ""), flush=True)
    json.dump(res, open("fullfield_manifest.json", "w"), indent=1)
    print("TOTAL %.0f s wall for %d runs" % (time.time() - t0, len(res)))
