"""The 50-draw paired parameter exploration: 100 runs, classified and gated.

DESIGN, as settled 2026-08-19
-----------------------------
50 parameter draws x {sequential, simultaneous} = 100 runs.  The paired
allocation was chosen deliberately over 100 single-config draws: it halves the
parameter coverage (50 independent points, ~0.62 power on the weakest ranked
axis) in exchange for keeping the configuration contrast, which is a question in
its own right and the only axis on which chloride clamping differs at all.

ADMISSIBILITY IS STRICT.  A run is inadmissible if `cl_floor_frac_units > 0` --
if ANY unit reached the 0.1 mM chloride floor at ANY timestep.  A unit at the
floor is not described by Liou Eq. 1; it is held by a numerical guard at
E_Cl = -187 mV against -77.7 mV at rest, so the run did not test the equations.
Runs are still simulated and stored (clamping cannot be predicted from the
parameters) and are marked inadmissible at ANALYSIS time, so the rule is
auditable and can be revisited without re-simulating.

  Measured cost of this rule on the 54 pilot runs: 11 excluded, ALL of them
  simultaneous (0 of 27 sequential), breaking 11 of 20 pairs.  The exclusion is
  therefore NON-RANDOM with respect to configuration, and any paired analysis
  must report the admissible pair count, not the drawn one.  `sweep50_labels.csv`
  carries `admissible` and `pair_admissible` so this cannot be overlooked.

WHAT IS REUSED
--------------
draw(k) is seeded from 1000+k alone, independent of the draw count, so draws
0-19 of this sweep ARE the pilot draws -- verified at import, not assumed.  Those
40 runs plus the 14 anchors already exist at sigma 0.11 in verify42_out/ and
confirmed12_out/, so only draws 20-49 are simulated here: 60 runs, ~25 min at 4
workers, ~0.85 GB.  A reused bundle whose stored `lfp_sigma` is not SIGMA is
refused rather than silently pooled.

THE ANCHOR GATE
---------------
The 14 anchor runs have known verdicts.  They are checked BEFORE any draw label
is written, and a mismatch is fatal: the anchors are the only runs whose answers
are known independently, so if they disagree the detector is wrong and every
draw label in the batch is worthless.  Pass --force to write labels anyway (for
diagnosing the mismatch itself); the output is then stamped anchors_failed.

KNOWN CONFOUND, not fixed here
------------------------------
pilot_sweep.draw sets g_I_global = (50, 300)[k % 2], tying the two-level control
arm to draw-index parity.  At 50 draws that is balanced 25/25 but perfectly
confounded with parity, so g_I_global cannot be separated from anything else that
happens to alternate with k.  Changing it would break the "first 20 draws are the
pilot draws" property that makes the pilot a free subset, so it is left alone and
recorded in the output header instead.

    PYTHONPATH=. python run_sweep50.py              # simulate what is missing, then label
    PYTHONPATH=. python run_sweep50.py --label-only  # relabel from existing bundles
    PYTHONPATH=. python run_sweep50.py --force       # write labels despite an anchor mismatch
"""
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
import las_model.pilot_sweep as PS      # noqa: E402
import las_model.pilot_worker as PW     # noqa: E402
from las_model import classify as CL    # noqa: E402
import rerun_pilot as RP                # noqa: E402

N_DRAWS = 50
SIGMA = 0.11
T_END = 80000.0
N_PROC = 4
OUT = "sweep50_out"
REUSE_DIRS = ("verify42_out", "confirmed12_out", OUT)
LABELS = "sweep50_labels.csv"
MANIFEST = "sweep50_manifest.json"


def rows_by_key(n_draws=N_DRAWS):
    return {PW.run_key(r): r for r in PS.manifest(n_draws)}


def assert_pilot_is_a_subset():
    """draws 0-19 of this sweep must be byte-identical to the pilot's.

    The whole reuse argument rests on it, and it is one line to check.
    """
    a, b = rows_by_key(20), rows_by_key(N_DRAWS)
    for k, row in a.items():
        if row["kind"] != "draw":
            continue
        for p, _lo, _hi in PS.SPEC:
            if row[p] != b[k][p]:
                raise SystemExit("draw reuse broken: %s differs in %s" % (k, p))
        if row["g_I_global"] != b[k]["g_I_global"]:
            raise SystemExit("draw reuse broken: %s differs in g_I_global" % k)


def existing():
    """run_key -> (path, summary) for every bundle already on disk at SIGMA."""
    out = {}
    for d in REUSE_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            k = os.path.splitext(os.path.basename(p))[0]
            with np.load(p, allow_pickle=True) as z:
                s = json.loads(str(z["summary"]))
            sig = s.get("lfp_sigma")
            if sig != SIGMA:
                raise SystemExit(
                    "%s was produced at lfp_sigma=%r, not %r. Amplitude and "
                    "discharge count are not comparable across sigmas, so this "
                    "bundle cannot join the sweep -- re-run it or drop the dir."
                    % (p, sig, SIGMA))
            out[k] = (p, s)
    return out


def one(key):
    row = rows_by_key()[key]
    imgs = RP.mnist_images()
    t0 = time.time()
    s = PW.run_and_reduce_field(row, imgs, n=56, T_end=T_END, outdir=OUT,
                                keep_full=False, sigma=SIGMA)
    return dict(s, key=key, wall_measured_s=round(time.time() - t0, 1))


def label_all(have):
    """Classify every run and apply the admissibility rule.  Returns rows."""
    man = rows_by_key()
    rows = []
    for k in sorted(man):
        row, (_p, s) = man[k], have[k]
        cls, margin = CL.sheet(s["unit_duty_median"])
        clamped_units = float(s["cl_floor_frac_units"])
        rows.append(dict(
            run_key=k, tag=row["tag"], kind=row["kind"],
            config="sequential" if row["sequential"] else "simultaneous",
            draw=int(row["tag"][4:]) if row["kind"] == "draw" else -1,
            sheet_class=cls, margin=round(margin, 4),
            unresolved=int(CL.is_unresolved(s["unit_duty_median"])),
            unit_duty_median=s["unit_duty_median"],
            peak_simultaneous_frac=s["peak_simultaneous_frac"],
            cl_floor_frac_units=clamped_units,
            cl_floor_frac_steps=float(s["cl_floor_frac_steps"]),
            # THE RULE: any unit ever at the floor makes the run inadmissible.
            admissible=int(clamped_units == 0.0),
            expected=row.get("expected") or "",
            expected_subtype=row.get("expected_subtype") or "",
            lfp_sigma=s["lfp_sigma"],
            **{p: row[p] for p, _l, _h in PS.SPEC},
            g_I_global=row["g_I_global"],
        ))
    # A pair is usable only if BOTH configurations are admissible.  Computed
    # here rather than left to the analyst: the exclusion is config-asymmetric,
    # so counting admissible RUNS overstates the paired sample.
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    for tag, rs in by_tag.items():
        ok = int(len(rs) == 2 and all(x["admissible"] for x in rs))
        for r in rs:
            r["pair_admissible"] = ok
    return rows


def anchor_gate(rows):
    """Check every anchor against its known verdict.  Returns failure messages."""
    msgs = []
    for r in rows:
        if r["kind"] != "anchor":
            continue
        m = PS.check_anchor(r["run_key"], r["expected"], r["sheet_class"])
        if m:
            msgs.append(m)
    return msgs


def main(argv):
    label_only = "--label-only" in argv
    force = "--force" in argv
    assert_pilot_is_a_subset()

    man = rows_by_key()
    have = existing()
    missing = [k for k in sorted(man) if k not in have]

    if missing and not label_only:
        os.makedirs(OUT, exist_ok=True)
        print("simulating %d missing runs at sigma %.3f (%d already on disk)"
              % (len(missing), SIGMA, len(have)), flush=True)
        t0 = time.time()
        with Pool(N_PROC) as p:
            res = p.map(one, missing)
        for r in sorted(res, key=lambda d: d["key"]):
            print("  %-14s %6.1f s %6.1f MB  duty %.4f  clfloor %.5f%s"
                  % (r["key"], r["wall_measured_s"], r.get("MB", float("nan")),
                     r["unit_duty_median"], r["cl_floor_frac_units"],
                     "  INADMISSIBLE" if r["cl_floor_frac_units"] else ""),
                  flush=True)
        json.dump(res, open(MANIFEST, "w"), indent=1)
        print("simulated %d runs in %.0f s" % (len(res), time.time() - t0))
        have = existing()
        missing = [k for k in sorted(man) if k not in have]
    if missing:
        raise SystemExit("%d runs still missing: %s" % (len(missing), missing[:6]))

    rows = label_all(have)
    fails = anchor_gate(rows)
    if fails:
        for m in fails:
            print("\n" + m, file=sys.stderr)
        if not force:
            raise SystemExit(
                "\nANCHOR GATE FAILED (%d of 14) -- no labels written. The "
                "anchors are the detector's independent check, so every draw "
                "label in this batch is suspect. Re-run with --force only to "
                "diagnose the mismatch itself." % len(fails))
        print("\n--force: writing labels despite %d anchor failures" % len(fails),
              file=sys.stderr)

    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dr = [r for r in rows if r["kind"] == "draw"]
    adm = [r for r in dr if r["admissible"]]
    pairs = len({r["tag"] for r in dr})
    pairs_ok = len({r["tag"] for r in dr if r["pair_admissible"]})
    print("\n%s : %d runs (%d draws + %d anchors), anchors %s"
          % (LABELS, len(rows), len(dr), len(rows) - len(dr),
             "FAILED" if fails else "all 14 pass"))
    print("admissible runs      : %d of %d  (%d excluded by the chloride-floor rule)"
          % (len(adm), len(dr), len(dr) - len(adm)))
    print("admissible PAIRS     : %d of %d  <- the paired sample size" % (pairs_ok, pairs))
    for c in CL.CLASSES:
        n_all = sum(1 for r in dr if r["sheet_class"] == c)
        n_adm = sum(1 for r in adm if r["sheet_class"] == c)
        print("  %-17s %3d drawn / %3d admissible" % (c, n_all, n_adm))
    unres = [r["run_key"] for r in adm if r["unresolved"]]
    print("near a threshold (unresolved): %d%s"
          % (len(unres), (" -- " + ", ".join(unres[:8])) if unres else ""))
    print("\nNOTE: g_I_global is tied to draw parity ((50,300)[k %% 2]), balanced "
          "25/25 but confounded with draw index parity.")


if __name__ == "__main__":
    main(sys.argv[1:])
