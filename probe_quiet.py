"""Is a QUIET_OR_BLOCK run quiet on the SHEET, or only at the 30 electrodes?

'Fraction of units ever above 10%' is a weak statistic over an 80 s run -- a
single transient qualifies a unit.  This measures the quantities that actually
distinguish a quiescent sheet from an active one, for one run of each automated
class, using identical code so the classes are directly comparable:

  * peak SIMULTANEOUS recruitment -- max over time of the fraction of units
    above threshold, which a transient cannot inflate;
  * duty cycle -- median over units of the fraction of TIME above threshold;
  * what the 30-electrode montage would report for the same run, so the gap
    between 'what the sheet did' and 'what we recorded' is quantified rather
    than asserted.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
import las_model.pilot_sweep as PS
import las_model.readout as RO
import rerun_pilot as RP

THR = 0.10   # same 10% of f_max used by the existing territory metrics


def main(keys):
    rows = {("%s_%s" % (r["tag"], "seq" if r["sequential"] else "sim")): r
            for r in PS.manifest(20)}
    imgs = RP.mnist_images()
    out = []
    for key in keys:
        row = rows[key]
        d = {k: row[k] for k in row if k in PS.BASE}
        z = PS.run_one(imgs, d, sequential=row["sequential"], n=56,
                       T_end=80000.0, seed=row["seed"])
        rate = np.asarray(z["rate"], dtype=np.float32)      # (T, 3136), /f_max
        act = rate > THR
        chs, dmask = RO.driven_channels(n=56)
        frac_t = act.mean(axis=1)            # fraction of sheet active per frame
        rec = dict(
            key=key,
            peak_simultaneous_frac=float(frac_t.max()),
            median_simultaneous_frac=float(np.median(frac_t)),
            unit_duty_median=float(np.median(act.mean(axis=0))),
            unit_duty_p90=float(np.percentile(act.mean(axis=0), 90)),
            frac_units_ever=float(act.any(axis=0).mean()),
            frac_units_duty_over_1pct=float((act.mean(axis=0) > 0.01).mean()),
            field_mean_rate=float(rate.mean()),
            # what the montage sees, same run
            montage_sites_ever=float(act[:, chs].any(axis=0).mean()),
        )
        out.append(rec)
        del z, rate, act
        print(json.dumps(rec), flush=True)
    json.dump(out, open("probe_quiet.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
