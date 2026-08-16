"""Pilot worker: run one configuration, persist the montage, discard the big arrays.

Each run holds a (16000, 3136) rate array (~0.4 GB).  The worker keeps only the
(T, 30) LFP montage, the field-mean slow states, and the morphology metrics --
about 4 MB -- so a multiprocess pool stays inside RAM.
"""
import gc
import json
import os
import time

import numpy as np

from . import pilot_sweep as PS
from . import readout as RO

OUT = "pilot_out"


def run_key(row):
    return "%s_%s" % (row["tag"], "seq" if row["sequential"] else "sim")


# ---------------------------------------------------------------- full field
# WHY THIS EXISTS (2026-08-14, at the reviewer's request): the montage samples
# 30 sites on a 3136-unit sheet (0.96%), so a flat montage bounds only the
# RECORDED sites and cannot prove the sheet was quiet -- nor show whether an
# event is focal or generalized.  Measured on the pilot: the two QUIET_OR_BLOCK
# anchors have ~100% of units above threshold ~99% of the time.  The montage is
# not merely under-sampling those runs, it is misreading them.
#
# No simulation change is needed: run_mnist already computes the full
# (16000, 3136) rate field every run and this worker used to discard it.
FIELD_THR = 0.10        # fraction of f_max counted as "recruited"; the same
                        # 10% threshold the port's territory metrics already use
FIELD_BIN_S = 0.080     # envelope bin; matches the adjudication heat lane
FIELD_EVERY = 200       # slow-state stride in steps: 200 ms (5 Hz) at the
                        # default dt=1.0 ms; tau_Cl=5000 ms so still 25x
                        # oversampled.  t_field is in MILLISECONDS.


def field_envelope(rate, t, bin_s=FIELD_BIN_S):
    """Max-pool the rate field into bin_s bins.

    MAX, not mean: the question the panel answers is "was this unit recruited",
    and averaging a 3 Hz discharge train over an 80 ms bin dilutes a real spike
    toward the silent baseline.  Max preserves peak simultaneous recruitment to
    within one bin, which the regression test asserts.
    """
    rate = np.asarray(rate)
    dt_s = float(t[1] - t[0])
    b = max(1, int(round(bin_s / dt_s)))
    nb = rate.shape[0] // b
    if nb == 0:
        return rate.astype(np.float16), np.asarray(t, dtype=np.float32), b
    env = rate[: nb * b].reshape(nb, b, rate.shape[1]).max(axis=1)
    tb = (np.arange(nb) + 0.5) * b * dt_s
    return env.astype(np.float16), tb.astype(np.float32), b


def run_and_reduce_field(row, images, *, n=56, T_end=80000.0,
                         outdir="fullfield_out", keep_full=False,
                         verify_against=None):
    """Run one row and persist the FULL SHEET alongside the usual montage.

    Deliberately a separate entry point from run_and_reduce: the 54-run
    adjudication pipeline is scored and must keep producing byte-identical
    outputs, so nothing here touches that path.

    ``verify_against`` is a directory of the previously stored runs.  When
    given, the recomputed montage is compared to the stored one and the run
    ABORTS on mismatch -- if the re-run is not the same realization, a
    full-field panel would silently describe different data than the montage
    that was scored.  The comparison is NaN-aware: the stored montage carries
    NaN in its first sample on all 30 channels (an LFP filter edge effect).
    """
    key = run_key(row)
    path = os.path.join(outdir, key + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as f:
            return dict(json.loads(str(f["summary"])), cached=True)

    t0 = time.time()
    d = {k: row[k] for k in row if k in PS.BASE}
    z = PS.run_one(images, d, sequential=row["sequential"], n=n, T_end=T_end,
                   seed=row["seed"], field_every=FIELD_EVERY)

    L, t_ = RO.lfp_montage(z, n=n)
    chs, dmask = RO.driven_channels(n=n)

    if verify_against:
        ref = os.path.join(verify_against, key + ".npz")
        with np.load(ref, allow_pickle=True) as f:
            L0 = f["L"]
        Lf = L.astype(np.float32)
        na, nb_ = np.isnan(Lf), np.isnan(L0)
        ok = (np.array_equal(na, nb_) and np.array_equal(Lf[~na], L0[~nb_]))
        if not ok:
            raise AssertionError(
                "%s: re-run montage differs from %s -- the full field would "
                "describe a different realization than the scored montage" % (key, ref))

    rate = np.asarray(z["rate"], dtype=np.float32)          # (T, n*n), /f_max
    f_max = float(z["meta"]["f_max"])
    env, t_env, nbin = field_envelope(rate, t_)

    # sheet-truth metrics: what the montage cannot see
    act = rate > FIELD_THR
    frac_t = act.mean(axis=1)
    duty = act.mean(axis=0)
    sheet = dict(
        peak_simultaneous_frac=float(frac_t.max()),
        median_simultaneous_frac=float(np.median(frac_t)),
        unit_duty_median=float(np.median(duty)),
        unit_duty_p90=float(np.percentile(duty, 90)),
        frac_units_ever=float(act.any(axis=0).mean()),
        frac_units_duty_over_1pct=float((duty > 0.01).mean()),
        # f_max = 1/T_refractory with T_refractory in MILLISECONDS (5 ms), so
        # f_max = 0.2 /ms = 200 Hz.  The 1e3 converts /ms to Hz; without it a
        # fully-recruited unit reads as 0.2 Hz instead of 200 Hz.
        field_mean_rate_hz=float(rate.mean() * f_max * 1e3),
        field_max_rate_hz=float(rate.max() * f_max * 1e3),
    )
    del act, duty

    rate_trace = rate.mean(axis=1).astype(np.float32)
    # the port's own warning: tonic-to-clonic detection needs the MAXIMUM rate
    # trace, not the mean -- the mean is diluted by growing silent territory.
    rate_max_trace = rate.max(axis=1).astype(np.float32)

    payload = dict(
        L=L.astype(np.float32), t=t_.astype(np.float32), driven=dmask,
        env=env, t_env=t_env, env_bin_frames=np.int32(nbin),
        rate_trace=rate_trace, rate_max_trace=rate_max_trace,
        Cl_in=np.asarray(z["Cl_in"], dtype=np.float32),
        g_K=np.asarray(z["g_K"], dtype=np.float32),
        f_max=np.float32(f_max), n=np.int32(n),
    )
    for k_src, k_dst in (("Cl_in_field", "Cl_field"), ("g_K_field", "gK_field"),
                         ("t_field", "t_field")):
        if k_src in z:
            payload[k_dst] = np.asarray(z[k_src], dtype=np.float32)
    if keep_full:
        payload["rate_full"] = rate.astype(np.float16)

    summary = dict(
        key=key, tag=row["tag"], kind=row["kind"], expected=row["expected"],
        sequential=bool(row["sequential"]), seed=int(row["seed"]),
        wall_s=round(time.time() - t0, 1), n_driven_ch=int(len(chs)),
        f_max=f_max, env_bin_s=FIELD_BIN_S, field_thr=FIELD_THR,
        keep_full=bool(keep_full), verified=bool(verify_against),
        **sheet, **{k: float(row[k]) for k in PS.BASE},
    )
    payload["summary"] = json.dumps(summary)

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(path, **payload)
    summary["MB"] = round(os.path.getsize(path) / 1e6, 1)
    del z, rate, payload, L
    gc.collect()
    return dict(summary, cached=False)


def run_and_reduce(row, images, *, n=56, T_end=80000.0, outdir=OUT):
    """Execute one pilot row.  Returns a small dict; writes an npz per run."""
    key = run_key(row)
    path = os.path.join(outdir, key + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as f:
            return dict(json.loads(str(f["summary"])), cached=True)

    t0 = time.time()
    d = {k: row[k] for k in row if k in PS.BASE}
    z = PS.run_one(images, d, sequential=row["sequential"], n=n, T_end=T_end,
                   seed=row["seed"])

    L, t_ = RO.lfp_montage(z, n=n)
    chs, dmask = RO.driven_channels(n=n)
    a, b = 0.55 * t_[-1], t_[-1]
    met, _ = RO.morph_driven(z, n=n, a=a, b=b)
    # early window too: nucleation latency needs the pre-0.55 segment
    met_early = RO.morph(L, t_, chs, 0.0, 0.55 * t_[-1], per_scan=False)

    rate = np.asarray(z["rate"])
    slow = dict(
        rate_mean=float(np.mean(rate)),
        rate_trace=rate.mean(axis=1).astype(np.float32),
        Cl_in=np.asarray(z["Cl_in"], dtype=np.float32),
        g_K=np.asarray(z["g_K"], dtype=np.float32),
    )
    del z, rate
    gc.collect()

    summary = dict(
        key=key, tag=row["tag"], kind=row["kind"], expected=row["expected"],
        sequential=bool(row["sequential"]), seed=int(row["seed"]),
        wall_s=round(time.time() - t0, 1), n_driven_ch=int(len(chs)),
        rate_mean=slow["rate_mean"],
        Cl_early=float(np.mean(slow["Cl_in"][: len(slow["Cl_in"]) // 4])),
        Cl_late=float(np.mean(slow["Cl_in"][-len(slow["Cl_in"]) // 4 :])),
        **{k: (None if not np.isfinite(v) else float(v)) for k, v in met.items()},
        **{"early_" + k: (None if not np.isfinite(v) else float(v))
           for k, v in met_early.items()},
        **{k: float(row[k]) for k in PS.BASE},
    )

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        path, L=L.astype(np.float32), t=t_.astype(np.float32),
        driven=dmask, rate_trace=slow["rate_trace"],
        Cl_in=slow["Cl_in"], g_K=slow["g_K"],
        summary=json.dumps(summary),
    )
    del L, slow
    gc.collect()
    return dict(summary, cached=False)
