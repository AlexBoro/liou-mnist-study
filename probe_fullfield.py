"""Scoping probe for the full-field re-run: determinism + storage cost.

Two questions the plan depends on and that cannot be answered from the stored
npz (which keeps only the 30-channel montage):

1. DETERMINISM.  rerun_pilot.py claims the pilot reproduces bit-for-bit.  If it
   does not, a full-field panel would describe a DIFFERENT realization than the
   montage that was scored, and every figure would be quietly mislabelled.  The
   check re-runs one row and compares the recomputed montage to the stored one.

2. STORAGE.  The full field is (T, n*n) = (16000, 3136) float32 = 200 MB/run
   raw.  Measured here at four levels of reduction so the retention policy is
   chosen from numbers rather than guesses.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
import las_model.pilot_sweep as PS
import las_model.readout as RO
import rerun_pilot as RP

KEY = sys.argv[1] if len(sys.argv) > 1 else "draw16_seq"


def main():
    rows = {("%s_%s" % (r["tag"], "seq" if r["sequential"] else "sim")): r
            for r in PS.manifest(20)}
    row = rows[KEY]
    imgs = RP.mnist_images()
    d = {k: row[k] for k in row if k in PS.BASE}

    t0 = time.time()
    z = PS.run_one(imgs, d, sequential=row["sequential"], n=56, T_end=80000.0,
                   seed=row["seed"])
    wall = time.time() - t0

    L, t_ = RO.lfp_montage(z, n=56)
    with np.load("pilot_out_v2/%s.npz" % KEY, allow_pickle=True) as f:
        L0 = f["L"]
    # NaN-aware: the stored montage carries NaN in its FIRST sample on all 30
    # channels (an edge effect of the LFP filter at t=0, 1/16000 rows), so a
    # plain array_equal reports False and a plain max-abs-diff reports nan even
    # for a bit-identical reproduction.  Compare NaN positions and finite values
    # separately.
    Lf = L.astype(np.float32)
    nan_a, nan_b = np.isnan(Lf), np.isnan(L0)
    same = bool(np.array_equal(nan_a, nan_b) and
                np.array_equal(Lf[~nan_a], L0[~nan_b]))
    dmax = float(np.max(np.abs(Lf[~nan_a] - L0[~nan_b]))) if same or nan_a.sum() == nan_b.sum() else float("nan")
    out_nan = dict(stored_nan=int(nan_b.sum()), rerun_nan=int(nan_a.sum()))

    rate = np.asarray(z["rate"], dtype=np.float32)   # (T, n*n), normalised to f_max
    T, U = rate.shape
    out = dict(key=KEY, wall_s=round(wall, 1), montage_bit_identical=bool(same),
               montage_max_abs_diff=dmax, shape=[int(T), int(U)],
               raw_MB=round(rate.nbytes / 1e6, 1), **out_nan)

    # (a) full rate, compressed float32
    np.savez_compressed("_probe_f32.npz", rate=rate)
    out["npz_f32_MB"] = round(os.path.getsize("_probe_f32.npz") / 1e6, 1)
    # (b) full rate, float16 -- rate is normalised to [0,1] so f16 gives ~3 decimal digits
    r16 = rate.astype(np.float16)
    np.savez_compressed("_probe_f16.npz", rate=r16)
    out["npz_f16_MB"] = round(os.path.getsize("_probe_f16.npz") / 1e6, 1)
    out["f16_max_abs_err"] = float(np.max(np.abs(r16.astype(np.float32) - rate)))
    # (c) decimated to 50 Hz (every 4th sample of the 200 Hz record)
    np.savez_compressed("_probe_50hz.npz", rate=rate[::4].astype(np.float16))
    out["npz_50Hz_f16_MB"] = round(os.path.getsize("_probe_50hz.npz") / 1e6, 1)
    # (d) 80 ms envelope, matching the existing heat lane's bin width
    dt_s = float(t_[1] - t_[0])          # 0.005 s -- t is in SECONDS, not ms
    b = max(1, int(round(0.080 / dt_s)))  # 80 ms envelope bin = 16 frames
    nb = T // b
    env = rate[: nb * b].reshape(nb, b, U).max(axis=1).astype(np.float16)
    np.savez_compressed("_probe_env.npz", env=env)
    out["dt_s"] = dt_s
    out["env_bin_frames"] = int(b)
    out["env_shape"] = [int(nb), int(U)]
    out["npz_env80ms_MB"] = round(os.path.getsize("_probe_env.npz") / 1e6, 1)

    # how much of the sheet the montage can actually see, for the same run
    chs, dmask = RO.driven_channels(n=56)
    out["n_driven_ch"] = int(len(chs))
    out["field_max_rate"] = float(rate.max())
    out["field_frac_units_ever_active"] = float((rate.max(axis=0) > 0.10).mean())

    json.dump(out, open("probe_fullfield.json", "w"), indent=1)
    for k, v in out.items():
        print("%-28s %s" % (k, v))


if __name__ == "__main__":
    main()
