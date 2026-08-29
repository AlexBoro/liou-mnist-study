"""Overview movies for batch 4, written under the blind id.

ONE TIER, NOT TWO, AND THAT IS SUFFICIENT -- established 2026-08-29.  The
overview movie is built from the stored 80 ms envelope, which is below Nyquist
for the ~17 Hz discharge rhythm and so cannot show individual discharges.  It
does not need to: the rubric's criteria are the recruitment wave and the tonic
front, both of which evolve over seconds.  The V3 pass scored all 32 runs from
exactly this tier and produced confident subtype calls describing wave
morphology, which is the existence proof.  A detail tier would require
keep_full=True at ~70 MB per run instead of ~12 MB, for no gain against this
rubric; it is still the right tier for wave-SPEED measurement.

    PYTHONPATH=. python render_batch4_movies.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
import render_movies as RM              # noqa: E402

MOV_DIR = "b4_movies"


def overview_blind(npz_path, blind_id, outdir=MOV_DIR):
    """render_movies.overview(), writing <blind_id>_overview.mp4."""
    with np.load(npz_path, allow_pickle=True) as f:
        env = np.asarray(f["env"], np.float32)
        t_env = np.asarray(f["t_env"], np.float32)
        fmax_hz = float(f["f_max"]) * 1e3        # f_max is per ms
    n = int(round(np.sqrt(env.shape[1])))
    if n * n != env.shape[1]:
        raise SystemExit("%s: %d units is not a square sheet"
                         % (npz_path, env.shape[1]))
    p = os.path.join(outdir, "%s_overview.mp4" % blind_id)
    RM.render_movie(env * fmax_hz, t_env, n, p, fps=RM.FPS, title=blind_id)
    return p, dict(blind_id=blind_id, frames=int(env.shape[0]),
                   MB=round(os.path.getsize(p) / 1e6, 2))


def main():
    import build_batch4_page as B4P
    m = B4P.blind_key()
    b = B4P.bundles(set(m))
    os.makedirs(MOV_DIR, exist_ok=True)
    t0 = time.time()
    metas = []
    for k in sorted(b, key=lambda x: m[x]):
        _p, meta = overview_blind(b[k], m[k])
        metas.append(meta)
        print("  %-8s %4d frames %5.1f MB" % (meta["blind_id"], meta["frames"],
                                              meta["MB"]), flush=True)
    json.dump(metas, open(os.path.join(MOV_DIR, "movies.json"), "w"), indent=1)
    # Nothing in the movie directory may carry a run key: the filename is the
    # only part of a movie the reader can see before playing it.
    bad = [f for f in os.listdir(MOV_DIR)
           if any(t in f for t in ("draw", "anchor", "rep", "seq", "sim"))]
    if bad:
        raise SystemExit("movie filenames leak the run identity: %s" % bad[:5])
    print("\n%d movies, %.1f MB, %.0f s"
          % (len(metas), sum(x["MB"] for x in metas), time.time() - t0))
    return metas


if __name__ == "__main__":
    main()
