"""Overview movies for the re-scoring pass, written under the blind id.

ONE TIER, AND THAT IS SUFFICIENT.  Established 2026-08-29 and re-confirmed by
the batch-4 pass: the overview movie is built from the stored 80 ms envelope,
which is below Nyquist for the ~17 Hz discharge rhythm and so cannot resolve
individual discharges -- but the rubric's criteria are the RECRUITMENT WAVE and
the TONIC FRONT, which evolve over seconds.  The user scored 32 V3 runs and 112
batch-4 runs from exactly this tier, with confident subtype calls.

Takes the bundle map and blind map from the caller rather than rebuilding them,
so the movies and the panels cannot disagree about which id is which run.

    imported by build_rescore_page.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import render_movies as RM

MOV_DIR = "r5_movies"


def overview_blind(npz_path, blind_id, outdir=MOV_DIR):
    """render_movies.overview(), writing <blind_id>_overview.mp4."""
    with np.load(npz_path, allow_pickle=True) as z:
        env = np.asarray(z["env"], dtype=np.float32)
        t_env = np.asarray(z["t_env"], dtype=np.float32)
        n = int(z["n"])
        fmax_hz = float(z["f_max"]) * 1e3
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "%s_overview.mp4" % blind_id)
    RM.render_movie(env * fmax_hz, t_env, n, p, fps=RM.FPS, title=blind_id)
    return p, dict(blind_id=blind_id, frames=int(env.shape[0]),
                   MB=round(os.path.getsize(p) / 1e6, 2))


def main(npz, blind, outdir=MOV_DIR):
    t0 = time.time()
    metas = []
    for k in sorted(npz, key=lambda x: blind[x]):
        _p, meta = overview_blind(npz[k], blind[k], outdir)
        metas.append(meta)
        print("  %-7s %4d frames %5.2f MB"
              % (meta["blind_id"], meta["frames"], meta["MB"]), flush=True)
    bad = [m["blind_id"] for m in metas
           if any(t in m["blind_id"] for t in ("draw", "anchor", "rep", "seq", "sim"))]
    if bad:
        raise SystemExit("movie filenames leak the run identity: %s" % bad[:5])
    json.dump(metas, open(os.path.join(outdir, "movies.json"), "w"), indent=1)
    print("\n%d movies, %.1f MB, %.0f s"
          % (len(metas), sum(m["MB"] for m in metas), time.time() - t0))
    return metas
