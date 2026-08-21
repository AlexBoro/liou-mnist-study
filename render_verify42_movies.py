"""Overview sheet movies for the 42 blind verification runs.

Companion to run_verify42.py / build_verify42_page.py.  The page's movie pane
(`m`) is empty until these exist, because build_fullfield_html._movies() looks
for ``<mov_dir>/<run id>_<kind>.mp4`` and degrades to "No movie for this run"
when the directory is absent.

BLINDING.  render_movies.overview() names its output ``<run_key>_overview.mp4``
and the page puts that path straight into the <video src>, so the pane -- and
the browser's network tab -- would read ``anchor_scale4_sim``.  This driver
passes the blind id instead.

VERIFIED, not assumed: the frames themselves carry ONLY the elapsed-time stamp.
render_movie() accepts a ``title=`` argument and never draws it (checked by
extracting a frame and by reading the function body -- ``title`` appears in the
signature and nowhere else), so nothing is burned into the pixels.  It is passed
here anyway so that the intent is recorded if that parameter is ever wired up,
and test_verify42_movie_files_are_blinded pins the filename property, which is
the one that actually leaks today.

OVERVIEW ONLY.  The detail tier needs ``rate_full`` (the raw 5 ms field), and
run_verify42.py stored none of it (keep_full=False, ~13 MB/run instead of
~94 MB).  The page's detail button therefore stays inactive, and its own note
explains that the overview cannot resolve the ~16.7 Hz discharge rhythm --
80 ms per frame is 0.7 frames per cycle, below Nyquist, so flicker at this
scale is aliasing.  Judge discharge morphology from the LFP montage and the
sheet raster, which are at full stored resolution; the movie is for
recruitment and spread.

    PYTHONPATH=. python render_verify42_movies.py
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
import render_movies as RM      # noqa: E402

NPZ_DIR = "verify42_out"
MOV_DIR = "vf42_movies"
KEY_CSV = "verify42_blind_key.csv"


def blind_key():
    with open(KEY_CSV, newline="", encoding="utf-8") as f:
        return {r["run_key"]: r["blind_id"] for r in csv.DictReader(f)}


def overview_blind(npz_path, lab, outdir=MOV_DIR):
    """render_movies.overview(), but titled and named by the BLIND id."""
    with np.load(npz_path, allow_pickle=True) as f:
        env = np.asarray(f["env"], np.float32)
        t_env = np.asarray(f["t_env"], np.float32)
        n = int(f["n"])
        fmax_hz = float(f["f_max"]) * 1e3
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "%s_overview.mp4" % lab)
    RM.render_movie(env * fmax_hz, t_env, n, p, fps=RM.FPS, title=lab)
    return p, dict(blind_id=lab, kind="overview", frames=int(env.shape[0]),
                   dt_ms=float((t_env[1] - t_env[0]) * 1e3), fps=RM.FPS,
                   MB=round(os.path.getsize(p) / 1e6, 2))


def main():
    m = blind_key()
    metas = []
    t0 = time.time()
    for rk in sorted(m):
        npz = os.path.join(NPZ_DIR, rk + ".npz")
        if not os.path.exists(npz):
            raise SystemExit("missing bundle: %s" % npz)
        _p, meta = overview_blind(npz, m[rk])
        metas.append(meta)
        print("%s  %5.2f MB  %d frames" % (meta["blind_id"], meta["MB"],
                                           meta["frames"]), flush=True)
    json.dump(metas, open(os.path.join(MOV_DIR, "movies.json"), "w"), indent=1)

    # No movie file may carry a real run key in its name.
    bad = [f for f in os.listdir(MOV_DIR)
           if f.endswith(".mp4") and not f[0] == "V"]
    if bad:
        raise SystemExit("un-blinded movie filenames: %s" % bad)
    print("TOTAL %.0f s for %d movies, %.1f MB"
          % (time.time() - t0, len(metas), sum(x["MB"] for x in metas)))


if __name__ == "__main__":
    main()
