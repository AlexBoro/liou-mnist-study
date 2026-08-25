"""Overview sheet movies for the 32 pass-3 runs, named by the V3 blind id.

Same blinding constraint as render_verify42_movies / render_pass2_movies:
render_movies.overview() names its output by run key and the page writes that
path into <video src>, so the filename would reveal the run in the pane and in
the browser's network tab.  render_movie() itself draws only an elapsed-time
stamp -- verified by extracting a frame, 2026-08-16 -- so the id passed as
title= is not burned in; the FILENAME is the only leak, and this driver fixes it.

Movies matter more on this page than on the earlier ones.  Both questions it
asks are dynamic: whether an event is a seizure at all (the class suspects), and
whether a slowly advancing tonic wavefront precedes the fast waves (the
subtype).  Neither is decidable from a static panel.

    PYTHONPATH=. python render_pass3_movies.py
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import render_movies as RM             # noqa: E402
import build_pass3_page as B3          # noqa: E402

MOV_DIR = B3.MOV_DIR
KEY_CSV = B3.KEY_CSV


def blind_key():
    if not os.path.exists(KEY_CSV):
        raise SystemExit("%s not found -- run build_pass3_page.py first" % KEY_CSV)
    with open(KEY_CSV, newline="", encoding="utf-8") as f:
        return {r["run_key"]: r["blind_id"] for r in csv.DictReader(f)}


def overview_blind(npz_path, blind_id, outdir=MOV_DIR):
    """render_movies.overview(), but writing <blind_id>_overview.mp4."""
    import numpy as np
    with np.load(npz_path, allow_pickle=True) as f:
        env = np.asarray(f["env"], np.float32)
        t_env = np.asarray(f["t_env"], np.float32)
        n = int(f["n"])
        fmax_hz = float(f["f_max"]) * 1e3
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "%s_overview.mp4" % blind_id)
    RM.render_movie(env * fmax_hz, t_env, n, p, fps=RM.FPS, title=blind_id)
    return p, dict(blind_id=blind_id, frames=int(env.shape[0]),
                   MB=round(os.path.getsize(p) / 1e6, 2))


def main():
    m = blind_key()
    b = B3.bundles(set(m))
    os.makedirs(MOV_DIR, exist_ok=True)
    t0 = time.time()
    metas = []
    for k in sorted(b):
        _p, meta = overview_blind(b[k], m[k])
        metas.append(meta)
        print("  %-8s %4d frames %5.1f MB" % (meta["blind_id"], meta["frames"],
                                              meta["MB"]), flush=True)
    json.dump(metas, open(os.path.join(MOV_DIR, "movies.json"), "w"), indent=1)
    bad = [f for f in os.listdir(MOV_DIR)
           if f.endswith(".mp4") and not f.startswith("V3-")]
    if bad:
        raise SystemExit("un-blinded movie filenames: %s" % bad)
    print("TOTAL %.0f s for %d movies, %.1f MB"
          % (time.time() - t0, len(metas), sum(x["MB"] for x in metas)))


if __name__ == "__main__":
    main()
