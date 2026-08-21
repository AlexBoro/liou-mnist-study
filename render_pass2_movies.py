"""Overview sheet movies for the 54 pass-2 runs, named by the P-blind id.

Same blinding constraint as render_verify42_movies: render_movies.overview()
names its output by run key and the page writes that path into <video src>, so
the filename would reveal the run.  See that module for the measured note on
render_movie's unused ``title=`` argument.

Overview tier only -- no bundle stored rate_full, so the detail button stays
inactive and the page's own caveat about 80 ms/frame being below the discharge
Nyquist applies.

    PYTHONPATH=. python render_pass2_movies.py
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import build_pass2_page as B                    # noqa: E402
from render_verify42_movies import overview_blind  # noqa: E402

MOV_DIR = "p2_movies"


def main():
    with open(B.KEY_CSV, newline="", encoding="utf-8") as f:
        m = {r["run_key"]: r["blind_id"] for r in csv.DictReader(f)}
    b = B.bundles()
    metas, t0 = [], time.time()
    for rk in sorted(m):
        _p, meta = overview_blind(b[rk], m[rk], outdir=MOV_DIR)
        metas.append(meta)
        print("%s %5.2f MB" % (meta["blind_id"], meta["MB"]), flush=True)
    json.dump(metas, open(os.path.join(MOV_DIR, "movies.json"), "w"), indent=1)
    bad = [f for f in os.listdir(MOV_DIR)
           if f.endswith(".mp4") and not f.startswith("P")]
    if bad:
        raise SystemExit("un-blinded movie filenames: %s" % bad)
    print("TOTAL %.0f s for %d movies, %.1f MB"
          % (time.time() - t0, len(metas), sum(x["MB"] for x in metas)))


if __name__ == "__main__":
    main()
