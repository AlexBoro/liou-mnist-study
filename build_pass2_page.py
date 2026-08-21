"""Build a re-scoring page over ALL 54 runs for an intra-rater reliability check.

The user has scored every run once (12 in liou_fullfield.html at sigma 0.025,
42 in liou_fullfield_verify42.html at sigma 0.11) and wants to re-score now that
a wider range of behaviours has been seen, to test whether the criteria drifted.

THREE THINGS THIS HAS TO GET RIGHT, or the second pass measures something else:

1. A DIFFERENT BLINDING.  Reusing V01..V42 would let a remembered label attach to
   an id -- the second pass would then partly measure recall of pass 1 rather
   than an independent read.  Ids here are P01..P54 from a different seed, and
   the ordering is reshuffled so neighbours differ too.  The pass1 <-> pass2
   mapping lives ONLY in pass2_blind_key.csv.

2. ONE INSTRUMENT.  All 54 panels are rendered from sigma-0.11 bundles.  The 12
   originally scored at 0.025 are re-run by run_confirmed12_sigma011.py first.
   Without this, a changed call on those 12 could be the electrode rather than
   the scorer: at 0.025 the montage is a sparse sampling whose summed sensitivity
   varies 625x across the sheet, at 0.11 it is near-uniform.

3. NO CALLS SHOWN, pass 1's included.  The page supplies no verdict file, so
   every run reads "?".  Pass 1's calls are not in the page in any form.

The 12 that carry a 0.025 score AND a 0.11 re-run are the one place where
instrument and pass are confounded; the analysis should report the 42 (same
instrument both passes) as the primary agreement figure and the 12 separately.

    PYTHONPATH=. python build_pass2_page.py
"""
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())
import render_fullfield_page as RFP     # noqa: E402
import build_fullfield_html as BFH      # noqa: E402

NPZ_DIRS = ("verify42_out", "confirmed12_out")
FIG_DIR = "p2_figs"
WEB_DIR = "p2_figs_webp"
KEY_CSV = "pass2_blind_key.csv"
PASS1_KEY = "verify42_blind_key.csv"
OUT_HTML = "liou_fullfield_pass2.html"
SHUFFLE_SEED = 20260819          # deliberately NOT verify42's 42


def bundles():
    """run_key -> npz path across both bundle dirs, refusing duplicates."""
    out = {}
    for d in NPZ_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            k = os.path.splitext(os.path.basename(p))[0]
            if k in out:
                raise SystemExit("%s appears in two bundle dirs: %s and %s"
                                 % (k, out[k], p))
            out[k] = p
    return out


def blind_map(keys):
    keys = sorted(keys)
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(keys))
    m = {keys[i]: "P%02d" % (j + 1) for j, i in enumerate(order)}
    assert len(set(m.values())) == len(keys), "blind ids collided"
    assert not any(("anchor" in v) or ("draw" in v) for v in m.values())
    return m


def main():
    b = bundles()
    if len(b) != 54:
        raise SystemExit("expected 54 bundles, found %d -- run "
                         "run_confirmed12_sigma011.py first" % len(b))
    m = blind_map(b)

    # Pass-1 ids must not survive into pass 2 for any run.
    if os.path.exists(PASS1_KEY):
        with open(PASS1_KEY, newline="", encoding="utf-8") as f:
            p1 = {r["run_key"]: r["blind_id"] for r in csv.DictReader(f)}
        same = [k for k, v in m.items() if p1.get(k) == v]
        if same:
            raise SystemExit("blind id unchanged from pass 1 for: %s" % same)
    else:
        p1 = {}

    with open(KEY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "run_key", "pass1_blind_id", "sigma"])
        for k in sorted(m, key=lambda x: m[x]):
            w.writerow([m[k], k, p1.get(k, ""), "0.11"])

    os.makedirs(WEB_DIR, exist_ok=True)
    for k in sorted(b):
        parts, _meta = RFP.render_run(b[k], FIG_DIR)
        for part, src in parts.items():
            q = os.path.join(WEB_DIR, "%s_%s.webp" % (m[k], part))
            Image.open(src).convert("RGB").save(q, format="WEBP",
                                                lossless=True, method=4)
        print("%s  <- %s" % (m[k], k), flush=True)

    if os.path.exists("fullfield_verdicts.csv"):
        raise SystemExit("fullfield_verdicts.csv in cwd would cue the read -- "
                         "move it aside before building")

    out, n, checked = BFH.build(fig_dir=WEB_DIR, out=OUT_HTML,
                                mov_dir="p2_movies")
    print("built %s : %d runs, js check = %s" % (out, n, checked))
    if n != 54:
        raise SystemExit("page has %d runs, expected 54" % n)


if __name__ == "__main__":
    main()
