"""Render the 42 unverified runs and build a BLIND full-field scoring page.

Companion to run_verify42.py.  Three steps: render each bundle's panels, convert
to lossless webp, build liou_fullfield_verify42.html.

BLINDING.  The 12-run full-field page used real run keys, which was acceptable
there because all 12 were being scored and the keys were already known to the
scorer.  Here it is not: names like ``anchor_scale4_sim`` announce which runs
carry a prediction under test, and ``anchor_gE220`` telegraphs a parameter
value.  So every run is relabelled V01..V42 in shuffled order, exactly as the
54-run montage adjudication did, and the mapping is written to
verify42_blind_key.csv -- which must NOT sit beside the HTML while scoring.

The shuffle is seeded (42) so the assignment is reproducible, and the key is
written separately so a re-build does not silently produce a different mapping
than the one already scored.

WHAT THE PAGE SHOWS.  No stored verdicts are supplied, so every run's "my call"
reads "?" and the reveal shows nothing to agree with.  That is deliberate: these
runs' automated calls are what is being tested, and showing them would cue the
read.  The comparison against the automated calls happens after export, by
joining on the blind key.

    PYTHONPATH=. python build_verify42_page.py
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

NPZ_DIR = "verify42_out"
FIG_DIR = "vf42_figs"
WEB_DIR = "vf42_figs_webp"
KEY_CSV = "verify42_blind_key.csv"
OUT_HTML = "liou_fullfield_verify42.html"
SHUFFLE_SEED = 42


def blind_map(keys):
    """run_key -> V01..VNN in seeded shuffled order."""
    keys = sorted(keys)
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(keys))
    m = {keys[i]: "V%02d" % (j + 1) for j, i in enumerate(order)}
    assert len(set(m.values())) == len(keys), "blind ids collided"
    assert not any("anchor" in v or "draw" in v for v in m.values())
    return m


def main():
    paths = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    if not paths:
        raise SystemExit("no bundles in %s -- run run_verify42.py first" % NPZ_DIR)
    keys = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    m = blind_map(keys)

    with open(KEY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "run_key"])
        for k in sorted(m, key=lambda x: m[x]):
            w.writerow([m[k], k])

    os.makedirs(WEB_DIR, exist_ok=True)
    for p in paths:
        rk = os.path.splitext(os.path.basename(p))[0]
        lab = m[rk]
        parts, _meta = RFP.render_run(p, FIG_DIR)
        for part, src in parts.items():
            q = os.path.join(WEB_DIR, "%s_%s.webp" % (lab, part))
            Image.open(src).convert("RGB").save(q, format="WEBP",
                                                lossless=True, method=4)
        print("%s  <- %s" % (lab, rk), flush=True)

    # No verdict file: _verdicts() returns {} for a missing path, so every run
    # renders with cls/verdict "?" and nothing is leaked.
    if os.path.exists("fullfield_verdicts.csv"):
        raise SystemExit(
            "fullfield_verdicts.csv is present in cwd -- it would supply "
            "automated calls for the 12 already-scored runs and cue the read. "
            "Move it aside before building the blind page.")

    out, n, checked = BFH.build(fig_dir=WEB_DIR, out=OUT_HTML,
                                mov_dir="vf42_movies_none")
    print("built %s : %d runs, js check = %s" % (out, n, checked))
    if n != len(paths):
        raise SystemExit("page has %d runs, expected %d" % (n, len(paths)))


if __name__ == "__main__":
    main()
