"""Build a scoring page that settles the CLASS and the SUBTYPE in one pass.

Two open questions, one page, because their run sets overlap by five runs.

  * CLASS.  Screening the 50-draw sweep on `frac_units_duty_over_1pct` -- the
    only stored statistic that separates the 54 human-verified calls cleanly
    (non-seizure <= 0.3986, seizure >= 0.9458, wide empty gap) -- turns up 10
    runs sitting INSIDE that gap.  Five are labelled NOT_SEIZURE yet exceed the
    non-seizure ceiling; five are labelled SEIZURE yet fall below the seizure
    floor.  The gap is empty in the verified set, so nothing tells us what a run
    living there should be called.  Only a human can.
  * SUBTYPE.  Of the 37 SEIZURE runs in the sweep, 21 have no subtype label.
    The onset distinction (slow tonic wavefront first, versus fast traveling
    waves at onset) is not recoverable from any stored scalar -- all seven
    end-window features overlap, and the recruitment ramp only NEARLY separates
    the 16 already-labelled runs.

Union = 26 runs, not 31: draw21_seq, draw21_sim, draw32_seq, draw47_sim and
draw48_sim are in both sets.  Those five are exactly the runs where one call
answers both questions -- score one a seizure and the class stands with a
subtype supplied; score it not-a-seizure and the subtype question dissolves.

ONE PASS SUFFICES BECAUSE THE OPTION LIST ALREADY ENCODES BOTH AXES.  Picking a
GROUP answers the class question; picking an ITEM WITHIN the seizure group
answers the subtype question.  Nothing is added to build_fullfield_html.OPTIONS.

SIX CONTROLS with stable prior calls are mixed in blind (two slow-onset, two
fast-onset, one non-seizure, one tonic).  This page asks for judgements in a
region no verified run occupies, so if the ambiguous runs prove hard the
controls say whether that is the runs or a shift in criteria on this pass --
a within-pass reliability check instead of a fourth full pass later.

BLINDING.  Ids are V3-01..V3-32 from a seed used by neither earlier pass, and
the builder refuses if any run keeps its pass-2 id.  Role (suspect / subtype /
control) is NOT in the id, the filenames, or the page: knowing a run is a
control would defeat it, and knowing a run is a suspect would bias it toward
the class it is suspected of.  The mapping lives only in pass3_blind_key.csv.

    PYTHONPATH=. python build_pass3_page.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())
import build_fullfield_html as BFH     # noqa: E402
import render_fullfield_page as RFP    # noqa: E402

NPZ_DIRS = ("sweep50_out", "verify42_out", "confirmed12_out")
RUNS_JSON = "pass3_runs.json"
FIG_DIR = "p3_figs"
WEB_DIR = "p3_figs_webp"
MOV_DIR = "p3_movies"
KEY_CSV = "pass3_blind_key.csv"
PASS2_KEY = "pass2_blind_key.csv"
OUT_HTML = "liou_fullfield_pass3.html"
SHUFFLE_SEED = 20260820           # not verify42's 42, not pass2's 20260819
SIGMA = 0.11


def wanted():
    """The 32 run keys, from the JSON the analysis wrote -- never hardcoded."""
    with open(RUNS_JSON, encoding="utf-8") as f:
        m = json.load(f)
    if len(m) != 32:
        raise SystemExit("%s lists %d runs, expected 32" % (RUNS_JSON, len(m)))
    return m


def bundles(keys):
    """run_key -> npz path, refusing anything not produced at SIGMA."""
    found = {}
    for d in NPZ_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            k = os.path.splitext(os.path.basename(p))[0]
            if k not in keys or k in found:
                continue
            with np.load(p, allow_pickle=True) as z:
                sig = json.loads(str(z["summary"])).get("lfp_sigma")
            if sig != SIGMA:
                raise SystemExit(
                    "%s was produced at lfp_sigma=%r, not %r -- amplitude is "
                    "not comparable across sigmas and the panels would mix two "
                    "instruments on one page." % (p, sig, SIGMA))
            found[k] = p
    missing = sorted(set(keys) - set(found))
    if missing:
        raise SystemExit("no bundle for: %s" % missing)
    return found


def blind_map(keys):
    """run_key -> V3-NN, carrying no hint of role, class, or parameter."""
    keys = sorted(keys)
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(keys))
    m = {keys[i]: "V3-%02d" % (j + 1) for j, i in enumerate(order)}
    assert len(set(m.values())) == len(keys), "blind ids collided"
    for v in m.values():
        for leak in ("anchor", "draw", "sim", "seq", "ctrl", "susp"):
            assert leak not in v, (v, leak)
    return m


def main():
    roles = wanted()
    b = bundles(set(roles))
    m = blind_map(list(b))

    # No run may keep its pass-2 id: a remembered "P37 was a seizure" attached
    # to the id would make this partly a recall test rather than a fresh read.
    if os.path.exists(PASS2_KEY):
        with open(PASS2_KEY, newline="", encoding="utf-8") as f:
            p2 = {r["run_key"]: r["blind_id"] for r in csv.DictReader(f)}
        same = [k for k, v in m.items() if p2.get(k) == v]
        if same:
            raise SystemExit("blind id unchanged from pass 2 for: %s" % same)

    with open(KEY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "run_key", "role", "prior_class",
                    "pass2_blind_id", "frac_over_1pct", "admissible"])
        for k in sorted(m, key=lambda x: m[x]):
            r = roles[k]
            w.writerow([m[k], k, r["role"], r["sheet_class"],
                        p2.get(k, "") if os.path.exists(PASS2_KEY) else "",
                        r["frac_over_1pct"], r["admissible"]])

    # Panels: render_run emits the LFP strip, its gutter, the full-resolution
    # sheet heatmap and the screen-height overview, all under the blind id.
    os.makedirs(WEB_DIR, exist_ok=True)
    for k in sorted(b):
        parts, _meta = RFP.render_run(b[k], FIG_DIR)
        for part, src in parts.items():
            q = os.path.join(WEB_DIR, "%s_%s.webp" % (m[k], part))
            Image.open(src).save(q, "WEBP", lossless=True, method=4)
        print("  %-22s -> %s (%d parts)" % (k, m[k], len(parts)), flush=True)

    # A stored verdict file in cwd would be read by the builder and shown.
    for cue in ("fullfield_verdicts.csv", "sweep50_labels.csv"):
        if os.path.exists(cue):
            raise SystemExit("%s in cwd would cue the read -- move it aside "
                             "before building the blind page." % cue)

    out, n, checked = BFH.build(fig_dir=WEB_DIR, out=OUT_HTML, mov_dir=MOV_DIR)
    print("built %s : %d runs, js check = %s" % (out, n, checked))
    if n != 32:
        raise SystemExit("page has %d runs, expected 32" % n)


if __name__ == "__main__":
    main()
