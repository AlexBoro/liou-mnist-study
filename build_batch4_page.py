"""Blind scoring page for batch 4: 100 unbiased draws, 6 replicates, 7 anchors.

WHAT THIS BATCH IS FOR.  Two things at once, from one unbiased sample.  It maps
the seizure boundary over 100 fresh parameter draws, and because ~34% of draws
land in the tonic/seizure duty overlap by base rate, it also triples the number
of scored runs in the region where the committed duty rule fails (12 -> ~46).
A targeted arm was designed, measured, and rejected: a surrogate on 35 draws
reaches CV R2 0.38, giving 1.6x lift, which would have cost 40 unbiased points
for ~8 extra in-band runs.  See batch4_surrogate_diagnostics.csv.

WHY EVERYTHING IS MIXED AND BLIND.

  * REPLICATES.  Six runs are two parameter sets at three weight seeds each.
    They give the label-noise estimate the corpus has never had -- every claim
    that class is a property of the parameters rests on an untested assumption.
    They only work if the reader cannot tell a replicate from a draw, so they
    carry ordinary ids and the shuffle is checked to separate each group.
  * ANCHORS.  Seven runs whose class is already known from two scoring passes.
    They are the within-pass reliability check: if the new draws prove hard, the
    anchors say whether that is the runs or a drift in criteria on this pass.
    Knowing which runs are anchors would defeat both purposes.

BLINDING.  Ids are B4-001.. from a seed used by no earlier pass.  Kind (draw /
replicate / anchor), parameters, and every automated statistic stay out of the
id, the filenames and the page; the mapping lives only in batch4_blind_key.csv.

    PYTHONPATH=. python build_batch4_page.py
"""
from __future__ import annotations

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
import run_batch4 as B4                # noqa: E402

NPZ_DIR = "batch4_out"
FIG_DIR = "b4_figs"
WEB_DIR = "b4_figs_webp"
MOV_DIR = "b4_movies"
KEY_CSV = "batch4_blind_key.csv"
OUT_HTML = "liou_batch4.html"
SHUFFLE_SEED = 20260829            # not 42, not pass2's 20260819, not pass3's 20260820
SIGMA = 0.11
PRIOR_KEYS = ("pass3_blind_key.csv", "pass2_blind_key.csv", "verify42_blind_key.csv")


def bundles(keys):
    """run_key -> npz path, refusing anything not produced at SIGMA."""
    found = {}
    for p in sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz"))):
        k = os.path.splitext(os.path.basename(p))[0]
        if k not in keys:
            continue
        with np.load(p, allow_pickle=True) as z:
            sig = json.loads(str(z["summary"])).get("lfp_sigma")
        if sig != SIGMA:
            raise SystemExit(
                "%s was produced at lfp_sigma=%r, not %r -- amplitude is not "
                "comparable across sigmas and the panels would mix two "
                "instruments on one page." % (p, sig, SIGMA))
        found[k] = p
    missing = sorted(set(keys) - set(found))
    if missing:
        raise SystemExit("no bundle for %d run(s): %s"
                         % (len(missing), missing[:6]))
    return found


def blind_key():
    """run_key -> B4-NNN, carrying no hint of kind, class or parameter."""
    keys = sorted(B4.rows())
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(keys))
    m = {keys[i]: "B4-%03d" % (j + 1) for j, i in enumerate(order)}
    assert len(set(m.values())) == len(keys), "blind ids collided"
    for v in m.values():
        for leak in ("draw", "anchor", "rep", "seq", "sim", "ctrl"):
            assert leak not in v, (v, leak)
    return m


def _check_shuffle(m, rows):
    """Replicate groups must not be adjacent, and no id may repeat a prior pass.

    Adjacent replicates are the failure mode that matters: three consecutive
    ids showing near-identical sheets tell the reader they are looking at the
    same parameters, and the label-noise estimate becomes a consistency test
    the reader knows they are taking.
    """
    groups = {}
    for k, r in rows.items():
        if r["kind"] in ("draw", "replicate"):
            # group on base_tag, not tag: a replicate now carries its own tag
            # (draw50r900) so that it gets its own bundle filename, but it is
            # the same parameters as draw50 and must not sit beside it.
            groups.setdefault(r["base_tag"], []).append(k)
    worst = None
    multi = 0
    for tag, ks in groups.items():
        if len(ks) < 2:
            continue
        multi += 1
        nums = sorted(int(m[k].split("-")[1]) for k in ks)
        gap = min(b - a for a, b in zip(nums, nums[1:]))
        worst = gap if worst is None else min(worst, gap)
        # This raise USED TO BE DEAD CODE.  It sat below the `return 0` in the
        # `if not multi` branch below, so from the day that branch was added
        # until 2026-09-03 no adjacency was ever rejected here -- the property
        # held only because the regression test asserted it separately and the
        # seed happened to give a gap of 6.  A guard that cannot fire is worse
        # than no guard: it reads as protection in review.
        if gap < 3:
            raise SystemExit(
                "replicate group %s lands at ids %s -- a gap of %d makes the "
                "repeat visible. Change SHUFFLE_SEED." % (tag, nums, gap))
    if not multi:
        # No group has two members, so no replicate survived into the manifest.
        # That is the collapse signature of the tag-collision bug, not a clean
        # batch -- return 0 so the caller's `gap >= 3` fails loudly instead of
        # comparing None and raising an unrelated TypeError.
        return 0
    for prior in PRIOR_KEYS:
        if not os.path.exists(prior):
            continue
        with open(prior, newline="", encoding="utf-8") as f:
            p = {r["run_key"]: r["blind_id"] for r in csv.DictReader(f)}
        same = [k for k, v in m.items() if p.get(k) == v]
        if same:
            raise SystemExit("blind id unchanged from %s for: %s" % (prior, same))
    return worst


def main():
    rows = B4.rows()
    b = bundles(set(rows))
    m = blind_key()
    gap = _check_shuffle(m, rows)

    with open(KEY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "run_key", "kind", "tag", "seed", "expected"])
        for k in sorted(m, key=lambda x: m[x]):
            r = rows[k]
            w.writerow([m[k], k, r["kind"], r["tag"], r["seed"], r["expected"]])

    os.makedirs(WEB_DIR, exist_ok=True)
    for i, k in enumerate(sorted(b, key=lambda x: m[x]), 1):
        parts, _meta = RFP.render_run(b[k], FIG_DIR)
        for part, src in parts.items():
            q = os.path.join(WEB_DIR, "%s_%s.webp" % (m[k], part))
            Image.open(src).save(q, "WEBP", lossless=True, method=4)
            os.remove(src)
        if i % 20 == 0 or i == len(b):
            print("  panels %3d/%3d" % (i, len(b)), flush=True)

    # A stored label file in cwd would be read by the builder and shown.
    for cue in ("fullfield_verdicts.csv", "sweep50_labels.csv", "batch4_labels.csv"):
        if os.path.exists(cue):
            raise SystemExit("%s in cwd would cue the read -- move it aside "
                             "before building the blind page." % cue)

    out, n, checked = BFH.build(fig_dir=WEB_DIR, out=OUT_HTML, mov_dir=MOV_DIR)
    print("built %s : %d runs, js check = %s, min replicate id gap = %s"
          % (out, n, checked, gap))
    return out, n, checked


if __name__ == "__main__":
    main()
