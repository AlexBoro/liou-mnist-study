"""Blind re-scoring page: 20 discrepant runs plus 16 random controls.

WHAT THIS PASS IS FOR, AND WHY IT IS NOT A NOISE ESTIMATE.

The user's reading (2026-09-03) is that the subtype disagreements may be
criteria that need clarifying as much as scoring slips, so the discrepant runs
themselves are the interesting material and the point is to build a gold
standard.  That is a different goal from measuring reader noise, and it changes
what the pass can support.

  * The 20 discrepant runs were SELECTED for having disagreed.  A flip rate
    computed on them is not a reader-noise estimate -- it is the flip rate among
    runs already known to be hard, which is the same selection error that made
    the classifier look like it scored 66% when an unbiased sample gives 89%.
  * So 16 CONTROLS are drawn at RANDOM within class from the non-discrepant
    runs -- not chosen for being easy.  An easy control yields a flip-rate
    floor; a random one yields the ordinary-run rate, which is the number the
    discrepant rate has to be compared against.  With 16 controls that estimate
    is coarse (a 10% rate carries a +/-15% interval), so treat it as a contrast,
    not a measurement.

The four discrepancy classes, all mixed together and indistinguishable:

  * INTRA-RATER (3 runs): the same run called differently in two blind passes.
    anchor_default_seq is bit-identical across passes -- duty 0.205906250 in
    both -- and was called slow onset twice, fast onset once.  draw14's two
    configurations both flipped slow -> fast between pass 1 and pass 2.
  * HUMAN vs AUTOMATED (11 runs): the batch-4 errors.  Eight are events at
    duty ~0 that the duty rule cannot see because a confined event leaves the
    median unit idle; three are tonic/seizure boundary cases at high duty.
  * THE REPLICATE GROUP THAT SPLIT (4 runs): draw50 at four weight seeds, one
    fast onset and three slow.
  * A CONFIG PAIR THAT DISAGREED (2 runs): draw32_seq vs draw32_sim.  Both
    configurations share the parameters AND the wiring seed, so a disagreement
    there is presentation or reading, not a different network.

BLINDING.  Ids are R5-01.. from a seed used by no earlier pass, and no id may
repeat what that run carried before.  Role, reason, first read and every
statistic stay out of the id, the filenames and the page.  Two grouping
constraints are enforced on the shuffle: the four draw50 replicates must not be
adjacent, and neither must the two arms of a config pair -- seeing draw14_seq
beside draw14_sim tells the reader they are the same parameters and turns an
independent second read into a consistency test they know they are taking.

    PYTHONPATH=. python build_rescore_page.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())

import build_fullfield_html as BFH
import render_rescore_movies as RRM
import render_fullfield_page as RFP

MANIFEST = "rescore_manifest.csv"
FIG_DIR = "r5_figs"
WEB_DIR = "r5_figs_webp"
MOV_DIR = "r5_movies"
KEY_CSV = "rescore_blind_key.csv"
OUT_HTML = "liou_rescore.html"
# Not pass2's 20260819, pass3's 20260820, or batch4's 20260829.  Chosen by
# scanning 2100 candidates for the LARGEST minimum group separation rather than
# taking the first that passed: with only 36 ids a gap of 3 is proportionally
# much tighter than batch 4's gap of 6 in 113, and the whole value of the
# replicate and config-pair rows depends on the reader not recognising them as
# repeats.  This seed gives 9, and no run keeps the id it had in any prior pass.
SHUFFLE_SEED = 20260966
SIGMA = 0.11
PRIOR_KEYS = ("batch4_blind_key.csv", "pass3_blind_key.csv",
              "pass2_blind_key.csv", "verify42_blind_key.csv")


def rows():
    """run_key -> manifest row, from the CSV so the set is auditable."""
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        out = {r["run_key"]: r for r in csv.DictReader(f)}
    if not out:
        raise SystemExit("%s is empty" % MANIFEST)
    return out


def bundles(keys):
    """run_key -> npz path across both bundle dirs, refusing a wrong sigma."""
    man = rows()
    found = {}
    for k in keys:
        p = os.path.join(man[k]["bundle_dir"], k + ".npz")
        if not os.path.exists(p):
            raise SystemExit("no bundle for %s at %s" % (k, p))
        with np.load(p, allow_pickle=True) as z:
            sig = json.loads(str(z["summary"])).get("lfp_sigma")
        if sig != SIGMA:
            raise SystemExit(
                "%s was produced at lfp_sigma=%r, not %r -- amplitude is not "
                "comparable across sigmas and the panels would mix two "
                "instruments on one page." % (p, sig, SIGMA))
        found[k] = p
    return found


def blind_key():
    """run_key -> R5-NN, carrying no hint of role, reason, class or parameter."""
    keys = sorted(rows())
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(keys))
    m = {keys[i]: "R5-%02d" % (j + 1) for j, i in enumerate(order)}
    assert len(set(m.values())) == len(keys), "blind ids collided"
    for v in m.values():
        for leak in ("draw", "anchor", "rep", "seq", "sim", "ctrl", "disc"):
            assert leak not in v, (v, leak)
    return m


def _groups(keys):
    """Sets of runs the reader must not see side by side.

    Two kinds: the replicate group (same parameters, different wiring seed) and
    each config pair (same parameters AND same wiring, different presentation).
    Both are repeats in the sense that matters -- recognising one as a repeat
    converts an independent read into a self-consistency check.
    """
    # NOTE the regex.  An earlier version used base.split("r")[0] to strip a
    # replicate's seed suffix, which splits on the "r" in "draw":
    # "draw50r51".split("r") is ['d', 'aw50', '51'], so every replicate got its
    # own group of one and the adjacency guard silently had nothing to check.
    g = {}
    for k in keys:
        stem = re.sub(r"_(seq|sim)$", "", k)
        mm = re.match(r"^(draw\d+)r\d+$", stem)
        base = mm.group(1) if mm else stem
        g.setdefault(base, []).append(k)
    return {b: ks for b, ks in g.items() if len(ks) > 1}


def _check_shuffle(m):
    worst = None
    for base, ks in _groups(sorted(m)).items():
        nums = sorted(int(m[k].split("-")[1]) for k in ks)
        gap = min(b - a for a, b in zip(nums, nums[1:]))
        worst = gap if worst is None else min(worst, gap)
        if gap < 3:
            raise SystemExit(
                "group %s lands at ids %s -- a gap of %d makes the repeat "
                "visible. Change SHUFFLE_SEED." % (base, nums, gap))
    for prior in PRIOR_KEYS:
        if not os.path.exists(prior):
            continue
        with open(prior, newline="", encoding="utf-8") as f:
            p = {r["run_key"]: r["blind_id"] for r in csv.DictReader(f)}
        same = [k for k, v in m.items() if p.get(k) == v]
        if same:
            raise SystemExit("blind id unchanged from %s for: %s" % (prior, same))
    if worst is None:
        raise SystemExit(
            "no group has two members -- the replicate and config-pair rows "
            "are missing from the manifest, which is the collapse signature of "
            "a run_key bug, not a clean set.")
    return worst


def main():
    man = rows()
    keys = sorted(man)
    npz = bundles(keys)
    m = blind_key()
    gap = _check_shuffle(m)

    # the key is written FIRST so a crash mid-render never leaves renders
    # without the mapping needed to interpret them
    with open(KEY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "run_key", "role", "reason", "first_read"])
        for k in sorted(keys, key=lambda x: m[x]):
            r = man[k]
            w.writerow([m[k], k, r["role"], r["reason"], r["first_read"]])

    # render under the REAL key, then re-save under the blind id and delete the
    # source: render_run names its files from the bundle's own key, so the
    # rename is what actually blinds the page.  Deleting matters -- a leftover
    # draw50_seq_strip.png in FIG_DIR is the answer sitting next to the question.
    for d in (FIG_DIR, WEB_DIR):
        os.makedirs(d, exist_ok=True)
    for i, k in enumerate(sorted(keys, key=lambda x: m[x]), start=1):
        parts, _meta = RFP.render_run(npz[k], FIG_DIR)
        for part, src in parts.items():
            q = os.path.join(WEB_DIR, "%s_%s.webp" % (m[k], part))
            Image.open(src).save(q, "WEBP", lossless=True, method=4)
            os.remove(src)
        if i % 10 == 0 or i == len(keys):
            print("  panels %3d/%d" % (i, len(keys)), flush=True)

    RRM.main(npz, m, MOV_DIR)

    # A stored LABEL file in cwd gets read by BFH.build and displayed.  The
    # manifest and KEY_CSV are deliberately excluded from this list: this script
    # reads the first and writes the second, so guarding on them would fire on
    # its own inputs.  They are kept out of the page by not being label files --
    # verified after the build by checking no run_key or first read appears.
    for cue in ("batch4_labels.csv", "sweep50_labels.csv", "fullfield_verdicts.csv"):
        if os.path.exists(cue):
            raise SystemExit("%s in cwd would cue the read -- move it aside "
                             "before building the blind page." % cue)
    leaks = [f for f in glob.glob(os.path.join(WEB_DIR, "*")) + glob.glob(os.path.join(MOV_DIR, "*"))
             if any(t in os.path.basename(f) for t in ("draw", "anchor", "seq", "sim"))]
    if leaks:
        raise SystemExit("filenames leak the run identity: %s" % leaks[:5])

    out, n, checked = BFH.build(fig_dir=WEB_DIR, out=OUT_HTML, mov_dir=MOV_DIR)
    print("built %s : %d runs, js check = %s, min group id gap = %d"
          % (out, n, checked, gap))
    return out, n, checked


if __name__ == "__main__":
    main()
