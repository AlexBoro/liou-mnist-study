"""Self-contained scrolling page for the 12 full-field runs.

Same reading mechanics as liou_adjudication.html -- fixed px/s, 10 s per screen,
pinned gutters, keyboard paging -- with two differences that follow from what
this page is for:

  * THE HEATMAP IS ALL 3136 UNITS, not the 30 electrodes.  It is a second image
    in the SAME horizontal scroll container as the LFP strip, so there is one
    scrollLeft and the two time axes cannot drift apart.  The LFP strip is
    sticky at the top of that container, so it stays visible while the reader
    pans down through the sheet.
  * NO SCORING UI.  These 12 runs are already adjudicated (see
    FULLFIELD_FINDINGS.md); this page exists to SHOW what the sheet did, and
    each run's header carries its montage class, its sheet verdict and the
    measured numbers rather than asking for a verdict.

Both gutters are scaled by their partner image's scale factor, exactly as the
adjudication page does -- see fitWindow().  The two gutters are separate images
because the sheet gutter is 3136 px tall and the LFP gutter is not.
"""
from __future__ import annotations

import csv
import json
import math
import os

PX_PER_S = 118.0      # must match render_fullfield_page.PX_PER_S
WINDOW_S = 10.0

# Scoring options, in the user's own wording (2026-08-15).  Four correspond to
# the sheet verdicts in fullfield_verdicts.csv -- three of them RENAMED, so the
# mapping below is load-bearing: without it the agreement comparison silently
# scores every run as a disagreement.  OTHER exists for morphologies not yet
# encountered and deliberately has NO automated counterpart, so a run scored
# OTHER can never count as agreement.
# ORDER AND GROUPING CHANGED 2026-08-16 after a mis-click cost two runs.
#
# The wording is the user's and is unchanged.  The problem was the LAYOUT: two
# options begin with the same word ("ENTRAINED, not a seizure" and "FOCAL
# SEIZURE (Liou-like)") and they sat at keys 1 and 3, split by an unrelated
# option, so a scan for "FOCAL" landed on key 1 first.  Both mis-clicks happened
# on the FIRST TWO runs scored, before the key positions were in muscle memory.
#
# The fix is to group by the distinction that actually matters -- is this a
# seizure or not -- so a slip between those two options now crosses a visible
# group boundary instead of being a neighbouring lookalike.  Keys follow display
# order, so the two "not a seizure" calls are 1-2 and the two seizure calls 3-4.
OPT_GROUPS = [
    ("not a seizure", ["ENTRAINED, not a seizure",
                       "TONIC SATURATION (not a seizure)"]),
    ("seizure",       ["FOCAL SEIZURE (Liou-like)",
                       "GENERALIZED SEIZURE (non-Liou morphology)"]),
    ("",              ["OTHER"]),
]
OPTIONS = [o for _lbl, opts in OPT_GROUPS for o in opts]
VERDICT_TO_OPTION = {
    "FOCAL / entrained, no spread":      "ENTRAINED, not a seizure",
    "GENERALIZED, non-Liou morphology":  "GENERALIZED SEIZURE (non-Liou morphology)",
    "SEIZURE (Liou-like)":               "FOCAL SEIZURE (Liou-like)",
    "TONIC SATURATION (not a seizure)":  "TONIC SATURATION (not a seizure)",
}
OPT_COLOUR = {
    "ENTRAINED, not a seizure":            "#2e7d32",
    "GENERALIZED SEIZURE (non-Liou morphology)": "#e67e22",
    "FOCAL SEIZURE (Liou-like)":                 "#c0392b",
    "TONIC SATURATION (not a seizure)":          "#8e24aa",
    "OTHER":                                     "#455a64",
}


def _verdicts(path="fullfield_verdicts.csv"):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def _buttons():
    """Radio labels, rendered in groups with a visible divider.

    The group headers are the whole point: the two confusable options differ by
    exactly the thing being judged, so putting "not a seizure" and "seizure"
    banners between them makes a mis-click cross a boundary the eye can see.
    """
    out = []
    for gi, (glabel, opts) in enumerate(OPT_GROUPS):
        if glabel:
            out.append('<span class="gsep%s">%s</span>'
                       % (" first" if gi == 0 else "", glabel))
        for o in opts:
            out.append('<label class="v" data-v="%s" style="--c:%s">'
                       '<input type="radio" name="v"><b>%d</b> %s</label>'
                       % (o, OPT_COLOUR[o], OPTIONS.index(o) + 1, o))
    return "\n  ".join(out)


def _movies(mov_dir, rid):
    """Movie files for one run, or {} if none were rendered.

    Movies are OPTIONAL: the page must build and score without them, so a
    missing ff_movies/ degrades to a hidden movie pane rather than a broken
    page.  The scoring workflow should not depend on a couple of minutes of
    video encoding having been run first.
    """
    out = {}
    for kind in ("overview", "detail"):
        p = os.path.join(mov_dir, "%s_%s.mp4" % (rid, kind))
        if os.path.exists(p):
            out[kind] = p.replace(os.sep, "/")
    return out


def _gutter_width(fig_dir, rid):
    """Natural width of the gutter images, asserting both panes agree.

    The page pins BOTH gutters to this one number so the shared grid column
    cannot be sized by whichever image is taller.  That is only sound if the two
    gutter rasters really are the same width, so it is checked here rather than
    assumed -- if the renderer ever emits different widths, pinning one value
    would silently squash one gutter and put its labels off their lanes.
    """
    try:
        from PIL import Image
    except ImportError:
        return 214
    ws = {}
    for part in ("gutter", "sheetgut", "sheetfitgut"):
        p = os.path.join(fig_dir, "%s_%s.webp" % (rid, part))
        if os.path.exists(p):
            ws[part] = Image.open(p).size[0]
    if len(set(ws.values())) > 1:
        raise SystemExit("gutter widths differ (%s) -- the page pins one width "
                         "for both panes, so they must match" % ws)
    return next(iter(ws.values())) if ws else 214


def _check_no_stray_escapes(tpl):
    """Refuse to build if a JS escape was eaten by the Python source parser.

    This JS lives inside a Python triple-quoted string, so a single-backslash
    escape (``\\n``, ``\\t``, ``\\u2013``) is consumed at Python parse time and
    emitted as the RAW character.  Inside a JS regex literal or string that is a
    syntax error, and a syntax error anywhere in the script means NOTHING runs --
    the page renders as bare alt text with the header stuck on its placeholder
    values, which is exactly how this shipped once (2026-08-15).

    The failure is invisible in Python: the template is a perfectly valid string.
    So the check looks for the raw control characters that a swallowed escape
    leaves behind, on lines that are JS code rather than prose.
    """
    for ln, line in enumerate(tpl.split("\n"), 1):
        # a JS string/regex literal cannot contain a raw tab or vertical space
        if "\t" in line or "\r" in line:
            raise SystemExit("template line %d contains a raw tab/CR -- a "
                             "single-backslash escape was eaten by Python: %r"
                             % (ln, line[:80]))
    # NOTE: an earlier version of this function also flagged unbalanced quotes
    # per line as a proxy for a split literal.  It was removed: JS comment
    # continuation lines legitimately contain apostrophes ("partner's"), so it
    # false-positived on prose and would have blocked every build.  Raw control
    # characters are the reliable signal here; a genuine split literal is caught
    # by _verify_parses(), which is an actual parser rather than a heuristic.


def _verify_parses(js):
    """Check the emitted JS. Never return quietly having checked nothing.

    esprima is the real check, but it is sdist-only so it cannot be installed
    into the shared environment and vanishes on every kernel reset -- which is
    exactly how this guard silently skipped once (2026-08-16) and let an
    unverified page ship.  So when esprima is absent we fall back to a targeted
    scan for the ONE defect class that has actually broken these pages twice: a
    JS string or regex literal split across lines by an escape that Python's
    source parser consumed.  That is much weaker than a parser, and the return
    value says which check ran so the caller can print it.

    Returns "esprima" | "fallback".  Raises SystemExit on a detected defect.
    """
    try:
        import esprima
    except ImportError:
        for ln, line in enumerate(js.split("\n"), 1):
            # a literal opened on this line and never closed on it
            for q in ('"', "'"):
                # ignore lines inside block comments / prose
                st = line.strip()
                if st.startswith(("*", "/*", "//")) or st.endswith("*/"):
                    continue
            # an unterminated regex character class is the signature of the
            # 2026-08-15 defect: "/[" with no closing "]" on the same line
            if "/[" in line and "]" not in line.split("/[", 1)[1]:
                raise SystemExit(
                    "line %d opens a regex character class that never closes on "
                    "this line -- a swallowed escape almost certainly split it: "
                    "%r" % (ln, line[:80]))
        return "fallback"
    esprima.parseScript(js)
    return "esprima"


def build(fig_dir="ff_figs_webp", out="liou_fullfield.html",
          mov_dir="ff_movies"):
    ids = sorted({f.rsplit("_", 1)[0] for f in os.listdir(fig_dir)
                  if f.endswith(".webp")})
    if not ids:
        raise SystemExit("no .webp figures in %s" % fig_dir)
    V = _verdicts()
    runs = []
    for rid in ids:
        parts = {k: "%s/%s_%s.webp" % (fig_dir, rid, k)
                 for k in ("strip", "gutter", "sheet", "sheetgut",
                           "sheetfit", "sheetfitgut")}
        for k, p in parts.items():
            if not os.path.exists(p):
                raise SystemExit("%s: missing %s" % (rid, k))
        v = V.get(rid, {})
        raw = v.get("sheet_verdict", "")
        mapped = VERDICT_TO_OPTION.get(raw)
        if raw and mapped is None:
            raise SystemExit(
                "%s: sheet_verdict %r has no entry in VERDICT_TO_OPTION -- add one, "
                "or every run of this class scores as a disagreement." % (rid, raw))
        runs.append(dict(id=rid, **parts, mov=_movies(mov_dir, rid),
                         cls=v.get("auto2", "?"),
                         verdict=mapped or "?",
                         peak=v.get("sheet_peak_frac", ""),
                         duty=v.get("duty_median", ""),
                         hz=v.get("mean_hz", "")))

    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d%H%M")
    _check_no_stray_escapes(HTML)
    import render_fullfield_page as _RFP
    pool = int(math.ceil(56 * 56 / float(_RFP.FIT_ROWS)))
    html = HTML.format(runs=json.dumps(runs), pxps=PX_PER_S, win=WINDOW_S,
                       n=len(runs), opts=json.dumps(OPTIONS), pool=pool,
                       gutw=_gutter_width(fig_dir, ids[0]),
                       ocol=json.dumps(OPT_COLOUR), stamp=stamp,
                       btns=_buttons())
    js = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    checked = _verify_parses(js)     # "esprima" (real parse) or "fallback"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out, len(runs), checked


HTML = """<!doctype html>
<meta charset="utf-8">
<title>Liou pilot -- full-field runs</title>
<style>
 :root {{ --bg:#fafafa; --fg:#111; --mut:#666; --line:#d8d8d8 }}
 * {{ box-sizing:border-box }}
 body {{ margin:0; font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background:var(--bg); color:var(--fg) }}
 header {{ position:sticky; top:0; z-index:30; background:var(--bg);
          border-bottom:1px solid var(--line); padding:6px 12px;
          display:flex; align-items:center; gap:14px; flex-wrap:wrap }}
 #runid {{ font-weight:600; font-family:ui-monospace,monospace }}
 .tag {{ font-size:11px; padding:1px 7px; border-radius:9px; border:1px solid var(--line);
        color:var(--mut); background:#fff }}
 .tag.v {{ font-weight:600 }}
 #stats {{ color:var(--mut); font-size:11.5px; font-variant-numeric:tabular-nums }}
 /* radio row: labels ARE the buttons; the inputs are visually hidden but remain
    real radios so keyboard/AT still see a single-choice group */
 #vote {{ display:flex; gap:6px; flex-wrap:wrap; align-items:center }}
 /* Group banners. The original defect was two options sharing a leading word;
    that is now fixed at the source (the entrained option was renamed to
    "ENTRAINED, not a seizure"), so the grouping is no longer load-bearing for
    that pair. It is kept because it encodes the decision the scorer is actually
    making -- seizure or not -- so any future slip between a seizure and a
    non-seizure call still crosses a visible divider. */
 .gsep {{ font-size:10px; letter-spacing:.06em; text-transform:uppercase;
         color:var(--mut); border-left:2px solid var(--line); padding-left:7px;
         margin-left:5px }}
 .gsep.first {{ border-left:none; margin-left:0; padding-left:0 }}
 .v {{ font-size:11px; padding:2px 8px; border:1px solid var(--line); border-radius:12px;
      background:#fff; color:var(--mut); cursor:pointer; user-select:none }}
 .v b {{ font-family:ui-monospace,monospace; margin-right:4px; color:#999 }}
 .v:hover {{ background:#f2f2f2 }}
 .v.on {{ border-color:var(--c); color:var(--c); font-weight:600;
         box-shadow:inset 0 0 0 1px var(--c) }}
 .v.on b {{ color:var(--c) }}
 .v input {{ position:absolute; opacity:0; pointer-events:none }}
 #mine {{ font-size:11.5px; padding:1px 8px; border-radius:9px; border:1px dashed var(--line);
         color:var(--mut); background:#fff }}
 #agree {{ font-size:11.5px; font-weight:600 }}
 #bar {{ flex:0 0 90px; height:5px; background:#e9e9e9; border-radius:3px; overflow:hidden }}
 #bar i {{ display:block; height:100%; background:#555; width:0 }}
 button {{ font:inherit; padding:4px 10px; border:1px solid var(--line); border-radius:5px;
          background:#fafafa; cursor:pointer }}
 button:hover {{ background:#f0f0f0 }}
 #wrap {{ padding:8px 12px 44px }}
 #tools {{ display:flex; align-items:center; gap:10px; margin:2px 0 6px;
          color:var(--mut); font-size:11.5px; flex-wrap:wrap }}
 #tpos {{ font-variant-numeric:tabular-nums; font-family:ui-monospace,monospace; color:var(--fg) }}
 #seek {{ flex:1 1 240px; min-width:160px }}
 /* ONE scroll box holds the gutters AND both images, as a 2x2 CSS grid:
 *
 *     [ LFP gutter  ][ LFP strip ]   <- row sticks to the TOP
 *     [ sheet gutter][ sheet     ]
 *       ^ column sticks to the LEFT
 *
 * WHY THE GUTTERS ARE INSIDE THE BOX (they were outside, and that was the bug
 * behind both vertical complaints).  With the gutter column as a sibling of the
 * scroller, vertical scrolling moved the sheet but NOT its row labels, so
 * "row 24" drifted away from row 24 as soon as you scrolled down -- and the
 * 3136 px gutter column overflowed the 82vh box instead of being clipped by it,
 * so the two gutters no longer lined up with their partners at all.
 *
 * Inside the grid each gutter is in normal flow, so it travels vertically with
 * the image beside it and stays in register by construction; `position:sticky;
 * left:0` pins it horizontally while time scrolls. The LFP row uses
 * `position:sticky; top:0` so the montage stays visible while the sheet is
 * panned. The LFP gutter is sticky on BOTH axes (it is in the pinned row AND
 * the pinned column), which is why it carries the higher z-index. */
 #mont {{ border:1px solid var(--line); border-radius:4px; background:#fff }}
 #scroll {{ overflow:auto; height:84vh;
           display:grid; grid-template-columns:max-content 1fr; align-items:start;
           grid-template-rows:auto 1fr }}
 #gut, #sgut, #strip, #sheet {{ display:block }}
 /* VERTICAL SPLIT.  The montage row is sticky at the top of the box and the
  * sheet scrolls beneath it.  Its height is set in JS (setSplit) to a FRACTION
  * of the box rather than left at the image's natural height: the strip renders
  * ~1157 px tall while the box is ~750 px on a laptop, so an unbounded sticky
  * row is taller than its own scrollport -- the sheet is pushed entirely below
  * the fold and vertical scrolling appears to do nothing.  That was the bug.
  *
  * The row must NOT get its own overflow: a nested scroller would let the
  * montage scroll horizontally independently of the sheet, which is exactly the
  * time-register drift this single-scroll-box design exists to prevent.  The
  * strip image is instead given an explicit height, compressing the 30 lanes
  * vertically.  That is a gain change, not a resample of the time axis -- lane
  * SPACING and trace WIDTH are independent in a montage, as on real EEG.
  * `[` and `]` adjust the split. */
 /* ONE grid, FOUR items -- gut | strip on row 1, sgut | sheet on row 2.
  *
  * There used to be a nested grid (#striprow) holding gut+strip, which meant the
  * gutter COLUMN WIDTH was computed independently in each grid.  Both gutters
  * are 214 px wide naturally but are displayed at different heights, and with
  * `width:auto` each one's rendered width follows its own aspect ratio: the LFP
  * gutter came out 161 px and the sheet gutter 237 px.  The two images therefore
  * started 76 px apart, which at the default zoom is 0.88 s of apparent time
  * shift between the panes -- the same x was a different instant in each.
  *
  * A single grid makes both panes share one column track, so their left edges
  * are identical by construction rather than by arithmetic.  The gutter widths
  * are ALSO pinned to the same explicit px in JS, so the track cannot be sized
  * by whichever image happens to be taller. */
 #gut, #strip {{ position:sticky; top:0; z-index:5; background:#fff }}
 #gut {{ left:0; z-index:7; border-right:1px solid var(--line);
        border-bottom:1px solid var(--line) }}
 #strip {{ border-bottom:1px solid var(--line) }}
 #sgut {{ position:sticky; left:0; z-index:4; background:#fff;
         border-right:1px solid var(--line) }}
 #gut, #sgut, #strip, #sheet {{ display:block }}
 #sheet {{ image-rendering:pixelated }}
 /* MOVIE PANE.  Collapsed by default and opened with `m`: the page's primary
  * job is scoring from the montage + sheet, and a video pane always open would
  * take screen the reviewer asked to give to those two.  When open it overlays
  * the sheet rather than shrinking it, so the montage geometry above is
  * untouched and closing it restores the exact scoring layout. */
 #movwrap {{ display:none; margin-top:6px; border:1px solid var(--line);
            border-radius:4px; background:#111; padding:6px }}
 #movwrap.on {{ display:block }}
 #movbar {{ display:flex; gap:8px; align-items:center; color:#bbb;
           font-size:11.5px; flex-wrap:wrap; margin-bottom:5px }}
 #movbar button {{ background:#222; color:#ddd; border-color:#444 }}
 #movbar button.sel {{ background:#2e7d32; color:#fff; border-color:#2e7d32 }}
 #mov {{ display:block; max-width:100%; max-height:70vh; background:#000 }}
 #movnote {{ color:#888; font-size:11px; margin-top:4px; line-height:1.35 }}
 #err {{ color:#b2182b; font-size:12px; padding:2px 0 }}
 footer {{ position:fixed; bottom:0; left:0; right:0; background:var(--bg);
          border-top:1px solid var(--line); padding:4px 12px; color:var(--mut);
          font-size:11.5px; display:flex; gap:16px; flex-wrap:wrap }}
 kbd {{ font:11px ui-monospace,monospace; border:1px solid var(--line);
       border-bottom-width:2px; border-radius:3px; padding:0 4px; background:#fafafa }}
</style>
<header>
  <span id="runid">--</span>
  <span id="prog">0 / {n}</span>
  <span id="bar"><i></i></span>
  <span id="stats"></span>
  <span id="mine">my call: hidden</span>
  <span id="agree"></span>
  <button id="prev">&larr; prev</button>
  <button id="next">next &rarr;</button>
  <button id="reveal">reveal mine</button>
  <button id="csv"><b>export CSV</b></button>
</header>
<div id="wrap2" style="padding:6px 12px 0">
  <div id="vote">
  {btns}
  </div>
</div>
<div id="wrap">
  <div id="err"></div>
  <div id="tools">
    <span>window <b id="tpos">0.0 &ndash; 10.0 s</b></span>
    <input id="seek" type="range" min="0" max="1000" value="0" step="1">
    <span><kbd>&lt;</kbd><kbd>&gt;</kbd> page &plusmn;10 s &middot; <kbd>,</kbd><kbd>.</kbd>
      nudge 1 s &middot; <kbd>Home</kbd><kbd>End</kbd> &middot; <kbd>z</kbd> time zoom &middot; <kbd>h</kbd> sheet height</span>
    <span id="zoom" style="font-style:italic"></span>
  </div>
  <div id="mont">
    <div id="scroll">
      <img id="gut" alt="LFP montage labels and fixed scales">
      <img id="strip" alt="30-channel LFP montage">
      <img id="sgut" alt="sheet row labels and colour scale">
      <img id="sheet" alt="all 3136 units, space-time envelope">
    </div>
  </div>
  <div id="movwrap">
    <div id="movbar">
      <b>sheet movie</b>
      <button id="mov-overview" data-kind="overview">whole run (80 ms/frame)</button>
      <button id="mov-detail" data-kind="detail">detail (20 ms/frame)</button>
      <span id="movmeta"></span>
      <button id="movclose" style="margin-left:auto">close (m)</button>
    </div>
    <video id="mov" controls loop preload="metadata"></video>
    <div id="movnote"></div>
  </div>
</div>
<footer>
  <span><kbd>&uarr;</kbd><kbd>&darr;</kbd> or <kbd>k</kbd><kbd>j</kbd> run</span>
  <span>LFP strip is pinned; scroll down inside the box to pan the sheet</span>
  <span><kbd>h</kbd> toggles whole-sheet overview / 1 px per unit</span>
  <span><kbd>[</kbd><kbd>]</kbd> montage/sheet split (50/50 default)</span>
  <span><kbd>m</kbd> sheet movie</span>
  <span><kbd>u</kbd> undo last call</span>
  <span><kbd>1</kbd>&ndash;<kbd>5</kbd> classify &middot; <kbd>r</kbd> reveal my call</span>
  <span>my classification is hidden so your read stays uncued</span>
  <span>scores autosave in this browser; export the CSV when done</span>
</footer>
<script>
const RUNS = {runs};
const PX_PER_S = {pxps}, WINDOW_S = {win};
const OPTS = {opts}, OCOL = {ocol};
const POOL = {pool};        // units per row in the pooled overview raster
const GUTW = {gutw};        // gutter width in px, identical for both panes
const KEY = "liou_ff_{stamp}";
/* My own classification is HIDDEN until the reader asks for it.  This project's
   acceptance criterion is agreement between an UNCUED read and the automated
   verdict (see the blinding procedure used for the 54-run pilot), so showing my
   call beside the radio buttons would anchor the judgement it is meant to test.
   Progress persists per browser; export the CSV to get it out. */
let i = 0, revealed = false;
let S = JSON.parse(localStorage.getItem(KEY) || "{{}}");
const $ = s => document.querySelector(s);
const strip = $("#strip"), gut = $("#gut"), sheet = $("#sheet"), sgut = $("#sgut");
const scroll = $("#scroll"), seek = $("#seek");

function secPerPx() {{
  const w = strip.clientWidth || strip.naturalWidth || 1;
  return ((strip.naturalWidth || 1) / PX_PER_S) / w;
}}
/* Visible width of the TIME axis only.  The gutters now live inside the scroll
   box, so scroll.clientWidth includes the pinned gutter column; using it raw
   overstated the visible window by the gutter width (~230 px) and the readout
   claimed more seconds on screen than were actually there. */
function visW() {{
  return Math.max(1, scroll.clientWidth - (gut.clientWidth || 0));
}}
function showTime() {{
  const t0 = scroll.scrollLeft * secPerPx();
  const t1 = t0 + visW() * secPerPx();
  $("#tpos").textContent = t0.toFixed(1) + " \\u2013 " + t1.toFixed(1) + " s";
  const max = Math.max(1, scroll.scrollWidth - scroll.clientWidth);
  seek.value = Math.round(1000 * scroll.scrollLeft / max);
}}
function pan(dSec) {{ scroll.scrollLeft += dSec / secPerPx(); showTime(); }}

/* One screen == WINDOW_S seconds.  Setting an image's CSS width rescales it
   vertically too (height:auto), so EACH gutter is scaled by ITS OWN partner's
   factor -- the LFP gutter by the strip's, the sheet gutter by the sheet's.
   They are separate images with very different natural heights, so a single
   shared factor would throw one of them out of register. */
let nativeZoom = false, split = 0.50;   // montage share of the box
/* fitSheet TRUE = show the whole sheet at once, using the PRE-POOLED overview
   raster (sheetfit, ~784 rows) rather than letting the browser squash the
   3136-row image.  That distinction is the whole point: a browser downscale is
   nearest-neighbour and SAMPLES ~800 of 3136 rows, which measured over the 12
   runs blanks 21 time bins that contain real activity -- concentrated in the
   sparse runs, where a missed discharge changes the verdict.  The pooled raster
   is a max over the same rows and loses none.
   Default TRUE: the reviewer works on a 1080x1920 portrait display and asked to
   see the full vertical extent of both panes at once. */
let fitSheet = true;
function fitWindow() {{
  const natW = strip.naturalWidth, vis = visW();
  if (!vis || !natW) return;
  const cssW = nativeZoom ? natW : vis * (natW / PX_PER_S) / WINDOW_S;
  const S = cssW / natW;
  strip.style.width = cssW + "px";
  /* Montage height: a fraction of the visible box, never its natural height.
     Both the strip and its gutter get the SAME height so the channel labels stay
     on their lanes. */
  const boxH = scroll.clientHeight || 700;
  const rowH = Math.max(120, Math.round(boxH * split));
  strip.style.height = rowH + "px";
  gut.style.height = rowH + "px";
  /* Both gutters are pinned to the SAME explicit width -- never width:auto.
     With auto, each gutter's width follows its own aspect ratio, and since the
     two are displayed at different heights they came out 161 px and 237 px, so
     the panes' t=0 sat 76 px apart (0.88 s of apparent shift at default zoom).
     GUTW is the natural gutter width, identical for both images by
     construction in the renderer. */
  gut.style.width = GUTW + "px";
  if (sheet.naturalWidth) {{
    /* The sheet is rendered at the SAME natural width as the strip, so the same
       CSS width puts a given x at the same instant in both images.

       HEIGHT IS SET EXPLICITLY, NOT LEFT TO height:auto.  With auto the sheet
       keeps its aspect ratio, so scaling the width to 10 s/screen also squashes
       3136 unit-rows into ~470 screen px -- 6.7 units per pixel, which is not
       the full resolution this page exists to show.  Pinning the height to
       naturalHeight keeps exactly one screen pixel per unit and lets the
       reader pan down through the sheet; `h` toggles a whole-sheet overview
       that restores the aspect ratio for a gross look. */
    /* In fit mode the sheet pane gets the REMAINDER of the box, and the
       overview raster (already pooled to ~784 rows) is scaled to it -- a small
       change either way, so no event is resampled away.  In full mode the image
       is pinned at one screen pixel per unit and the pane scrolls. */
    const sh = fitSheet ? Math.max(80, boxH - rowH - 2) : sheet.naturalHeight;
    sheet.style.width = cssW + "px";
    sheet.style.height = sh + "px";
    sgut.style.height = sh + "px";
    sgut.style.width = GUTW + "px";
  }}
  $("#zoom").textContent = (nativeZoom
    ? "time native 1:1 (" + (vis / PX_PER_S).toFixed(1) + " s/screen)"
    : (S < 0.98 ? "time scaled to " + Math.round(100 * S) + "%" : "time native 1:1"))
    + "  \u00b7  montage " + Math.round(100 * split) + "% of pane"
    + "  \u00b7  sheet " + (fitSheet
      ? "whole sheet, " + POOL + " units/row pooled"
      : "1 px per unit (scroll to pan)");
  showTime();
}}

function save() {{ localStorage.setItem(KEY, JSON.stringify(S)); }}

function draw() {{
  const r = RUNS[i], st = S[r.id] || {{}};
  strip.src = r.strip; gut.src = r.gutter;
  sheet.src = fitSheet ? r.sheetfit : r.sheet;
  sgut.src = fitSheet ? r.sheetfitgut : r.sheetgut;
  strip.onerror = () => {{ $("#err").textContent =
    "Cannot load " + r.strip + " -- keep the ff_figs_webp folder beside this HTML file."; }};
  strip.onload = () => {{ fitWindow(); scroll.scrollLeft = 0; showTime(); }};
  /* Vertical start position: the driven patch is the middle band of the sheet
     (rows (n-band)/2 .. +band), so a top-of-sheet start shows undriven tissue
     and the reader has to hunt for the action.  Open centred on the driven
     band instead, once the sheet's real height is known. */
  sheet.onload = () => {{
    fitWindow();
    /* Only pan in full mode: in fit mode the whole sheet is on screen, so
       scrolling to the driven band would push the montage off the top for no
       gain. */
    scroll.scrollTop = fitSheet ? 0 : Math.max(
      0, sheet.clientHeight * 0.5 - (scroll.clientHeight - strip.clientHeight) * 0.5);
    showTime();
  }};
  $("#runid").textContent = r.id;
  /* Deliberately NOT shown: r.cls (the montage class) and the measured peak
     recruitment / duty / rate. All three come from the same automated pipeline
     as r.verdict, so displaying them while the reader votes leaks the answer as
     effectively as the verdict itself would. They are all in the exported CSV. */
  $("#stats").textContent = "";
  document.querySelectorAll(".v").forEach(el => {{
    const on = el.dataset.v === st.verdict;
    el.classList.toggle("on", on);
    el.querySelector("input").checked = on;
  }});
  if ($("#movwrap").classList.contains("on")) movShow();
  const n = Object.values(S).filter(x => x.verdict).length;
  $("#prog").textContent = (i + 1) + " / " + RUNS.length + "   \\u00b7   " + n + " scored";
  $("#bar i").style.width = (100 * n / RUNS.length) + "%";
  showMine();
}}
function showMine() {{
  const r = RUNS[i], st = S[r.id] || {{}};
  const m = $("#mine"), a = $("#agree");
  if (!revealed) {{ m.textContent = "my call: hidden"; m.style.color = "";
                   a.textContent = ""; return; }}
  m.textContent = "my call: " + r.verdict;
  m.style.color = OCOL[r.verdict] || "";
  if (!st.verdict) {{ a.textContent = ""; return; }}
  const ok = st.verdict === r.verdict;
  a.textContent = ok ? "agree" : "DISAGREE";
  a.style.color = ok ? "#2e7d32" : "#c0392b";
}}

/* Undo, added 2026-08-16: a mis-click previously survived into the export
   because nothing echoed the choice back and nothing could take it back. Now
   the last vote is remembered and `u` restores the value it replaced (or
   clears it if there was none). One level is enough -- the failure being
   guarded is a slip on the run in front of you, not a rethink of the set. */
let lastVote = null;
function undo() {{
  if (!lastVote) return;
  const {{id, prev}} = lastVote;
  if (prev === undefined || prev === null) delete S[id]; else S[id] = prev;
  lastVote = null;
  save();
  const at = RUNS.findIndex(r => r.id === id);
  if (at >= 0) i = at;
  draw();
}}

function score(v) {{
  const id = RUNS[i].id;
  lastVote = {{id: id, prev: S[id] ? Object.assign({{}}, S[id]) : null}};
  /* revealed_before_scoring travels with every vote: a score cast after the
     reveal is NOT an uncued read and must be separable in the analysis, not
     silently pooled with the blind ones. */
  S[id] = Object.assign({{}}, S[id], {{verdict: v, at: new Date().toISOString(),
                                     revealed_before_scoring: revealed}});
  save();
  draw();
}}

/* ---- movie pane -------------------------------------------------------
   Two tiers exist because the stored data cannot support one movie that does
   both jobs, and the captions say so explicitly rather than leaving the reader
   to assume a movie shows everything:

     overview  80 ms/frame, whole 80 s run, 2.0x real time.  Shows recruitment
               and spread.  CANNOT show the discharge rhythm: that is 16.7 Hz
               on the ictal runs and 80 ms sampling is 0.7 frames per cycle,
               below Nyquist -- any flicker at this rate is an aliasing
               artefact, not the rhythm.
     detail    20 ms/frame from the RAW field, a 6 s window, 0.5x real time.
               3 samples per discharge cycle, so the rhythm and the inward wave
               are real.  Only available where rate_full was retained. */
let movKind = "overview";
function movShow(kind) {{
  const r = RUNS[i], m = r.mov || {{}};
  if (kind && m[kind]) movKind = kind;
  if (!m[movKind]) movKind = m.overview ? "overview" : (m.detail ? "detail" : null);
  const vid = $("#mov"), note = $("#movnote");
  ["overview", "detail"].forEach(k => {{
    const btn = $("#mov-" + k);
    btn.disabled = !m[k];
    btn.classList.toggle("sel", k === movKind);
    btn.title = m[k] ? "" : "not rendered for this run (needs the raw field)";
  }});
  if (!movKind) {{
    vid.removeAttribute("src"); vid.load();
    note.textContent = "No movie for this run. Render with: python render_movies.py";
    $("#movmeta").textContent = "";
    return;
  }}
  if (vid.getAttribute("src") !== m[movKind]) {{
    vid.setAttribute("src", m[movKind]); vid.load();
  }}
  $("#movmeta").textContent = r.id;
  note.innerHTML = movKind === "overview"
    ? "Whole 80 s run at 80 ms per frame, 2.0&times; real time. Shows recruitment "
      + "and spread. <b>Does not resolve the discharge rhythm</b> \u2014 that is "
      + "~16.7 Hz on the ictal runs, i.e. 0.7 frames per cycle here, below "
      + "Nyquist; flicker at this scale is aliasing, not the rhythm. Colour is "
      + "the same fixed 0\u2013200 Hz magma scale as the heatmap above."
    : "<b>A 6 s window, not the whole run</b> (18\u201324 s), from the raw 5 ms "
      + "field at 20 ms per frame, 0.5&times; real time. 3 samples per discharge "
      + "cycle, so individual discharges and the inward wave are real structure. "
      + "The trace lane below the sheet spans only this window.";
}}
function movToggle(on) {{
  const w = $("#movwrap");
  const show = on === undefined ? !w.classList.contains("on") : on;
  w.classList.toggle("on", show);
  if (show) movShow(); else {{ $("#mov").pause(); }}
}}

function go(d) {{ i = Math.min(RUNS.length - 1, Math.max(0, i + d)); draw(); }}

document.querySelectorAll(".v").forEach(el =>
  el.addEventListener("click", () => score(el.dataset.v)));
["overview", "detail"].forEach(k =>
  $("#mov-" + k).addEventListener("click", () => movShow(k)));
$("#movclose").onclick = () => movToggle(false);
$("#prev").onclick = () => go(-1);
$("#next").onclick = () => go(1);
$("#reveal").onclick = () => {{
  revealed = !revealed;
  $("#reveal").textContent = revealed ? "hide mine" : "reveal mine";
  showMine();
}};
$("#csv").onclick = () => {{
  const rows = [["run_key", "your_call", "my_call", "agree",
                 "revealed_before_scoring", "montage_class", "peak_recruit_frac",
                 "duty_median", "mean_hz", "scored_at"]];
  RUNS.forEach(r => {{
    const st = S[r.id] || {{}};
    rows.push([r.id, st.verdict || "", r.verdict,
               st.verdict ? (st.verdict === r.verdict ? "agree" : "disagree") : "",
               st.verdict ? String(!!st.revealed_before_scoring) : "",
               r.cls, r.peak, r.duty, r.hz, st.at || ""]);
  }});
  const csv = rows.map(rw => rw.map(c => {{
    const s = String(c === undefined || c === null ? "" : c);
    /* The class and the newline are matched via \\n and \\x22 rather
       than written literally: this JS lives inside a PYTHON triple-quoted
       template, so a single-backslash "\n" is consumed by the Python source
       parser and emitted as a REAL newline -- which lands inside this regex
       literal and breaks the whole script at parse time (the page then renders
       as bare alt text because draw() never runs). Any escape written here
       must be doubled. */
    return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }}).join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {{type: "text/csv"}}));
  a.download = "liou_fullfield_scores.csv";
  a.click();
}};
scroll.addEventListener("scroll", showTime);
seek.addEventListener("input", () => {{
  const max = Math.max(1, scroll.scrollWidth - scroll.clientWidth);
  scroll.scrollLeft = max * seek.value / 1000;
  showTime();
}});
window.addEventListener("resize", fitWindow);
document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
  const k = e.key;
  if (k === "z") {{ nativeZoom = !nativeZoom; fitWindow(); e.preventDefault(); return; }}
  if (k === "h") {{
    /* h swaps which RASTER is shown, so the images must be re-sourced, not just
       re-fitted -- draw() does both. Scroll position is preserved in fit mode
       (nothing to pan) and reset to the driven band in full mode. */
    fitSheet = !fitSheet; draw(); e.preventDefault(); return;
  }}
  if (k === "[" || k === "]") {{
    split = Math.min(0.75, Math.max(0.15, split + (k === "]" ? 0.05 : -0.05)));
    fitWindow(); e.preventDefault(); return;
  }}
  if (k >= "1" && k <= String(OPTS.length)) {{ score(OPTS[+k - 1]); e.preventDefault(); return; }}
  if (k === "r") {{ $("#reveal").click(); e.preventDefault(); return; }}
  if (k === "m") {{ movToggle(); e.preventDefault(); return; }}
  if (k === "u") {{ undo(); e.preventDefault(); return; }}
  const map = {{
    "ArrowDown": () => go(1), "j": () => go(1),
    "ArrowUp": () => go(-1), "k": () => go(-1),
    "ArrowRight": () => pan(WINDOW_S), ">": () => pan(WINDOW_S), ".": () => pan(1),
    "ArrowLeft": () => pan(-WINDOW_S), "<": () => pan(-WINDOW_S), ",": () => pan(-1),
    "Home": () => {{ scroll.scrollLeft = 0; showTime(); }},
    "End": () => {{ scroll.scrollLeft = scroll.scrollWidth; showTime(); }},
  }};
  if (map[k]) {{ map[k](); e.preventDefault(); }}
}});
draw();
</script>
"""


if __name__ == "__main__":
    p, n, checked = build()
    print("%s: %d runs | JS checked by %s%s" % (
        p, n, checked,
        "" if checked == "esprima" else "  (WEAK -- pip install esprima for a real parse)"))
