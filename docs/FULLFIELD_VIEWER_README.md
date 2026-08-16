# Liou pilot -- full-field viewer

`liou_fullfield.html` shows the **same 12 runs** analysed in
`FULLFIELD_FINDINGS.md`, with the 30-electrode LFP montage above a space-time
heatmap of **all 3136 units** on the 56x56 sheet.

Unzip and open `liou_fullfield.html`. Keep the `ff_figs_webp/` folder beside it.

## What is on screen

* **LFP montage (top, pinned).** The same 30 electrodes, the same fixed gain
  (24x10^3 per lane) and the same clip colour as the adjudication page, so a
  run reads identically on both. Green = electrode inside the driven patch.
  Below it, mean rate and [Cl-]_in on the same time axis at their fixed ranges.
* **Full-resolution sheet (below).** One pixel row per unit -- 3136 rows, in
  row-major order: units 0..55 are sheet row 0, units 56..111 are sheet row 1,
  and so on. A spatially contiguous blob on the sheet therefore appears as a
  block of adjacent lanes, and the 56-row periodicity you see in an organised
  run is the sheet's own row structure, not an artefact.
* Colour is firing rate, **fixed across all 12 runs** at 0 to 200 Hz (= f_max),
  in 80 ms max-envelope bins. The paper's Figure 6 ceiling of 80 Hz is marked
  on the gutter ramp.

## Reading it

* The LFP strip and the sheet are **one scroll box and one CSS grid**, gutters
  included: they share a single horizontal scroll position *and* a single gutter
  column, so any x is the same instant in both -- the two x-axes line up to the
  pixel. (Until 2026-08-16 the montage row was a nested grid, which sized its
  gutter independently: the panes' t=0 sat 76 px apart, 0.88 s of apparent
  shift.) Scrolling *down* pans through the sheet while the montage stays pinned
  at the top, and each gutter travels with its own image so labels never drift
  off their lanes.
* **Both panes are fully visible at once, at roughly 50/50.** On a 1080x1920
  portrait display that is ~870 px each: 29 px per LFP lane (the adjudication
  page uses 24) and the whole 56x56 sheet with nothing below the fold. `[` and
  `]` change the split (15-75%); the montage's 30 lanes compress to fit the
  chosen share, which is a lane-spacing change only -- trace width and the time
  axis are unaffected.
* `h` switches the sheet between the **whole-sheet overview** (default) and
  **one screen pixel per unit** (scroll to pan). These are two different
  rasters, not the same image at two sizes -- see below.
* `<` `>` page +/-10 s, `,` `.` nudge 1 s, `Home`/`End`, `z` toggles native
  time resolution, `h` toggles a whole-sheet overview (fits all 3136 rows on
  screen at reduced vertical resolution).
* `k`/`j` or up/down arrows move between runs. Each run opens **centred on the
  driven band** rather than at the top of the sheet, since that is where events
  start.

## Scoring

Classify each run with the radio buttons or keys `1`-`5`:

1. ENTRAINED, not a seizure
2. GENERALIZED SEIZURE (non-Liou morphology)
3. FOCAL SEIZURE (Liou-like)
4. TONIC SATURATION (not a seizure)
5. OTHER

The first four are the classes in `fullfield_verdicts.csv`; OTHER covers
morphologies not yet encountered and has no automated counterpart, so a run
scored OTHER never counts as agreement.

**My own classification is hidden until you press `r` (or "reveal mine").** The
acceptance criterion for this project is agreement between an *uncued* read and
the automated verdict, so showing my call -- or the montage class, or the
measured recruitment/duty/rate, all of which come from the same pipeline --
would anchor the judgement it is meant to test. Every vote records whether the
reveal was open when it was cast (`revealed_before_scoring` in the CSV), so a
cued score can be separated from a blind one rather than silently pooled.

Scores autosave in the browser. **Export CSV** writes `run_key`, `your_call`,
`my_call`, `agree`, `revealed_before_scoring`, the montage class and the sheet
metrics.

## Sheet movies (`m`)

Press `m` for a movie of the sheet, analogous to the video accompanying Figure 1
of Liou et al. 2020. Two tiers, because the stored data cannot support one movie
that does both jobs:

| tier | frame | span | speed | shows | limit |
|---|---|---|---|---|---|
| **overview** | 80 ms | whole 80 s run | 2.0x | recruitment, spread, wave fronts | **cannot show discharges** |
| **detail** | 20 ms | 6 s window (18-24 s) | 0.5x | individual discharges, inward wave | 6 s of the 80 s |

The overview limit is measured, not cautious wording: the discharge rhythm on
the ictal runs is **16.7 Hz** (median IPI 60 ms), so 80 ms bins give 0.75 samples
per cycle -- below Nyquist. The overview movie genuinely cannot show the rhythm,
and any flicker at that scale is an aliasing artefact. The detail tier's 20 ms
gives 3 samples per cycle, which resolves it.

**All 12 runs now have both tiers.** The 11 that had only the envelope were
re-run on 2026-08-16 with `keep_full=True`; each passed the bit-for-bit
determinism check against its scored montage in `pilot_out_v2`, and every
recruitment/duty metric came out identical to the original, so these are the
same realizations that were scored -- not new simulations that resemble them.

Colour is the same fixed 0-200 Hz magma scale as the page's heatmap, so a movie
frame and a heatmap column are directly comparable. The lane under the sheet is
the recruited fraction with a playhead.

Regenerate with `python render_movies.py`; movies are optional and the page
builds and scores without them.

## Why the overview is a separate image

Fitting 3136 unit rows into ~870 px needs a 3.6x vertical reduction. Letting the
browser do it would be nearest-neighbour: it *samples* ~800 of the 3136 rows and
discards the rest. Measured across all 12 runs, that blanks **21 time bins that
contain real activity** -- and the losses concentrate in the sparse runs (9 bins
in `anchor_gKmax120_seq`, 7 in `anchor_gIlocal600_seq`, 4 in `draw04_seq`),
i.e. exactly the runs where one missed discharge changes the verdict.

So the overview is rendered as a **max pool** over each group of 4 units, which
loses zero bins: a lit unit always lights its pooled row. The cost is that the
overview *overstates* spatial extent by up to 4 units, which is why `h` still
gives the true one-row-per-unit image for judging focality. Both rasters share
the same time axis and the same fixed colour scale.

## Why the colour scale is 200 Hz, not the paper's 80

At an 80 Hz ceiling, 98% of `anchor_gE220_seq` and 42% of `draw16_seq` sit
above the top of the scale, so those maps carried no information -- they were a
flat field of over-colour. The full 0-200 Hz range keeps every run readable and
the paper's level is marked on the ramp instead.

## What this viewer settles

The three `QUIET_OR_BLOCK` runs read as a flat montage and a **uniformly bright
sheet**: depolarization block, not quiescence. `draw16` generalizes within a
second. `anchor_default_{seq,sim}` show a focal patch that expands. That is the
whole argument of `FULLFIELD_FINDINGS.md`, visible directly.

## Files

| file | purpose |
|---|---|
| liou_fullfield.html | the viewer (self-contained; no network) |
| ff_movies/ | sheet movies, .mp4 (h264); optional |
| ff_figs_webp/ | 12 runs x 6 images (strip, gutter, sheet, sheetgut, sheetfit, sheetfitgut), lossless WebP |
| render_fullfield_page.py | builds the four images per run |
| build_fullfield_html.py | builds the page |

Regenerate with:

    PYTHONPATH=. python render_fullfield_page.py
    PYTHONPATH=. python build_fullfield_html.py
