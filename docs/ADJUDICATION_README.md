# Blind ictal adjudication -- Liou pilot (54 runs)

## What to do

1. Unpack `liou_adjudication.zip` and open **liou_adjudication.html** from the
   unpacked folder (no internet needed). Keep `adj_figs_webp/` beside it -- the
   page references the images as files rather than inlining them, which is what
   keeps a 43 MB figure set openable.
2. **Read each run at 10 s per screen.** The montage scrolls horizontally;
   `>` / `<` (or the arrow keys) page forward and back by one 10 s screen, `.`
   and `,` nudge by 1 s, `Home` / `End` jump to the run's start and end. The
   window position is shown above the trace. The channel labels and the scale
   legend are pinned on the left and do not scroll away.
3. For each run, choose one: **seizure / no-seizure / entrained / unsure**.
   Keys `1 2 3 4`; the page advances automatically and echoes the previous call
   in the header so a mis-key is visible. `k` / `j` (or up/down) move between
   runs, `u` jumps to the next unscored one.
4. Click **export CSV** when the tally shows 54 scored.
5. Send me the CSV (or run `python join_adjudication_scores.py <that file>`).

Progress autosaves in the browser, so a closed tab loses nothing. The CSV is the
authoritative record.

## What the runs are

The page is **blind**: labels are neutral (`R01`..`R54`) in shuffled order, and it
carries no parameter values, no metrics, and no automated verdict. The mapping to
run keys is in `liou_pilot_blind_key.csv` -- **do not open it until scoring is
done**, since knowing which runs are the hand-picked anchors would contaminate the
remaining reads. That independence is the whole value of the exercise: it tests the
automated classifier against an uncued expert read.

## The three categories

| verdict | what it means here |
|---|---|
| seizure | an ictal event: evolves over time (amplitude and/or frequency), self-sustained, and **not time-locked to the input** -- it has its own rhythm rather than following the stimulus. Spread beyond the driven patch is *supporting* evidence, not a requirement: a seizure confined to the driven patch is still a seizure. |
| no-seizure | interictal spiking, quiet, or depolarization block -- active but not an event |
| entrained | discharges locked to the 250 ms image-presentation rhythm: a stimulus-following response, not an autonomous event |
| unsure | genuinely ambiguous; flagged for a second look rather than forced |

**Why `entrained` is a separate category and not a kind of seizure.** In the
pilot, a majority of the runs with large-amplitude rhythmic discharges fire at an
integer multiple of the 250 ms image period. Amplitude and interval-variability
thresholds alone score those as ictal, which is wrong -- they are the network
following the stimulus. Any criterion we adopt has to reject integer-locked runs
before applying amplitude tests.

## Reading each page

* **The montage (scrolling, 30 channels).** Green = electrode inside the driven
  digit patch (13 ch), grey = undriven surround (17 ch). Ordered by position on
  the 56x56 sheet (row, then column), labelled `row,col` in the pinned left
  gutter.
* **All scales are fixed across all 54 runs** -- trace gain, mean rate, and
  [Cl-]_in -- so amplitudes and state levels are directly comparable between
  runs, not just between channels within a run. A run that reads as a near-flat
  line genuinely is quiet; that is a finding, not a rendering failure.
  Following EEG practice, the gain is set so a typical trace fills its lane and
  brief large events are allowed to overlap the neighbouring trace; only genuine
  saturation clips, drawn in orange, with the channel count annotated on the
  whole-run panel. The scale bar in the gutter gives the lane height in LFP units.
* **Light vertical rules mark 1 s; the faint rules mark image onsets** -- judge
  time-locking against those.
* **The strip is five lanes on ONE time axis** -- montage, space-time heatmap,
  recruitment fractions, mean firing rate, [Cl-]_in -- rendered as a single wide
  image inside a single scrolling viewport. There is therefore no way for them to
  drift out of register: what is vertically aligned *is* simultaneous. The two
  state lanes carry their own fixed ranges (printed in the gutter).
  **These two are whole-network; the traces above them are not** -- see "What
  each panel can and cannot see" below.
* **Note on adjacency.** The montage steps diagonally through the sheet (stride
  233), so driven channels are *not* contiguous in trace order -- the longest run
  of adjacent driven lanes is 6 of the 13. Do not judge spread by how many
  neighbouring lines are active; use the recruitment map, which plots each
  electrode at its true position on the 56x56 sheet.
* **Space-time heatmap (a lane of the strip).** Every one of the 30 electrodes as
  one row, log10 RMS envelope in 80 ms bins, row order identical to the montage
  (green bars in the pinned gutter mark the driven rows). The colour scale is fixed
  across all 54 runs, so brightness is comparable between runs. Spread reads
  directly here: activity confined to the green rows is focal, brightening that
  fills the non-driven rows is generalized. **It shares the montage's time axis
  and scrolls with it** (changed 2026-08-14), so the bin under a discharge you are
  looking at is the bin directly below it -- previously this was a separate
  whole-run panel and the same instant had to be found twice, at two different
  time scales.
* **Recruitment lane (under the heatmap, same time axis).** Fraction of driven (solid green) and
  of non-driven (dashed grey) electrodes exceeding a fixed amplitude threshold,
  per bin. `all` on the y-axis means every electrode in that group is over
  threshold; the axis carries deliberate empty headroom above it, so a curve
  pinned at `all` sits clear of the frame instead of hiding under it. The two
  curves converging is recruitment of the surround; the driven curve alone
  staying high is a focal discharge; both collapsing to zero is termination or
  block; **both pinned at `all`** is whole-sheet recruitment -- the dashed grey
  under the green is what tells you the second group is there too.
* **Recruitment map (right, 2nd panel).** Channel RMS over the last 20 s, drawn at
  true sheet position, marker area proportional to RMS. This is where focal
  (large markers confined to the patch) separates from generalized (surround
  markers comparable to driven ones).
* **Whole-run overview (diagnostics, left).** Driven-channel mean across the
  entire run at the same fixed gain -- for seeing onset, evolution, and
  termination at a glance before scrolling in for morphology.
* **Slow states (diagnostics, right).** [Cl-]_in and g_K over the whole run at
  fixed axes -- the two variables carrying the model's seizure mechanism. A run
  that ramps chloride and then collapses its rate is entering depolarization
  block, not seizing.

## Files

| file | role |
|---|---|
| liou_adjudication.html | the scoring page (~10 kB; reads figures from `adj_figs_webp/`) |
| adj_figs_webp/ | per-run strip / gutter / diagnostics images (lossless WebP) -- must stay beside the HTML |
| join_adjudication_scores.py | joins the exported CSV to run keys + automated classes, prints the confusion table |
| build_adjudication_html.py | rebuilds the page from `adj_figs_webp/` |
| render_adjudication_figs.py | renders the per-run figures from `pilot_out_v2/*.npz` |
| render_all_runs.py | batch-renders all 54 runs and writes `adj_fig_meta.csv` |
| liou_pilot_blind_key.csv | blind_id -> run_key (open after scoring) |

## What each panel can and cannot see

The panels do not all look at the same thing, and the difference changes what a
quiet panel is allowed to prove.

**The LFP panels are a sparse 30-site montage, not a picture of the network.**
The montage, the space-time heatmap, the recruitment lane and the recruitment map
all come from the same 30 simulated electrodes on a 56x56 = 3136-unit sheet --
**0.96% of the units**. Each electrode is not a point sample: it is an
exponentially weighted sum over nearby tissue, `w = exp(-r/L)` with e-folding
length `L = sigma_LFP * n = 0.025 * 56 = 1.4 units` -- ~46 units' worth of
effective support (participation ratio), with 90% of its weight inside a radius
of ~5.4 units. Taken together the 30
footprints cover roughly 59% of the sheet at that radius, and the coverage is
*uneven* -- summed sensitivity varies 625-fold across the sheet, and 62.5% of
units sit below a tenth of peak sensitivity. **Caveat on the spatial constant.** These runs used `sigma_LFP = 0.025`, the
`lfp_2d` default -- the value that keeps the LFP kernel on the same normalised
convention as `sigma_E`/`sigma_I`. It is *not* the 0.11 that was chosen on
2026-08-12 to make the 2-D electrode integrate the same network fraction as
Liou's 1-D N=2000 electrode; `readout.lfp_montage` passes no `sigma`, so the
decision never reached the pilot. At 0.11 the montage would be far less local
(L = 6.2 units, ~880 units of effective support, full coverage, sensitivity
near-uniform). Score these runs as a deliberately local recording; the
comparison against a wider electrode is an open item, not something these
figures answer.

**The two thin state lanes are whole-network means.** Mean firing rate, [Cl-]_in
and g_K are averaged over *every* unit on the sheet, not over the electrodes.

Practical consequences when scoring:

* A flat montage does **not** establish that the network is quiet. Activity in a
  low-sensitivity gap between electrodes is genuinely under-represented. What a
  flat montage establishes is that there is no discharge *at the recorded sites*.
* A rising whole-field [Cl-]_in or mean rate **is** network-wide by construction,
  so those lanes are the better evidence for a global state change -- and they can
  move while the montage stays quiet, which is not a contradiction.
* Conversely, the seizure criterion is deliberately built on the montage, because
  it asks about discharge morphology *within the driven area*, where electrode
  coverage is densest. Score morphology from the montage; treat the state lanes
  as context, not as a substitute.

## A note on overlapping traces

Because the gain is fixed across runs, the loudest runs (the large-amplitude
rhythmic ones) draw traces that cross several lanes. That is deliberate: the
alternative -- clipping at the lane spacing -- would render the typical run as a
near-flat line and destroy amplitude comparability between runs, which is the
thing the fixed scale exists to preserve. On those runs, read rhythm and
morphology from the montage and read *spatial extent* from the recruitment map
rather than trying to attribute individual lines to channels.
