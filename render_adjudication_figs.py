"""Render per-run adjudication panels: a wide scrollable LFP strip + fixed diagnostics.

WHY THIS SHAPE (changed 2026-08-14 at the reviewer's request):

1. FIXED SCALES ACROSS RUNS.  Every run is drawn at the same LFP gain, the same
   mean-rate y-range and the same [Cl-]_in y-range, so amplitude is comparable by
   eye from screen to screen.  Previously each run was auto-scaled, which made a
   quiet run and an ictal run look equally tall.  The cost of a common scale is
   that extreme runs clip; clipped samples are drawn in a distinct colour and
   counted in an annotation, so a clipped trace can never be mistaken for a
   saturated one.  Scale constants live in SCALES and are computed from the
   pooled run set by choose_scales() -- not hand-picked.

2. 10 SECONDS PER SCREEN.  EEG is read at ~10 s/page; at 80 s across 1200 px a
   3 Hz discharge train is 240 cycles in 1200 px and morphology is unreadable.
   The montage is therefore rendered as ONE WIDE image at a fixed px/s and shown
   in a horizontally scrolling viewport, so the reader pans through the run at
   constant time resolution instead of seeing it pre-compressed.

3. STATE TRACES SCROLL WITH THE LFP.  mean rate and [Cl-]_in share the strip's
   time axis, because the question "is the rate collapsing while chloride ramps"
   has to be answered at the same instant as the LFP being inspected.

4. THE LEFT GUTTER IS A SEPARATE IMAGE, pinned outside the scrolling viewport.
   Channel labels drawn inside the wide strip would scroll off-screen, leaving an
   unlabelled trace stack -- unreadable for judging where an event sits.

Electrode geometry: the 30 electrodes are flat indices at stride 233, wrapped
onto the sheet with ``EL30 % (n*n)`` exactly as RO.lfp_montage does.  The sheet
width n is NOT stored in the run npz, so it is recovered per run by inverting
RO.driven_channels (see infer_sheet_n) -- it was previously hardcoded to 84 while
the pilot runs were executed at n=56, which mislabelled 29 of 30 electrode
positions and made "montage order" the identity order rather than a spatial one.
Consecutive electrodes step diagonally and the driven channels are NOT contiguous
in trace order, so the WC-MNIST ">=7 contiguous channels" criterion does not
transfer; adjacency must be judged in 2-D, which is what the recruitment map is
for.

SPATIAL SUPPORT differs between panels and this is not cosmetic.  The LFP panels
(strip, heatmap, recruitment lane/map) come from `L` -- 30 Gaussian-weighted
electrode sites, ~46 effective units each, covering a minority of the sheet
unevenly.  The rate / [Cl-] / g_K traces are WHOLE-FIELD means over every unit.
A quiet montage therefore does not prove a quiet network, while a rising
whole-field chloride level is network-wide by construction.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import las_model.readout as RO

IMAGE_PERIOD_S = 0.250
GREEN = "#1b7a4b"
GREY = "#9a9a9a"
CLIP = "#d95f0e"          # clipped samples: never confusable with a real trace
HEAT_BIN_S = 0.080        # envelope bin: resolves the 250 ms image period
HEAT_LOG_MIN = 0.0        # fixed log10 colour limits, common to every run
HEAT_LOG_MAX = 4.5
HEAT_RECRUIT_THR = 1000.0  # fixed envelope threshold for the recruitment lane
PX_PER_S = 118.0          # 10 s window ~= 1180 px of traces
WINDOW_S = 10.0

# Fixed display scales, from choose_scales() over the pooled run set.
# LANE is the lane SPACING; traces clip at +/-LANE/2*CLIP_MULT, i.e. adjacent
# lanes are allowed to overlap slightly rather than clipping at the spacing.
# WHY DECOUPLED: clipping at the spacing forces a choice between flattening the
# median run and destroying the loudest one -- measured over all 54 runs, a lane
# that keeps the worst run under 1% sample loss leaves the typical run using 30%
# of its height. Real EEG montages solve this the same way: sensitivity is set so
# typical traces fill their lane, brief large events overlap the neighbour, and
# only genuine saturation clips. At LANE=24000 / CLIP_MULT=1.5 the median run
# clips 0.009% of samples and the loudest 15.4% (annotated), lane overlap is
# 0.22% of samples for the median run, and only the 2 genuinely silent runs read
# as flat -- which is the correct reading for them, not a rendering failure.
SCALES = dict(LANE=24000.0, CLIP_MULT=1.5, RATE_MAX=0.70, CL_MIN=0.0,
              CL_MAX=55.0, GK_MAX=3.5)


def infer_sheet_n(driven, candidates=range(20, 141)):
    """Recover the sheet width n from a run's own driven-channel mask.

    WHY THIS EXISTS: the sheet size is NOT stored in the run npz (`summary["n"]`
    is the discharge COUNT), and it was previously hardcoded to 84 here while
    every pilot run was executed at n=56.  `RO.lfp_montage` wraps the electrode
    indices with ``EL30 % (n*n)``, so the wrong n silently relabels 29 of the 30
    electrodes -- by up to 74 rows / 56 columns -- with no error anywhere.

    `RO.driven_channels(n)` is a pure function of n, and the mask it returns is
    stored in the run, so n is recoverable by inversion.  Over 20..140 the mask
    is unique to a single n for all 54 pilot runs, which makes this a check
    rather than a guess: an ambiguous or empty match raises instead of drawing
    electrodes at fabricated positions.
    """
    driven = np.asarray(driven, dtype=bool)
    hits = []
    for n in candidates:
        try:
            _, m = RO.driven_channels(n=n)
        except Exception:
            continue
        if len(m) == len(driven) and bool((np.asarray(m, bool) == driven).all()):
            hits.append(int(n))
    if len(hits) != 1:
        raise ValueError(
            "cannot recover sheet size from the driven mask: %d candidates %s. "
            "Electrode (row,col) positions and montage row order both depend on "
            "n, so drawing them without it would fabricate the 2-D geometry."
            % (len(hits), hits))
    return hits[0]


def electrode_positions(n):
    """(row, col) on the n x n sheet for each of the 30 montage electrodes.

    Must use the SAME wrap as RO.lfp_montage (``EL30 % (n*n)``), or the labels
    describe a different sheet than the one the LFP was read from.
    """
    return np.divmod(np.asarray(RO.EL30) % (n * n), n)


def _amp_label(v):
    """Format an LFP amplitude for a scale-bar / clip annotation.

    WHY: the previous code used "%.0f x 10^3" unconditionally, so any run whose
    scale bar was below 500 units printed as "0x10^3" -- an uninterpretable
    scale. Small-amplitude runs are exactly the ones where the reader most needs
    the number (they decide quiet vs. active), so the label must never round to
    zero. Switch units at 1000 and keep one decimal below 10x10^3.
    """
    a = abs(float(v))
    if a >= 1e4:
        return "%.0f$\\times$10$^3$" % (v / 1e3)
    if a >= 1e3:
        return "%.1f$\\times$10$^3$" % (v / 1e3)
    if a >= 10:
        return "%.0f" % v
    return "%.2g" % v


def choose_scales(npz_paths, target_p99_usage=0.65):
    """Recompute the common scales from a pooled run set (audit / re-tune helper).

    The lane is chosen so the MEDIAN run's 99th-percentile |LFP| reaches
    `target_p99_usage` of the half-lane: that fills the typical trace without
    reference to the single loudest run, whose amplitude is ~14x the quietest
    and would otherwise set the gain for everyone.

    rate / [Cl-] / g_K use the pooled MAX, not a percentile: they are slow
    scalars whose ABSOLUTE level carries the meaning (depolarization block,
    chloride ceiling), so clipping them would hide the state being checked for.

    NOTE the t=0 sample is dropped -- the LFP filter's first output is undefined
    (NaN in all 30 channels of every run), and a plain percentile over it
    returns NaN, silently poisoning any scale derived from it.
    """
    p99, rmax, clmax, gkmax = [], [], [], []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as f:
            L = np.asarray(f["L"], dtype=float)[1:]
            L = L - L.mean(axis=0)
            p99.append(np.percentile(np.abs(L), 99))
            rmax.append(float(np.nanmax(f["rate_trace"])))
            clmax.append(float(np.nanmax(f["Cl_in"])))
            gkmax.append(float(np.nanmax(f["g_K"])))
    half = float(np.median(p99)) / target_p99_usage
    return dict(
        LANE=float(np.ceil(2 * half / 1000.0) * 1000.0),
        CLIP_MULT=1.5,
        RATE_MAX=float(np.ceil(max(rmax) * 20.0) / 20.0),
        CL_MIN=0.0,
        CL_MAX=float(np.ceil(max(clmax) / 5.0) * 5.0),
        GK_MAX=float(np.ceil(max(gkmax) * 2.0) / 2.0),
    )


def _order_channels(n):
    """Trace order: by sheet row, then column (spatially meaningful top-to-bottom).

    n is REQUIRED: sorting on positions computed for the wrong sheet returned the
    identity order 0..29, which looks like a plausible montage order but is not
    spatially meaningful -- adjacent lanes were not adjacent tissue.
    """
    rows, cols = electrode_positions(n)
    return np.lexsort((cols, rows)), rows, cols


def _envelope(L, t, bin_s):
    """Non-overlapping RMS of each channel in bin_s bins -> ((n_ch, n_bin), bin_s).

    NaN-safe: sample 0 is NaN on every channel in every stored run (one initial
    timestep), which nanmean absorbs without leaving a column artifact.
    """
    dt = float(t[1] - t[0])
    k = max(1, int(round(bin_s / dt)))
    n = (L.shape[0] // k) * k
    X = np.asarray(L[:n], float).reshape(n // k, k, L.shape[1])
    return np.sqrt(np.nanmean(X ** 2, axis=1)).T, k * dt


def _clip_split(y, lim):
    """Return (inside, outside) copies of y, NaN-masked, for two-colour drawing."""
    inside = np.where(np.abs(y) <= lim, y, np.nan)
    outside = np.where(np.abs(y) > lim, np.clip(y, -lim, lim), np.nan)
    return inside, outside


def render_run(npz_path, outdir, label=None, dpi=110, scales=None):
    """Render one run as three images: wide scrolling strip, pinned gutter, diagnostics.

    Returns (paths dict, meta dict).  `label` overrides the displayed title so the
    page can be blinded -- the acceptance test is agreement between an UNCUED read
    and the automated verdict, so the true run key must not appear.
    """
    S = dict(SCALES); S.update(scales or {})
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["summary"]))
    # Drop the t=0 sample: the LFP filter's first output is undefined (NaN in
    # all 30 channels of every run). Left in, it renders as a gap at the left
    # edge of every trace and makes plain percentiles return NaN.
    L = np.asarray(z["L"], dtype=float)[1:]        # (T-1, 30)
    t = np.asarray(z["t"], dtype=float)[1:]
    if t.max() > 1000.0:                            # stored in ms
        t = t / 1000.0
    driven = np.asarray(z["driven"], dtype=bool)
    rate = np.asarray(z["rate_trace"], dtype=float)[1:]
    Cl = np.asarray(z["Cl_in"], dtype=float)[1:]
    gK = np.asarray(z["g_K"], dtype=float)[1:]
    sheet_n = infer_sheet_n(driven)
    order, rows, cols = _order_channels(sheet_n)
    step = float(S["LANE"])                       # lane spacing
    lim = step / 2.0 * float(S["CLIP_MULT"])      # clip threshold (> spacing)
    stem = label if label is not None else meta["key"]
    os.makedirs(outdir, exist_ok=True)
    out = {}

    # ---------------- envelope, needed by the strip (heat + recruitment lanes)
    # Computed BEFORE the strip figure because the space-time heatmap and the
    # recruitment lane now live INSIDE the scrolling strip rather than in a
    # separate figure below it.
    Ehm, hm_bin = _envelope(L, t, HEAT_BIN_S)
    Ghm = np.log10(np.clip(Ehm[order], 10.0 ** HEAT_LOG_MIN, None))
    ab = Ehm > HEAT_RECRUIT_THR
    tb = (np.arange(Ehm.shape[1]) + 0.5) * hm_bin

    # ---------------- wide scrolling strip: LFP + heat + recruit + rate + [Cl-]
    T_s = float(t[-1])
    w_in = max(T_s * PX_PER_S / dpi, 6.0)
    # Panel heights: the LFP gets the bulk; rate and [Cl-] get their OWN thin
    # lanes rather than sharing one twinx axis. On a FIXED scale both slow traces
    # sit near the bottom of their range for most runs, so overlaying them on one
    # axis stacks two flat lines on top of each other -- separate lanes keep each
    # readable at its own fixed range.
    # The heat and recruitment lanes are in THIS figure (not a separate one) so
    # that they cannot drift out of time register with the montage: one image in
    # one scroll box means one scroll position, with no JS to keep in sync.
    # Their heights are chosen so the merged strip costs no net vertical space --
    # the standalone heat figure it replaces occupied ~253 px on screen, and
    # these two lanes occupy ~200 px at the same viewport width.
    h_lfp, h_hm, h_rc, h_r, h_c = 6.6, 1.55, 0.62, 0.78, 0.78
    h_tot = h_lfp + h_hm + h_rc + h_r + h_c + 0.42
    figW = plt.figure(figsize=(w_in, h_tot))
    gsW = figW.add_gridspec(5, 1, height_ratios=[h_lfp, h_hm, h_rc, h_r, h_c],
                            hspace=0.10, left=0.0, right=1.0, top=0.995,
                            bottom=0.075)
    axS = figW.add_subplot(gsW[0])
    axH = figW.add_subplot(gsW[1], sharex=axS)
    axRec = figW.add_subplot(gsW[2], sharex=axS)
    axR = figW.add_subplot(gsW[3], sharex=axS)
    axC = figW.add_subplot(gsW[4], sharex=axS)

    n_clipped = 0
    for k, ch in enumerate(order):
        y0 = L[:, ch] - np.nanmean(L[:, ch])
        base = (len(order) - 1 - k) * step
        ins, outs = _clip_split(y0, lim)
        col = GREEN if driven[ch] else GREY
        axS.plot(t, base + ins, lw=0.42, color=col, zorder=2)
        if np.isfinite(outs).any():
            n_clipped += 1
            axS.plot(t, base + outs, lw=0.7, color=CLIP, zorder=3)
    axS.set_ylim(-step, len(order) * step)
    axS.set_xlim(0.0, T_s)
    axS.set_yticks([])
    # NOT set_xticklabels([]): axR/axC share this x-axis, and blanking the
    # labels here blanks the shared formatter, leaving the whole strip with no
    # time axis at all. labelbottom hides them on THIS axes only.
    axS.tick_params(length=0, labelbottom=False)
    for s_ in ("top", "right", "left"):
        axS.spines[s_].set_visible(False)
    # image-onset ticks: entrainment is judged against these
    for x in np.arange(0.0, T_s, IMAGE_PERIOD_S):
        axS.axvline(x, color="#ececec", lw=0.3, zorder=0)
    for x in np.arange(0.0, T_s + 1e-9, 1.0):
        axS.axvline(x, color="#cfcfcf", lw=0.45, zorder=1)

    # ---- heat + recruitment lanes, on the strip's own scrolling time axis ----
    # Both were previously a separate static figure whose x-axis was unrelated to
    # the scroll position, so a feature at t=43 s in the montage had to be found
    # by eye in a panel showing all 80 s at 1/8 the time resolution.  Drawn here
    # they share axS's x-axis exactly.
    im = axH.imshow(Ghm, aspect="auto", cmap="magma", interpolation="nearest",
                    vmin=HEAT_LOG_MIN, vmax=HEAT_LOG_MAX,
                    extent=[0.0, T_s, len(order), 0.0])
    axH.set_ylim(len(order), 0.0)
    axH.set_yticks([])
    axH.tick_params(labelbottom=False, length=0)
    for s_ in ("top", "right", "left", "bottom"):
        axH.spines[s_].set_visible(False)
    # Driven-row marks are drawn in the PINNED GUTTER, not here: inside the wide
    # image they would scroll off-screen with everything else.

    axRec.plot(tb, ab[~driven].mean(0), lw=1.0, color="#4a4a4a", ls=(0, (3, 2)),
               zorder=3)
    axRec.plot(tb, ab[driven].mean(0), lw=0.7, color=GREEN, zorder=4)
    axRec.axhline(1.0, lw=0.4, ls=(0, (2, 2)), color="#b0b0b0", zorder=1)
    for x in np.arange(0.0, T_s + 1e-9, 1.0):
        axRec.axvline(x, color="#efefef", lw=0.35, zorder=0)
    axRec.set_ylim(-0.08, 1.22)
    axRec.set_xlim(0.0, T_s)
    axRec.set_yticks([])
    axRec.tick_params(labelbottom=False, length=0)
    for s_ in ("top", "right", "left"):
        axRec.spines[s_].set_visible(False)

    for a_, y_, cl_, hi_, lo_ in ((axR, rate, "#333333", S["RATE_MAX"], 0.0),
                                  (axC, np.clip(Cl, S["CL_MIN"], S["CL_MAX"]),
                                   "#b2182b", S["CL_MAX"], S["CL_MIN"])):
        for frac in (0.25, 0.5, 0.75):
            a_.axhline(lo_ + frac * (hi_ - lo_), color="#e4e4e4", lw=0.4, zorder=0)
        for x in np.arange(0.0, T_s + 1e-9, 1.0):
            a_.axvline(x, color="#efefef", lw=0.35, zorder=0)
        a_.plot(t, y_, lw=0.6, color=cl_, zorder=2)
        a_.set_ylim(lo_, hi_)
        a_.set_xlim(0.0, T_s)
        a_.set_yticks([])
        for s_ in ("top", "right", "left"):
            a_.spines[s_].set_visible(False)
    axR.tick_params(labelbottom=False, length=0)
    axC.set_xticks(np.arange(0, T_s + 1e-9, 1.0))
    axC.set_xticklabels(["%d" % v if v % 5 == 0 else "" for v in np.arange(0, T_s + 1e-9, 1.0)])
    axC.tick_params(labelsize=6.6)
    # The axes span the figure edge to edge -- that is REQUIRED, because the page
    # derives seconds-per-pixel from the image width, so any margin would break the
    # window arithmetic. Consequence: the first and last numerals are centred on the
    # figure boundary and half of each gets cut. Align them inward instead of
    # padding the axes.
    _tl = axC.get_xticklabels()
    if _tl:
        _tl[0].set_ha("left")
        _tl[-1].set_ha("right")
    axC.set_xlabel("time (s)", fontsize=7)
    p_strip = os.path.join(outdir, "%s_strip.png" % stem)
    figW.savefig(p_strip, dpi=dpi)
    plt.close(figW)
    out["strip"] = p_strip

    # ---------------- pinned gutter: channel labels + fixed-scale legend
    # Width is set so "(row,col)" labels fit without truncation -- at 1.28 in the
    # two-digit-comma-two-digit labels were being clipped at the left edge.
    # Height and height_ratios MUST match figW exactly: the page scales the
    # gutter by the strip's scale factor, so any difference in the lane stack
    # puts every label out of register with the lane it names.
    figG = plt.figure(figsize=(1.95, h_tot))
    gsG = figG.add_gridspec(5, 1, height_ratios=[h_lfp, h_hm, h_rc, h_r, h_c],
                            hspace=0.10, left=0.02, right=0.995, top=0.995,
                            bottom=0.075)
    axG = figG.add_subplot(gsG[0])
    axGh = figG.add_subplot(gsG[1])
    axGrc = figG.add_subplot(gsG[2])
    axGr = figG.add_subplot(gsG[3])
    axGc = figG.add_subplot(gsG[4])
    axG.set_ylim(-step, len(order) * step)
    axG.set_xlim(0, 1)
    for k, ch in enumerate(order):
        base = (len(order) - 1 - k) * step
        axG.text(0.99, base, "%d,%d" % (rows[ch], cols[ch]), fontsize=5.4,
                 ha="right", va="center",
                 color=GREEN if driven[ch] else "#666666")
    # amplitude scale bar: one LANE tall, in the strip's own units
    axG.plot([0.055, 0.055], [0.0, step], color="#222222", lw=1.4,
             solid_capstyle="butt", clip_on=False)
    axG.text(0.10, step / 2.0, _amp_label(step), fontsize=5.8, rotation=90,
             ha="left", va="center")
    axG.text(0.99, len(order) * step * 0.985, "electrode (row,col)", fontsize=5.4,
             ha="right", va="top", color="#999999", style="italic")
    axG.axis("off")
    # heat lane gutter: driven-row marks + the colour scale, pinned so they stay
    # visible while the image scrolls.
    axGh.set_xlim(0, 1)
    axGh.set_ylim(len(order), 0.0)
    axGh.axis("off")
    for i_ in np.where(driven[order])[0]:
        axGh.add_patch(plt.Rectangle((0.955, i_ + 0.08), 0.030, 0.84,
                                     color=GREEN, lw=0, zorder=5, clip_on=False))
    axGh.text(0.93, len(order) * 0.02, "space$\\times$time envelope", fontsize=5.4,
              ha="right", va="top", color="#555555")
    axGh.text(0.93, len(order) * 0.30, "log$_{10}$ RMS, %d ms bins" % int(round(hm_bin * 1e3)),
              fontsize=4.9, ha="right", va="top", color="#999999", style="italic")
    axGh.text(0.93, len(order) * 0.62, "%s \u2013 %s"
              % (_amp_label(10.0 ** HEAT_LOG_MIN), _amp_label(10.0 ** HEAT_LOG_MAX)),
              fontsize=5.0, ha="right", va="center", color="#555555")
    axGh.text(0.93, len(order) * 0.86, "rows = montage order", fontsize=4.9,
              ha="right", va="center", color="#999999", style="italic")
    axGh.text(0.905, len(order) * 0.50, "driven", fontsize=4.9, color=GREEN,
              rotation=90, ha="left", va="center")
    # Colour ramp swatch, drawn in the gutter's own axes coordinates: a colorbar
    # in the wide figure would scroll away with the data it explains.
    ramp = np.linspace(HEAT_LOG_MIN, HEAT_LOG_MAX, 128).reshape(1, -1)
    axGh.imshow(ramp, cmap="magma", aspect="auto", vmin=HEAT_LOG_MIN,
                vmax=HEAT_LOG_MAX, extent=[0.42, 0.93, len(order) * 0.78,
                                           len(order) * 0.70], zorder=4)

    # recruitment lane gutter
    axGrc.set_xlim(0, 1)
    axGrc.set_ylim(-0.08, 1.22)
    axGrc.axis("off")
    # header shifted left of x=0.945: the "all"/"0" endpoint labels occupy the
    # right edge and this lane is only 0.62 in tall, so a right-aligned title
    # here overprints them.
    axGrc.text(0.72, 1.24, "fraction > %s" % _amp_label(HEAT_RECRUIT_THR),
               fontsize=5.2, ha="right", va="top", color="#555555")
    axGrc.plot([0.955, 0.985], [0.0, 0.0], color="#b0b0b0", lw=0.5, clip_on=False)
    axGrc.plot([0.955, 0.985], [1.0, 1.0], color="#b0b0b0", lw=0.5, clip_on=False)
    axGrc.text(0.945, 0.0, "0", fontsize=5.0, ha="right", va="center", color="#777777")
    axGrc.text(0.945, 1.0, "all", fontsize=5.0, ha="right", va="center", color="#777777")
    axGrc.plot([0.60, 0.72], [0.60, 0.60], color=GREEN, lw=0.7, clip_on=False)
    axGrc.text(0.735, 0.60, "driven", fontsize=4.9, color=GREEN, ha="left", va="center")
    axGrc.plot([0.60, 0.72], [0.28, 0.28], color="#4a4a4a", lw=1.0,
               ls=(0, (3, 2)), clip_on=False)
    axGrc.text(0.735, 0.28, "not driven", fontsize=4.9, color="#4a4a4a",
               ha="left", va="center")

    for a_, lab, lo_, hi_, cl_ in ((axGr, "mean rate", 0.0, S["RATE_MAX"], "#333333"),
                                   (axGc, "[Cl$^-$]$_{in}$ (mM)", S["CL_MIN"],
                                    S["CL_MAX"], "#b2182b")):
        a_.set_xlim(0, 1)
        a_.set_ylim(lo_, hi_)
        # Range printed as text INSIDE the axes, not as y-ticks: labelright
        # draws outside the right spine, which is flush with the figure edge
        # here (the gutter is pinned against the strip), so tick labels were
        # being clipped away entirely.
        a_.axis("off")
        a_.text(0.93, hi_, lab, fontsize=5.6, color=cl_, ha="right", va="top")
        a_.text(0.93, lo_ + 0.42 * (hi_ - lo_), "%g \u2013 %g" % (lo_, hi_),
                fontsize=5.6, color=cl_, ha="right", va="center")
        a_.plot([0.965, 0.965], [lo_, hi_], color=cl_, lw=0.9,
                solid_capstyle="butt", clip_on=False)
    # Sits ABOVE the lane's baseline, not at it: at the bottom of the gutter it
    # collided with the strip's time-axis numerals once the two images were
    # scale-locked.
    axGc.text(0.93, S["CL_MIN"] + 0.16 * (S["CL_MAX"] - S["CL_MIN"]),
              "fixed across runs", fontsize=4.9, color="#999999",
              style="italic", ha="right", va="bottom")
    p_gut = os.path.join(outdir, "%s_gutter.png" % stem)
    figG.savefig(p_gut, dpi=dpi)
    plt.close(figG)
    out["gutter"] = p_gut

    # ---------------- fixed diagnostics: whole-run overview + recruitment map
    figD = plt.figure(figsize=(11.0, 2.55))
    gsD = figD.add_gridspec(1, 3, width_ratios=[3.1, 1.0, 1.5], wspace=0.30,
                            left=0.055, right=0.985, top=0.86, bottom=0.20)
    axO = figD.add_subplot(gsD[0])
    axM = figD.add_subplot(gsD[1])
    axCl = figD.add_subplot(gsD[2])

    # whole-run overview: driven-channel envelope, fixed y, so the reader keeps
    # global context while the strip is zoomed to 10 s
    dm = L[:, driven].mean(axis=1)
    dm = dm - np.nanmean(dm)
    axO.fill_between(t, np.clip(dm, -lim, lim), 0, lw=0, color="#7f7f7f")
    axO.set_xlim(0.0, T_s); axO.set_ylim(-lim, lim)
    axO.set_yticks([-lim, 0, lim])
    axO.set_yticklabels(["-" + _amp_label(lim), "0", _amp_label(lim)], fontsize=5.6)
    axO.set_xlabel("time (s)", fontsize=6.4); axO.tick_params(labelsize=6.0)
    axO.set_title("whole run, driven-channel mean (fixed scale)", fontsize=7.2, loc="left")

    late = t >= max(t[-1] - 20.0, 0.0)
    rms = np.sqrt(np.nanmean(L[late] ** 2, axis=0))
    sz = 24.0 + 300.0 * np.clip(rms / lim, 0, 1)
    axM.scatter(cols[~driven], rows[~driven], s=sz[~driven], c=GREY,
                edgecolor="white", lw=0.4, zorder=2)
    axM.scatter(cols[driven], rows[driven], s=sz[driven], c=GREEN,
                edgecolor="white", lw=0.4, zorder=3)
    # Axes span the ACTUAL sheet: fixed 0..84 ticks under n=56 data both
    # mislabelled the coordinates and left a third of the panel as dead margin.
    pad = max(4, int(round(sheet_n * 0.05)))
    axM.set_xlim(-pad, sheet_n + pad); axM.set_ylim(sheet_n + pad, -pad)
    axM.set_aspect("equal")
    half = sheet_n // 2
    axM.set_xticks([0, half, sheet_n]); axM.set_yticks([0, half, sheet_n])
    axM.tick_params(labelsize=5.4)
    axM.set_title("recruitment, last 20 s  (%d$\\times$%d sheet)\n(area $\\propto$ RMS, fixed)"
                  % (sheet_n, sheet_n), fontsize=7.0, loc="left")

    axCl.plot(t, Cl, lw=0.6, color="#b2182b")
    axCl.set_ylim(S["CL_MIN"], S["CL_MAX"])
    axCl.set_ylabel("[Cl$^-$]$_{in}$", color="#b2182b", fontsize=6.4)
    axCl.tick_params(axis="y", labelcolor="#b2182b", labelsize=5.8)
    axCl.tick_params(axis="x", labelsize=5.8)
    axCl.set_xlim(0.0, T_s); axCl.set_xlabel("time (s)", fontsize=6.4)
    axKK = axCl.twinx()
    axKK.plot(t, gK, lw=0.6, color="#2166ac")
    axKK.set_ylim(0.0, S["GK_MAX"])          # fixed, and NOT auto: g_K spans
    axKK.set_yticks([0.0, S["GK_MAX"] / 2.0, S["GK_MAX"]])   # 0.05-3.3 across runs,
    axKK.set_yticklabels(["0", "%.2g" % (S["GK_MAX"] / 2.0), "%.2g" % S["GK_MAX"]])
    axKK.set_ylabel("g$_K$", color="#2166ac", fontsize=6.4)  # and the default
    axKK.tick_params(axis="y", labelcolor="#2166ac", labelsize=5.8)  # formatter
    # printed every tick as "0".
    axCl.set_title("slow states (fixed [Cl$^-$] axis)", fontsize=7.0, loc="left")

    if n_clipped:
        axO.text(0.995, 0.03, "%d of 30 channels clip at $\\pm$%s" % (n_clipped, _amp_label(lim)),
                 transform=axO.transAxes, ha="right", va="bottom", fontsize=5.6, color=CLIP)
    # ---- space-time heatmap: NOW PART OF THE SCROLLING STRIP --------------
    # It used to be a separate figure here (figH, "%s_heat.png").  Removed
    # 2026-08-14 at the reviewer's request: a static panel spanning the whole run
    # cannot be read against a montage window showing 10 s of it -- the reader had
    # to locate the same instant twice, in two different time scales.  The heatmap
    # and the recruitment lane are now axes of figW sharing axS's x-axis, so they
    # scroll with the montage and are in exact register with it by construction.
    # Their legends/colour scale moved to the pinned gutter (axGh/axGrc), because
    # anything drawn inside the wide image scrolls out of view.
    # The rationale for the FIXED log colour scale, the 0-floor and the
    # recruitment lane's 1.22 headroom is preserved at those axes.

    p_diag = os.path.join(outdir, "%s_diag.png" % stem)
    figD.savefig(p_diag, dpi=dpi)
    plt.close(figD)
    out["diag"] = p_diag

    meta = dict(meta, n_clipped_ch=int(n_clipped), T_s=T_s,
                px_per_s=PX_PER_S, window_s=WINDOW_S, lfp_step=step,
                heat_bin_s=hm_bin, heat_log_min=HEAT_LOG_MIN,
                heat_log_max=HEAT_LOG_MAX, heat_recruit_thr=HEAT_RECRUIT_THR,
                rate_max=S["RATE_MAX"], cl_max=S["CL_MAX"], clip_lim=lim)
    return out, meta
