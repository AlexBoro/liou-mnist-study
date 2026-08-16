"""Scrolling page for the 12 full-field runs: LFP montage + ALL 3136 units.

Same reading conventions as the adjudication page (fixed scales across runs,
10 s per screen, pinned gutters) but the space-time heatmap shows every unit on
the 56x56 sheet instead of the 30 electrodes.

WHY TWO IMAGES IN ONE SCROLL BOX, not one image as in the adjudication strip.
The adjudication heat lane has 30 rows and fits inside the LFP strip figure.
This one has 3136 -- at one pixel per unit the heat lane alone is 3136 px tall,
so merging it into the strip would push the LFP off-screen whenever the reader
looked at the bottom of the sheet.  Instead:

  * both images are rendered at the SAME width and the same PX_PER_S, and both
    live in the same horizontally-scrolling div, so there is one scrollLeft and
    the time axes cannot drift apart -- the same guarantee the merged strip
    gives, obtained structurally rather than by JS sync;
  * the LFP strip is ``position: sticky`` at the top of that box, so it stays
    visible while the sheet is panned vertically.

WHY THE HEATMAP IS BUILT WITH NUMPY + PIL RATHER THAN MATPLOTLIB.  A
9440 x 3136 px matplotlib figure rasterises slowly and holds ~120 MB of RGBA
per run.  The heatmap is a pure raster -- one pixel row per unit, colour from a
fixed colormap -- so it is composed directly as an array and upscaled
horizontally with nearest-neighbour.  All axis decoration lives in the pinned
gutter, which is a small matplotlib figure.

FULL RESOLUTION IS LITERAL: every one of the n*n units gets its own pixel row,
in row-major sheet order (rows 0..n-1 of sheet row 0, then sheet row 1, ...),
so a spatially contiguous 2-D blob reads as a block of adjacent lanes.  No
spatial pooling is applied anywhere.  Time is the stored 80 ms MAX envelope,
the same bin as the adjudication heat lane.

Colour scale is FIXED across all 12 runs (0 to 80 Hz, the paper's Figure 6
ceiling) so brightness is comparable run to run; units above the ceiling are
drawn in a distinct colour rather than flattened into the top of the ramp.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import las_model.readout as RO
import render_adjudication_figs as RAF

PX_PER_S = RAF.PX_PER_S       # identical to the adjudication page: same reading
WINDOW_S = RAF.WINDOW_S       # rate, so a reader moves between them unchanged
# COLOUR SCALE: 0 to f_max (200 Hz), the full physical range, so NOTHING is
# clipped and no run is drawn as a flat field of over-colour.  An earlier
# version used the paper's Figure 6 ceiling of 80 Hz.  Measured across all 12
# full-field runs, the fraction of (unit, bin) samples above 80 Hz is:
#   anchor_gE220_seq 98%, anchor_gE220_sim 93%, draw01_seq 92%,
#   draw16_seq 42%, draw16_sim 38%, anchor_default_seq 14%, _sim 14%,
#   and <2% for the five focal/entrained runs.
# Seven of the twelve therefore lose more than a tenth of their samples to the
# ceiling, and the three block runs lose essentially all of theirs -- those maps
# carried no information at all.  The paper's 80 Hz level is marked
# on the gutter ramp instead, which keeps the comparison available without
# destroying the runs that exceed it.
FIT_ROWS = 800                # target row count for the whole-sheet overview:
                              # ~870 px of pane on a 1080x1920 portrait display,
                              # so the pooled image upscales slightly rather than
                              # being resampled down (which is what loses events)
HZ_MAX = 200.0                # = f_max; the full representable range
HZ_PAPER = 80.0               # paper Fig 6 ceiling, marked on the ramp
GREEN = RAF.GREEN
GREY = RAF.GREY
CLIP = RAF.CLIP
IMAGE_PERIOD_S = RAF.IMAGE_PERIOD_S
SCALES = dict(RAF.SCALES)     # LFP lane spacing / clip / rate / [Cl-] ranges


def _match_height(path, target_h):
    """Force a gutter image to its partner's exact pixel height.

    The page scales each gutter by its partner image's scale factor, so a
    mismatch of even a few pixels shifts every pinned label off the lane it
    names.  matplotlib rounds figsize*dpi, so the height cannot be guaranteed at
    figure-creation time; resampling a text-only gutter by <1% is invisible,
    whereas mislabelled lanes are not.
    """
    im = Image.open(path)
    if im.size[1] != target_h:
        im.resize((im.size[0], int(target_h)), Image.LANCZOS).save(path)
    return target_h


def _sheet_raster(env_hz, width_px):
    """(n_units, n_bins) Hz -> (n_units, width_px, 3) uint8, magma + over-colour.

    Nearest-neighbour along time only.  No pooling along units: the array has
    exactly one row per unit, which is what makes this full resolution.
    """
    cmap = mpl.colormaps["magma"]
    lut = (np.asarray([cmap(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8)
    q = np.clip(env_hz / HZ_MAX, 0.0, 1.0)
    idx = (q * 255).astype(np.uint8)
    rgb = lut[idx]                                   # (units, bins, 3)
    # Horizontal upscale onto the strip's time axis.  Bin j covers
    # [j*bin, (j+1)*bin), and the strip's axis is 0..T_s across the same
    # width_px, so mapping pixel x to bin floor(x * nb / width_px) puts every
    # bin over exactly the seconds it represents -- the +0.5 bin-centre
    # convention would shift the sheet half a bin (40 ms, ~4.7 px) later than
    # the LFP, which is visible as the two panes disagreeing about where a
    # discharge starts.
    src_w = rgb.shape[1]
    take = np.clip((np.arange(width_px) * src_w) // width_px, 0, src_w - 1)
    return rgb[:, take, :]


def _time_rules(rgb, T_s, width_px):
    """Brighten one pixel column per second so the sheet carries the time grid.

    x is computed from T_s -- the SAME span the strip's matplotlib axis uses --
    so a second mark lands on the identical pixel column in both images and the
    two panes can be checked against each other by eye.
    """
    for s in range(1, int(T_s) + 1):
        x = int(round(s * width_px / T_s))
        if 0 <= x < width_px:
            col = rgb[:, x, :].astype(np.int16)
            rgb[:, x, :] = np.clip(col + 46, 0, 255).astype(np.uint8)
    return rgb


def render_run(npz_path, outdir, dpi=110):
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["summary"]))
    key = meta["key"]
    n = int(z["n"])
    fmax_hz = float(z["f_max"]) * 1e3      # f_max is per MILLISECOND: 0.2 -> 200 Hz

    L = np.asarray(z["L"], dtype=float)[1:]        # drop the t=0 LFP-filter NaN
    t = np.asarray(z["t"], dtype=float)[1:]
    if t.max() > 1000.0:
        t = t / 1000.0
    driven = np.asarray(z["driven"], dtype=bool)
    rate = np.asarray(z["rate_trace"], dtype=float)[1:]
    Cl = np.asarray(z["Cl_in"], dtype=float)[1:]
    env = np.asarray(z["env"], dtype=np.float32)   # (bins, units), fraction f_max
    t_env = np.asarray(z["t_env"], dtype=np.float32)
    thr = float(meta["field_thr"])

    sheet_n = RAF.infer_sheet_n(driven)
    if sheet_n != n:
        raise ValueError("sheet size disagreement: npz n=%d, driven mask implies %d"
                         % (n, sheet_n))
    order, rows, cols = RAF._order_channels(sheet_n)
    T_s = float(t[-1])
    width_px = int(round(T_s * PX_PER_S))
    os.makedirs(outdir, exist_ok=True)
    out = {}

    # ------------------------------------------------ 1. LFP strip (sticky)
    step = float(SCALES["LANE"])
    lim = step / 2.0 * float(SCALES["CLIP_MULT"])
    # h_lfp raised from 6.6 to 8.4: at lw=1.0 the 30 lanes need more vertical
    # room or adjacent traces visually merge where both are active.
    h_lfp, h_r, h_c = 8.4, 0.85, 0.85
    h_tot = h_lfp + h_r + h_c + 0.42
    w_in = max(width_px / dpi, 6.0)
    figW = plt.figure(figsize=(w_in, h_tot))
    gsW = figW.add_gridspec(3, 1, height_ratios=[h_lfp, h_r, h_c], hspace=0.10,
                            left=0.0, right=1.0, top=0.995, bottom=0.075)
    axS = figW.add_subplot(gsW[0])
    axR = figW.add_subplot(gsW[1], sharex=axS)
    axC = figW.add_subplot(gsW[2], sharex=axS)

    n_clipped = 0
    for k, ch in enumerate(order):
        y0 = L[:, ch] - np.nanmean(L[:, ch])
        base = (len(order) - 1 - k) * step
        ins, outs = RAF._clip_split(y0, lim)
        # LINE WIDTH: 1.0 pt, not the adjudication strip's 0.42 pt.  At dpi=110
        # a 0.42 pt stroke is 0.64 PIXELS wide, so antialiasing spreads it over
        # two pixel columns at partial opacity and the trace reads as faint grey
        # rather than a line -- the reviewer could not see them.  1.0 pt = 1.53 px
        # renders as a solid stroke.  The adjudication page got away with 0.42
        # because it is viewed at its own scale; here the sheet below sets the
        # visual weight of the page and a hairline LFP disappears beside it.
        axS.plot(t, base + ins, lw=1.0, color=GREEN if driven[ch] else GREY, zorder=2)
        if np.isfinite(outs).any():
            n_clipped += 1
            axS.plot(t, base + outs, lw=1.3, color=CLIP, zorder=3)
    axS.set_ylim(-step, len(order) * step)
    axS.set_xlim(0.0, T_s)
    axS.set_yticks([])
    axS.tick_params(length=0, labelbottom=False)
    for s_ in ("top", "right", "left"):
        axS.spines[s_].set_visible(False)
    for x in np.arange(0.0, T_s, IMAGE_PERIOD_S):
        axS.axvline(x, color="#ececec", lw=0.3, zorder=0)
    for x in np.arange(0.0, T_s + 1e-9, 1.0):
        axS.axvline(x, color="#cfcfcf", lw=0.45, zorder=1)

    for a_, y_, cl_, hi_ in ((axR, rate, "#333333", SCALES["RATE_MAX"]),
                             (axC, Cl, "#1f6feb", SCALES["CL_MAX"])):
        a_.plot(t, y_, lw=1.0, color=cl_)   # same sub-pixel reasoning as axS
        a_.set_ylim(SCALES["CL_MIN"] if a_ is axC else 0.0, hi_)
        a_.set_xlim(0.0, T_s)
        a_.set_yticks([])
        for s_ in ("top", "right", "left"):
            a_.spines[s_].set_visible(False)
        for x in np.arange(0.0, T_s + 1e-9, 1.0):
            a_.axvline(x, color="#efefef", lw=0.35, zorder=0)
    axR.tick_params(labelbottom=False, length=0)
    axC.set_xlabel("time (s)", fontsize=6)
    axC.tick_params(labelsize=5.5, length=2)

    p_strip = os.path.join(outdir, "%s_strip.png" % key)
    figW.savefig(p_strip, dpi=dpi)
    plt.close(figW)
    out["strip"] = p_strip

    # ------------------------------- 2. LFP gutter (must match the strip stack)
    figG = plt.figure(figsize=(1.95, h_tot))
    gsG = figG.add_gridspec(3, 1, height_ratios=[h_lfp, h_r, h_c], hspace=0.10,
                            left=0.02, right=0.995, top=0.995, bottom=0.075)
    axG, axGr, axGc = (figG.add_subplot(gsG[i]) for i in range(3))
    axG.set_xlim(0, 1)
    axG.set_ylim(-step, len(order) * step)
    axG.axis("off")
    for k, ch in enumerate(order):
        base = (len(order) - 1 - k) * step
        axG.text(0.97, base, "%d,%d" % (rows[ch], cols[ch]), fontsize=4.4,
                 ha="right", va="center", color=GREEN if driven[ch] else "#666666")
    axG.text(0.97, len(order) * step, "LFP montage, 30 electrodes", fontsize=5.4,
             ha="right", va="top", color="#555555")
    axG.text(0.97, -step * 0.55, "gain %s/lane" % RAF._amp_label(step),
             fontsize=5.0, ha="right", va="bottom", color="#555555")
    for a_, lab, hi_, cl_ in ((axGr, "mean rate", SCALES["RATE_MAX"], "#333333"),
                              (axGc, "[Cl$^-$]$_{in}$", SCALES["CL_MAX"], "#1f6feb")):
        a_.set_xlim(0, 1)
        a_.set_ylim(0, hi_)
        a_.axis("off")
        a_.text(0.97, hi_ * 0.96, lab, fontsize=5.2, ha="right", va="top", color=cl_)
        a_.text(0.97, hi_ * 0.06, "0 - %g" % hi_, fontsize=4.8, ha="right",
                va="bottom", color="#777777")
    p_gut = os.path.join(outdir, "%s_gutter.png" % key)
    figG.savefig(p_gut, dpi=dpi)
    plt.close(figG)
    _match_height(p_gut, Image.open(p_strip).size[1])
    out["gutter"] = p_gut

    # --------------------------------- 3. full-resolution sheet (3136 rows)
    # The raster is built at the strip's ACTUAL rendered width, not the nominal
    # one: matplotlib rounds figsize*dpi and came out 24 px wider than
    # round(T_s * PX_PER_S) (9463 vs 9439 on anchor_default_seq).  Taking the measured width means the two images in
    # the scroll box are the same number of pixels wide by construction, so a
    # given x is the same instant in both -- the whole point of the shared box.
    strip_w, strip_h = Image.open(p_strip).size
    hz = env.T * fmax_hz                       # (units, bins)
    rgb = _sheet_raster(hz, strip_w)
    rgb = _time_rules(rgb, T_s, strip_w)
    Image.fromarray(rgb).save(os.path.join(outdir, "%s_sheet.png" % key))
    out["sheet"] = os.path.join(outdir, "%s_sheet.png" % key)
    sheet_h = rgb.shape[0]

    # ---- overview raster: the WHOLE sheet at a height a screen can show -----
    # The full sheet is one pixel row per unit (3136 px).  On the reviewer's
    # 1080x1920 portrait display a 50/50 split gives each pane ~870 px, so
    # showing all 3136 rows at once needs a ~3.6x vertical reduction.
    #
    # This is rendered as a MAX-POOLED image rather than left to the browser,
    # because a browser downscale is nearest-neighbour: it SAMPLES ~870 of the
    # 3136 rows and discards the rest.  Measured over all 12 runs, that drops
    # 21 time bins that contain activity at full resolution to completely blank
    # -- and the losses are concentrated in the SPARSE runs (9 bins in
    # anchor_gKmax120_seq, 7 in anchor_gIlocal600_seq, 4 in draw04_seq), i.e.
    # exactly the runs where one missed discharge changes the verdict.  A max
    # pool over the same rows loses ZERO bins: a lit unit always lights its
    # pooled row.  The trade is that the overview overstates spatial extent by
    # up to POOL units, which is why it is an OVERVIEW and `h` still gives the
    # true one-row-per-unit image.
    pool = int(np.ceil(n * n / float(FIT_ROWS)))
    nrow = (hz.shape[0] // pool) * pool
    hz_pool = hz[:nrow].reshape(-1, pool, hz.shape[1]).max(axis=1)
    rgb_fit = _sheet_raster(hz_pool, strip_w)
    rgb_fit = _time_rules(rgb_fit, T_s, strip_w)
    Image.fromarray(rgb_fit).save(os.path.join(outdir, "%s_sheetfit.png" % key))
    out["sheetfit"] = os.path.join(outdir, "%s_sheetfit.png" % key)

    # ------------------------- 4. sheet gutters ------------------------------
    # Built twice, once per sheet image: the full one is one pixel row per unit
    # (sheet_h px) and the overview is pooled (rgb_fit height).  Both are drawn
    # in the SAME unit coordinates -- y runs 0..n*n regardless of pixel height --
    # so "row 24" marks the same tissue in both, and each gutter is exactly as
    # tall as the image it labels.  The page picks the matching pair.
    for _tag, _hpx in (("sheetgut", sheet_h), ("sheetfitgut", rgb_fit.shape[0])):
      figSG = plt.figure(figsize=(1.95, _hpx / dpi))
      axSG = figSG.add_axes([0.0, 0.0, 1.0, 1.0])
      axSG.set_xlim(0, 1)
      axSG.set_ylim(n * n, 0)
      axSG.axis("off")
      # sheet-row ticks: units are in row-major order, so every n-th lane starts a
      # new sheet row.  Labelling all n would be unreadable; label every 8th.
      for r in range(0, n, 8):
          y = r * n
          axSG.plot([0.90, 1.0], [y, y], lw=0.5, color="#bbbbbb", clip_on=False)
          axSG.text(0.88, y, "row %d" % r, fontsize=4.6, ha="right", va="center",
                    color="#777777")
      # driven band: which units are inside the stimulated patch
      band = int(round(n * (56 / 84.))) if n != 84 else 56
      lo = (n - band) // 2
      axSG.add_patch(plt.Rectangle((0.955, lo * n), 0.030, band * n, color=GREEN,
                                   lw=0, alpha=0.55, zorder=5, clip_on=False))
      axSG.text(0.88, (lo + band / 2.0) * n, "driven\nrows", fontsize=4.8,
                ha="right", va="center", color=GREEN)
      # Positions are fractions of the UNIT axis (0..n*n), not of a pixel
      # height: this gutter is rendered at two different pixel heights and the
      # y-limit is n*n in both, so a sheet_h-scaled offset would put the labels
      # off-image on the pooled one.
      axSG.text(0.97, n * n * 0.004, "all %d units, row-major" % (n * n),
                fontsize=5.4, ha="right", va="top", color="#555555")
      axSG.text(0.97, n * n * 0.030, "%.0f ms bins, max envelope" % (
          1e3 * float(t_env[1] - t_env[0])), fontsize=4.8, ha="right", va="top",
          color="#777777")
      # colour ramp, drawn in gutter coordinates (an inset colorbar would scroll
      # away if it lived in the sheet image)
      y0, hgt = n * n * 0.055, n * n * 0.16
      for i in range(64):
          c = mpl.colormaps["magma"](i / 63.0)
          axSG.add_patch(plt.Rectangle((0.62, y0 + hgt * i / 64.0), 0.10, hgt / 64.0,
                                       color=c, lw=0, clip_on=False))
      axSG.text(0.60, y0, "0 Hz", fontsize=4.6, ha="right", va="center", color="#777777")
      axSG.text(0.60, y0 + hgt, "%d Hz = f$_{max}$" % HZ_MAX, fontsize=4.6,
                ha="right", va="center", color="#777777")
      # mark the paper's Figure 6 ceiling on the ramp so a pilot snapshot can
      # still be placed against a paper panel without rescaling the whole map
      yp = y0 + hgt * (HZ_PAPER / HZ_MAX)
      axSG.plot([0.615, 0.735], [yp, yp], lw=0.7, color="#39d0d8", clip_on=False)
      axSG.text(0.745, yp, "%d Hz\n(paper Fig 6)" % HZ_PAPER, fontsize=4.2,
                ha="left", va="center", color="#2aa8b0")
      p_sg = os.path.join(outdir, "%s_%s.png" % (key, _tag))
      figSG.savefig(p_sg, dpi=dpi)
      plt.close(figSG)
      _match_height(p_sg, _hpx)
      out[_tag] = p_sg

    m = dict(key=key, n=n, units=int(n * n), width_px=width_px,
             sheet_h=int(sheet_h), T_s=T_s, n_clipped=int(n_clipped),
             peak_frac=float((env > thr).mean(axis=1).max()),
             duty_median=float(meta["unit_duty_median"]),
             mean_hz=float(np.mean(rate) * fmax_hz))
    return out, m


def main(indir="fullfield_out", outdir="ff_figs"):
    import glob
    metas = []
    for p in sorted(glob.glob(os.path.join(indir, "*.npz"))):
        o, m = render_run(p, outdir)
        mb = sum(os.path.getsize(v) for v in o.values()) / 1e6
        m["MB"] = round(mb, 1)
        metas.append(m)
        print("%-24s %5d x %4d px  %5.1f MB" % (m["key"], m["width_px"],
                                                m["sheet_h"], mb), flush=True)
    return metas


if __name__ == "__main__":
    main()
