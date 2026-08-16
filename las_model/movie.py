"""Render network activity as a movie.

Works on any 2D mean-field result dict from ``las_model.run`` (``run_exp1``,
``run_exp1_torus``, ``run_exp6``) or on a raw ``(n_frames, n_rows, n_cols)``
array.  Two entry points:

* :func:`activity_movie` -- one panel, the firing-rate field.
* :func:`state_movie`    -- the firing rate alongside the slow state variables
  (chloride, potassium conductance, synaptic resource) that drive termination.

Both write MP4 via ffmpeg when available and fall back to animated GIF.

Design notes
------------
* **Colour scale is fixed across the whole movie** and stated in the colorbar.
  A per-frame autoscale would make a dying seizure look as intense as a fully
  recruited one -- the single most misleading thing a movie of this model can do.
* Time is annotated on every frame; a movie without a clock cannot be read
  against the paper's time-course figures.
* ``stride`` subsamples frames.  The 10 ms write cycle over 100 s gives ~10k
  frames; at 30 fps that is a 5.5-minute movie, so the default strides to a
  ~20 s movie and *reports the resulting playback speed-up on the frame*.
"""

from __future__ import annotations

import os
import numpy as np

__all__ = ["activity_movie", "state_movie", "ensure_writer"]


def ensure_writer(prefer: str = "ffmpeg") -> str:
    """Return an available matplotlib animation writer name.

    Registers the bundled ``imageio-ffmpeg`` binary if ffmpeg is not on PATH.
    Falls back to ``pillow`` (GIF) when no ffmpeg can be found.
    """
    import matplotlib as mpl
    import matplotlib.animation as anim

    if prefer == "ffmpeg" and not anim.writers.is_available("ffmpeg"):
        try:
            import imageio_ffmpeg

            mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    if prefer in anim.writers.list() and anim.writers.is_available(prefer):
        return prefer
    return "pillow"


def _unpack(result, key="rate"):
    """Accept a run-dict or a bare array; return (field, t_ms, f_max, mask)."""
    if isinstance(result, dict) or hasattr(result, "files"):
        keys = set(result.files) if hasattr(result, "files") else set(result.keys())
        get = lambda k: result[k] if k in keys else None
        if key not in keys:
            raise KeyError("%r not in result (have %s)" % (key, sorted(keys)))
        field = np.asarray(get(key))
        t = np.asarray(get("t")) if get("t") is not None else np.arange(len(field))
        fm = get("f_max")
        f_max = float(np.max(fm)) if fm is not None else None
        mask = get("mask")
        mask = None if mask is None or np.ndim(mask) == 0 else np.asarray(mask)
    else:
        field, t, f_max, mask = np.asarray(result), None, None, None
        t = np.arange(len(field), dtype=float)
    if field.ndim != 3:
        raise ValueError("expected (frames, rows, cols), got %s" % (field.shape,))
    return field, t, f_max, mask


def _apply_mask(a, mask):
    if mask is None:
        return a
    out = a.astype(float).copy()
    out[~mask.astype(bool)] = np.nan
    return out


def _fmt_title(t_ms, speed):
    s = "t = %6.2f s" % (t_ms / 1000.0)
    return s if speed is None else "%s   (%.0f$\\times$ real time)" % (s, speed)


def activity_movie(
    result,
    path: str = "activity.mp4",
    *,
    stride: int = 20,
    fps: int = 30,
    cmap: str = "magma",
    vmax=None,
    normalise: bool = True,
    title: str = "",
    dpi: int = 130,
    trace: bool = True,
):
    """Movie of the firing-rate field, optionally with a mean-activity trace.

    Parameters
    ----------
    result : dict | npz | ndarray
        A run result, or a ``(frames, rows, cols)`` array.
    stride : int
        Keep every ``stride``-th recorded frame.
    normalise : bool
        Divide by ``f_max`` so the colour scale is a firing-rate fraction.
    vmax : float, optional
        Upper colour limit.  Default: the true maximum over the whole movie, so
        nothing ever clips -- a movie whose ictal core saturates misrepresents
        the seizure's intensity.  Pass an explicit percentile-derived value only
        if a genuine outlier pixel is flattening the rest.
    trace : bool
        Add a lower panel showing mean activity with a moving time cursor, so a
        viewer can place the current frame within the seizure.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim

    field, t, f_max, mask = _unpack(result)
    if normalise and f_max:
        field = field / f_max
    field = field[::stride]
    t = t[::stride]

    dt_rec = float(np.median(np.diff(t))) if len(t) > 1 else 1.0   # ms per kept frame
    speed = (dt_rec * fps) / 1000.0 if dt_rec > 0 else None        # x real time

    if vmax is None:
        vmax = max(float(field.max()), 1e-6)
    clipped = float((field > vmax).mean())

    mk = None if mask is None else mask.astype(bool)
    active = np.ones(field.shape[1:], bool) if mk is None else mk
    mean_act = field[:, active].mean(axis=1)

    if trace:
        fig, (ax, axt) = plt.subplots(
            2, 1, figsize=(4.2, 4.7), height_ratios=[3.1, 1.0],
            gridspec_kw=dict(hspace=0.32, left=0.17, right=0.85, top=0.90, bottom=0.13))
    else:
        fig, ax = plt.subplots(figsize=(4.0, 3.9))
        axt = None

    im = ax.imshow(_apply_mask(field[0], mk), vmin=0, vmax=vmax, cmap=cmap,
                   origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    ttl = ax.set_title(_fmt_title(t[0], speed), fontsize=9, loc="left", family="monospace")
    if title:
        fig.suptitle(title, fontsize=9, x=0.06, ha="left")

    cax = fig.add_axes([0.865, ax.get_position().y0, 0.025, ax.get_position().height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("firing rate / $f_{max}$" if (normalise and f_max) else "firing rate",
                 fontsize=7)
    cb.ax.tick_params(labelsize=6)

    cursor = None
    if axt is not None:
        axt.plot(t / 1000.0, mean_act, color="#333333", lw=0.8)
        axt.set_xlabel("time (s)", fontsize=8)
        axt.set_ylabel("mean\nactivity", fontsize=8)
        axt.tick_params(labelsize=6)
        axt.set_xlim(t[0] / 1000.0, t[-1] / 1000.0)
        axt.margins(y=0.08)
        cursor = axt.axvline(t[0] / 1000.0, color="#c1553b", lw=1.0)

    def update(i):
        im.set_data(_apply_mask(field[i], mk))
        ttl.set_text(_fmt_title(t[i], speed))
        if cursor is not None:
            cursor.set_xdata([t[i] / 1000.0, t[i] / 1000.0])
        return (im, ttl) if cursor is None else (im, ttl, cursor)

    a = anim.FuncAnimation(fig, update, frames=len(field), blit=False)
    writer = ensure_writer("ffmpeg")
    if writer == "pillow" and path.endswith(".mp4"):
        path = path[:-4] + ".gif"
    a.save(path, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return dict(path=path, n_frames=len(field), fps=fps, speed_x=speed,
                vmax=float(vmax), clipped_fraction=clipped, writer=writer,
                size_mb=os.path.getsize(path) / 1e6)


def state_movie(
    result,
    path: str = "state.mp4",
    *,
    stride: int = 20,
    fps: int = 30,
    keys=("rate", "Cl_in", "g_K", "phi"),
    labels=None,
    cmaps=("magma", "cividis", "viridis", "bone"),
    dpi: int = 130,
):
    """Multi-panel movie: firing rate beside the slow variables driving it.

    The seizure's *termination* is invisible in a rate movie alone -- it is
    driven by chloride accumulation, potassium conductance, and synaptic
    resource depletion.  This renders them side by side on a shared clock.
    Panels whose key is absent from ``result`` are skipped.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim

    default_labels = {
        "rate": "firing rate / $f_{max}$",
        "Cl_in": "[Cl$^-$]$_{in}$ (mM)",
        "g_K": "$g_K$ (nS)",
        # phi is the ADAPTIVE FIRING THRESHOLD (mV), entering as activation(V - phi):
        # baseline phi0 = -45 mV, relaxing toward phi0 + 60*(R_f/f_max) with tau 100 ms.
        # It is not a synaptic resource; rising phi is spike-frequency adaptation.
        "phi": "threshold $\\phi$ (mV)",
        "V": "$V$ (mV)",
    }
    labels = labels or default_labels

    src = result
    have = [k for k in keys if (k in src.files if hasattr(src, "files") else k in src)]
    if not have:
        raise ValueError("none of %r present in result" % (keys,))

    field0, t, f_max, mask = _unpack(src, have[0])
    t = t[::stride]
    dt_rec = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    speed = (dt_rec * fps) / 1000.0

    fields, vlims = [], []
    for k in have:
        f = np.asarray(src[k])[::stride].astype(np.float32)
        if k == "rate" and f_max:
            f = f / f_max
        fields.append(f)
        lo = float(np.percentile(f, 0.1)); hi = float(np.percentile(f, 99.9))
        if k == "rate":
            lo = 0.0
        vlims.append((lo, hi if hi > lo else lo + 1e-6))

    mk = None if mask is None else mask.astype(bool)
    n = len(have)
    fig, axes = plt.subplots(1, n, figsize=(2.35 * n, 2.95))
    axes = np.atleast_1d(axes)
    ims = []
    for ax, k, f, (lo, hi), cm in zip(axes, have, fields, vlims, cmaps):
        im = ax.imshow(_apply_mask(f[0], mk), vmin=lo, vmax=hi, cmap=cm,
                       origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5)
        ax.set_title(labels.get(k, k), fontsize=7.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5.5)
        ims.append(im)
    sup = fig.suptitle(_fmt_title(t[0], speed), fontsize=9, family="monospace")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(i):
        for im, f in zip(ims, fields):
            im.set_data(_apply_mask(f[i], mk))
        sup.set_text(_fmt_title(t[i], speed))
        return ims + [sup]

    a = anim.FuncAnimation(fig, update, frames=len(fields[0]), blit=False)
    writer = ensure_writer("ffmpeg")
    if writer == "pillow" and path.endswith(".mp4"):
        path = path[:-4] + ".gif"
    a.save(path, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return dict(path=path, panels=have, n_frames=len(fields[0]), fps=fps,
                speed_x=speed, writer=writer, size_mb=os.path.getsize(path) / 1e6)
