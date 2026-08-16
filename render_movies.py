"""Sheet movies for the full-field runs, analogous to the Liou Fig 1 video.

TWO TIERS, because the stored data cannot support one movie that does both jobs.

  OVERVIEW (this module, from the stored 80 ms max envelope)
      The whole 80 s run at 80 ms per frame.  Shows recruitment and spread:
      the fast wave crosses the sheet in ~1.16 s = ~15 frames, so travelling
      structure is resolved.  Plays 2.0x real time (25 fps x 80 ms), so the
      80 s run takes 40 s to watch.

  DETAIL (needs the raw 5 ms field -- see render_movies_detail)
      The discharge rhythm on anchor_default_seq is 16.7 Hz (median IPI 60 ms).
      At 80 ms sampling that is 0.7 frames per cycle, i.e. BELOW Nyquist: the
      overview movie cannot show discharges at all, and any apparent flicker in
      it is an aliasing artefact rather than the rhythm.  Resolving 16.7 Hz
      needs <=30 ms sampling; 20 ms (every 4th raw frame) gives 3 samples per
      cycle.  Only anchor_default_seq retained its raw field, so the other 11
      require a re-run -- which is exact, since the pilot reproduces bit for bit.

Frame layout is deliberately close to the paper's: the 56x56 sheet as the whole
frame, firing rate on the SAME fixed 0-200 Hz magma scale the page's heatmap
uses, so a movie frame and a heatmap column are directly comparable.  A thin
lane under the sheet carries the recruited-fraction trace with a playhead, so
the movie is readable on its own when detached from the page.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import matplotlib as mpl

import imageio_ffmpeg

HZ_MAX = 200.0          # = f_max; identical to render_fullfield_page.HZ_MAX
THR = 20.0              # Hz, "recruited" for the fraction lane
UPSCALE = 8             # 56 -> 448 px, nearest-neighbour: units stay square
LANE_H = 54             # px, the recruited-fraction lane under the sheet
FPS = 25


def _lut():
    cm = mpl.colormaps["magma"]
    return (np.asarray([cm(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8)


def _digits():
    """5x7 bitmap digits, so the time readout needs no font dependency."""
    glyphs = {
        "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
        "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
        "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
        "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
        "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
        "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
        "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
        "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
        "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
        ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
        "s": ("00000", "00000", "01110", "10000", "01110", "00001", "11110"),
        " ": ("00000",) * 7,
    }
    return {k: np.array([[c == "1" for c in row] for row in v], bool)
            for k, v in glyphs.items()}


_GLY = _digits()


def _stamp(frame, text, x0, y0, scale=2, colour=(255, 255, 255)):
    """Draw text into an RGB array with the bitmap font (no PIL/font needed)."""
    for ch in text:
        g = _GLY.get(ch)
        if g is None:
            x0 += 6 * scale
            continue
        h, w = g.shape
        blk = np.repeat(np.repeat(g, scale, 0), scale, 1)
        yh, xw = blk.shape
        if y0 + yh <= frame.shape[0] and x0 + xw <= frame.shape[1]:
            frame[y0:y0 + yh, x0:x0 + xw][blk] = colour
        x0 += (w + 1) * scale
    return x0


def render_movie(hz, t_s, n, out_path, *, fps=FPS, title="", upscale=UPSCALE,
                 speed_note=""):
    """hz: (n_frames, n*n) firing rate in Hz -> an h264 .mp4 at out_path.

    Encoding goes through imageio_ffmpeg's bundled binary: there is no system
    ffmpeg in this sandbox, and matplotlib animation writers need one.  Frames
    are piped as raw rgb24 so nothing is written to disk between frames.
    """
    lut = _lut()
    nf = hz.shape[0]
    sheet_px = n * upscale
    W = sheet_px
    H = sheet_px + LANE_H
    # even dimensions required by yuv420p
    W += W % 2
    H += H % 2

    frac = (hz > THR).mean(axis=1)
    lane_y0 = sheet_px + 4
    lane_h = LANE_H - 8

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(fps), "-i", "-",
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-preset", "veryfast",
           "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # static part of the lane: the whole-run recruitment trace
    lane_bg = np.zeros((lane_h, W, 3), np.uint8)
    lane_bg[:] = (18, 18, 18)
    xs = np.clip((np.arange(nf) * W // max(1, nf)), 0, W - 1)
    ys = np.clip(lane_h - 1 - (frac * (lane_h - 1)).astype(int), 0, lane_h - 1)
    lane_bg[ys, xs] = (200, 200, 200)

    try:
        for k in range(nf):
            f = np.zeros((H, W, 3), np.uint8)
            idx = np.clip((hz[k] / HZ_MAX) * 255, 0, 255).astype(np.uint8)
            img = lut[idx].reshape(n, n, 3)
            big = np.repeat(np.repeat(img, upscale, 0), upscale, 1)
            f[:big.shape[0], :big.shape[1]] = big
            f[lane_y0:lane_y0 + lane_h, :W] = lane_bg
            px = int(k * W / max(1, nf))
            f[lane_y0:lane_y0 + lane_h, max(0, px - 1):px + 1] = (230, 80, 60)
            _stamp(f, "%.1f s" % t_s[k], 6, 6, scale=2)
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode()[-400:]
        if proc.wait() != 0:
            raise SystemExit("ffmpeg failed: %s" % err)
    finally:
        if proc.poll() is None:
            proc.kill()
    return out_path


def overview(npz_path, outdir="ff_movies"):
    """Whole-run movie from the stored 80 ms envelope. No re-run needed."""
    with np.load(npz_path, allow_pickle=True) as f:
        env = np.asarray(f["env"], np.float32)
        t_env = np.asarray(f["t_env"], np.float32)
        n = int(f["n"])
        fmax_hz = float(f["f_max"]) * 1e3
        key = json.loads(str(f["summary"]))["key"]
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "%s_overview.mp4" % key)
    render_movie(env * fmax_hz, t_env, n, p, fps=FPS, title=key)
    return p, dict(key=key, kind="overview", frames=int(env.shape[0]),
                   dt_ms=float((t_env[1] - t_env[0]) * 1e3), fps=FPS,
                   MB=round(os.path.getsize(p) / 1e6, 2))


def detail(npz_path, outdir="ff_movies", t0_s=18.0, dur_s=6.0, stride=4,
           fps=25):
    """Short window from the RAW 5 ms field, at 20 ms/frame.

    Why a window and not the whole run: 80 s of raw field at 20 ms is 4000
    frames and plays for 2.7 min.  Why 20 ms: the discharge rhythm on
    anchor_default_seq is 16.7 Hz (median IPI 60 ms), so 20 ms gives 3 samples
    per cycle -- above Nyquist, where the stored 80 ms envelope gives 0.7 and
    aliases the rhythm away entirely.  Playback is 0.5x real time (25 fps x
    20 ms = 0.5 s of model time per second watched), which is what makes the
    individual discharges and the inward wave watchable rather than a flicker.

    Requires rate_full in the npz; only anchor_default_seq retained it.  The
    others need a re-run with keep_full=True, which is exact -- the pilot
    reproduces bit for bit (verified on all 12).
    """
    with np.load(npz_path, allow_pickle=True) as f:
        if "rate_full" not in f.files:
            return None, dict(key=json.loads(str(f["summary"]))["key"],
                              kind="detail", available=False,
                              reason="no rate_full: re-run with keep_full=True")
        t = np.asarray(f["t"], np.float32)
        n = int(f["n"])
        fmax_hz = float(f["f_max"]) * 1e3
        key = json.loads(str(f["summary"]))["key"]
        dt_s = float(t[1] - t[0])
        i0 = int(t0_s / dt_s)
        i1 = min(int((t0_s + dur_s) / dt_s), f["rate_full"].shape[0])
        raw = np.asarray(f["rate_full"][i0:i1:stride], np.float32)
        tt = t[i0:i1:stride]
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "%s_detail.mp4" % key)
    render_movie(raw * fmax_hz, tt, n, p, fps=fps)
    return p, dict(key=key, kind="detail", available=True,
                   frames=int(raw.shape[0]), dt_ms=round(dt_s * stride * 1e3, 1),
                   window=[t0_s, t0_s + dur_s], fps=fps,
                   speed="%.2fx real time" % (fps * dt_s * stride),
                   MB=round(os.path.getsize(p) / 1e6, 2))


def main(indir="fullfield_out", outdir="ff_movies"):
    import glob
    metas = []
    for p in sorted(glob.glob(os.path.join(indir, "*.npz"))):
        _, m = overview(p, outdir)
        metas.append(m)
        pd_, md = detail(p, outdir)
        if md.get("available"):
            metas.append(md)
        print("%-24s %s" % (m["key"], m["MB"]), flush=True)
    json.dump(metas, open(os.path.join(outdir, "movies.json"), "w"), indent=1)
    return metas
