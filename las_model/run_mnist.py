"""10 s MNIST-driven runs on the Liou sheet, WC-analogous protocol.

Mirrors ``param_explore.py``'s exploration loop: a stream of randomly chosen
digits, one every ``IMAGE_DURATION_MS``, delivered continuously for the whole
run with no blank intervals and no training.  State is NOT reset between
images -- the WC study does the same (each digit lands on whatever the last one
left behind), so this is a deliberate match rather than an oversight.

Records the slow states (``Cl_in``, ``g_K``) alongside rate, because
``tau_Cl = tau_K = 5000 ms`` means a 10 s run is only two time constants and
whether the sheet has equilibrated is an empirical question, not an assumption.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .model import MeanFieldModel, ExternalInput
from . import params as P
from .mnist_drive import PatchDrive, MNISTDrive, IMAGE_DURATION_MS, MNIST_INPUT_SCALE

__all__ = ["run_mnist", "image_stream"]


def image_stream(images, rng, n_images: int) -> np.ndarray:
    """Indices of ``n_images`` digits drawn uniformly with replacement."""
    return rng.integers(0, len(images), size=n_images)


def run_mnist_patch(*args, **kw):
    """3x3 patch projection variant; see run_mnist for the identity-map version."""
    kw['projection'] = True
    return run_mnist(*args, **kw)


def run_mnist(images, labels=None, *, scale: float = MNIST_INPUT_SCALE,
              sequential: bool = False, T_end: float = 80000.0,
              dt: float = 1.0, n: int = 56, seed: int = 0,
              rec_every: int = 10, weights: Optional[np.ndarray] = None,
              params=None, noise_D: float = 0.0, cl_floor=0.1, projection=False, n_grid=None, compensate: object = "sqrt", rec=None, image_duration=None, probe=None, record_lfp_currents: bool = False, field_every: int = 0,
              progress: bool = False) -> dict:
    """Drive the sheet with a stream of MNIST digits for ``T_end`` ms.

    Parameters mirror the WC exploration protocol.  ``sequential=False`` is
    WC's simultaneous configuration (all 784 pixels live); ``True`` scans rows
    on the continuous-time schedule.

    Returns a dict of recorded arrays.  ``rate`` is (T, n*n) normalised to
    ``f_max``; ``Cl_in`` and ``g_K`` are field means of the slow states.
    """
    rng = np.random.default_rng(seed)
    pp = P.exp1_template((n, n)) if params is None else params
    model = MeanFieldModel(pp, seed=seed)
    rkw = dict(topology="circular")
    if rec:
        rkw.update(rec)
    model.add_standard_recurrent_connections(**rkw)
    model.cl_floor = cl_floor
    DriveCls = PatchDrive if projection else MNISTDrive
    dkw = dict(compensate=compensate) if projection else {}
    drive = DriveCls(pp, rng=rng, scale=scale, n=n, sequential=sequential,
                     weights=weights, **dkw)
    # One ExternalInput carries BOTH the digit and the OU/white noise, as
    # Exp1.m/Exp6.m do.  noise_D is a diffusion coefficient (pA^2/ms), not an
    # amplitude -- sigma_from_diffusion does the dt-dependent conversion, and
    # passing D straight through as sigma would inject 100x the intended noise.
    from .run import sigma_from_diffusion
    sig = sigma_from_diffusion(noise_D, dt) if noise_D else 0.0
    if probe is None:
        _drv = lambda x1, x2, t: drive._field
    else:
        # Focal test pulse of Exp1.m geometry ADDED to the MNIST field: a disc
        # of radius probe["r"] at normalised centre probe["x"], amplitude
        # probe["Ic"] pA, live during probe["t"] (seconds).  Used to measure
        # seizure susceptibility of a network whose operating point is set by
        # the MNIST drive -- the drive itself is spatially distributed and
        # engages global inhibition everywhere, so it cannot nucleate the
        # focal Cl-loading runaway that Liou's seizures require.
        _pr = dict(Ic=200.0, r=0.05, x=(0.5, 0.5), t=(20.0, 23.0))
        _pr.update(probe)
        _yy, _xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        _d1 = _yy / n - _pr["x"][0]
        _d2 = _xx / n - _pr["x"][1]
        _d1 -= np.round(_d1); _d2 -= np.round(_d2)
        _disc = (np.sqrt(_d1 ** 2 + _d2 ** 2) < _pr["r"]) * _pr["Ic"]
        _t0, _t1 = _pr["t"][0] * 1000.0, _pr["t"][1] * 1000.0
        def _drv(x1, x2, t, _disc=_disc, _t0=_t0, _t1=_t1):
            return drive._field + _disc if _t0 < t < _t1 else drive._field
    model.ext = [ExternalInput(deterministic=_drv, sigma=sig, tmax=None)]

    n_steps = int(round(T_end / dt))
    idur = IMAGE_DURATION_MS if image_duration is None else float(image_duration)
    steps_per_image = int(round(idur / dt))
    n_images = int(np.ceil(n_steps / steps_per_image))
    idx = image_stream(images, rng, n_images)

    f_max = float(np.mean(pp.f_max))
    rec = list(range(0, n_steps, rec_every))
    T = len(rec)
    out = dict(
        t=np.empty(T, dtype=np.float32),
        rate=np.empty((T, n * n), dtype=np.float32),
        Cl_in=np.empty(T, dtype=np.float32),
        g_K=np.empty(T, dtype=np.float32),
        V=np.empty(T, dtype=np.float32),
        img_idx=np.empty(T, dtype=np.int32),
        row=np.empty(T, dtype=np.int16),
    )
    if record_lfp_currents:
        out["EPSC"] = np.empty((T, n * n), dtype=np.float32)
        out["IPSC"] = np.empty((T, n * n), dtype=np.float32)

    # Optional PER-UNIT slow states.  Cl_in/g_K above are field MEANS, which
    # cannot show a focal chloride load -- the thing that makes a Liou seizure
    # focal.  Recorded on their own coarse stride because the slow states are
    # slow: tau_Cl = tau_K = 5000 ms, so at the pilot's field_every=200 with the
    # default dt=1.0 ms the sampling period is 200 ms (5 Hz) -- still 25x
    # oversampled relative to the time constant, for 1/40 the memory of the rate
    # field (which is recorded every 5 steps).
    #
    # UNITS: t_field is in MILLISECONDS (step * dt, and dt is in ms), unlike the
    # `t_env` axis in the pilot npz which is in seconds.  A reader who assumes
    # seconds here reads the 200 ms period as 200 s.
    field_rec = []
    if field_every:
        field_rec = list(range(0, n_steps, int(field_every)))
        TF = len(field_rec)
        out["Cl_in_field"] = np.empty((TF, n * n), dtype=np.float32)
        out["g_K_field"] = np.empty((TF, n * n), dtype=np.float32)
        out["t_field"] = np.empty(TF, dtype=np.float32)
    fptr = 0

    cur_img = -1
    ptr = 0
    for step in range(n_steps):
        i = min(step // steps_per_image, n_images - 1)
        if i != cur_img:
            drive.set_image(images[idx[i]])
            cur_img = i
        drive.set_step(step, dt)

        if ptr < T and step == rec[ptr]:
            out["t"][ptr] = step * dt
            out["rate"][ptr] = np.asarray(model.R_f).ravel() / f_max
            out["Cl_in"][ptr] = float(np.mean(model.Cl_in))
            out["g_K"][ptr] = float(np.mean(model.g_K))
            out["V"][ptr] = float(np.mean(model.V))
            out["img_idx"][ptr] = idx[i]
            out["row"][ptr] = -1 if not sequential else (drive._row if drive._row is not None else -1)
            if record_lfp_currents:
                e, s = model.psc()
                out["EPSC"][ptr] = np.asarray(e).ravel()
                out["IPSC"][ptr] = np.asarray(s).ravel()
            ptr += 1

        if field_rec and fptr < len(field_rec) and step == field_rec[fptr]:
            out["Cl_in_field"][fptr] = np.asarray(model.Cl_in).ravel()
            out["g_K_field"][fptr] = np.asarray(model.g_K).ravel()
            out["t_field"][fptr] = step * dt
            fptr += 1

        model.step(dt)
        if progress and step % 2000 == 0:
            print("  step %d/%d" % (step, n_steps), flush=True)

    _fieldk = ("Cl_in_field", "g_K_field", "t_field")
    out = {k: (v[:ptr] if (k not in _fieldk and getattr(v, "ndim", 0) and len(v) == T) else v)
           for k, v in out.items()}
    if field_rec:
        for k in ("Cl_in_field", "g_K_field", "t_field"):
            out[k] = out[k][:fptr]
    out["meta"] = dict(scale=scale, sequential=sequential, dt=dt, n=n, noise_D=noise_D,
                       seed=seed, T_end=T_end, f_max=f_max, field_every=int(field_every),
                       I_threshold=drive.I_threshold,
                       steps_per_image=steps_per_image, n_images=n_images)
    out["weights"] = drive.weights
    out["labels"] = None if labels is None else labels[idx]
    # Count of Cl_in clamp activations (see MeanFieldModel.step).  Nonzero means
    # this run departed from Liou's equations; a parameter search should treat a
    # large value as "outside the model's valid range", not as a data point.
    out["cl_floor_hits"] = int(getattr(model, "cl_floor_hits", 0))
    out["cl_floor_unit_mask"] = np.asarray(getattr(model, "cl_floor_units", False)).copy()
    out["cl_floor_frac_units"] = float(np.asarray(getattr(model, "cl_floor_units", 0)).mean())
    out["cl_floor_frac_steps"] = float(getattr(model, "cl_floor_steps", 0)) / max(1, int(T_end / dt))
    out["mask"] = drive.mask
    return out


def physiological_mask(runs, key="cl_floor_frac_units"):
    """Boolean array: True where a run is inside the model's valid range.

    A run is REJECTED if any unit's ``Cl_in`` was ever clamped.  Clamping means
    intracellular chloride reached <= 0, where ``E_Cl = 26.7*log(Cl_in/Cl_ex)``
    puts the unit at -187 mV (floor 0.1 mM) against a resting -78 mV -- outside
    the range Liou Eq. 1 describes, at any floor value.  See PORT_NOTES.md
    "Chloride clamp: what a large hit count does and does not mean": the
    rejection is physiological, not numerical (results are insensitive to the
    floor over 500x).

    Accepts a single run dict, a list of them, or a DataFrame-like mapping.
    """
    if isinstance(runs, dict) and key in runs:
        runs = [runs]
    return np.array([float(r[key]) == 0.0 for r in runs], dtype=bool)


def assert_physiological(run, context=""):
    """Raise if ``run`` clamped.  Use in analysis paths that must not silently
    include out-of-range runs."""
    if float(run.get("cl_floor_frac_units", 0.0)) > 0.0:
        raise ValueError(
            "run left the model's valid range: %d clamp events on %.2f%% of "
            "units (Cl_in hit the floor; E_Cl <= -144 mV). %s"
            % (int(run.get("cl_floor_hits", 0)),
               100 * float(run["cl_floor_frac_units"]), context))
    return run
