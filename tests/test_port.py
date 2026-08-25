"""Equivalence tests for the LAS-Model port."""

import numpy as np
from scipy import signal

from las_model.kernels import (
    conv_field,
    conv_field_reference,
    gaussian_kernel_1d,
    gaussian_kernel_2d,
    sigma_pixels_from_fraction,
)


def matlab_conv2_field_linear(R, ker):
    """Literal transcription of conv2_field.m for topology='linear'.

    MATLAB: O = conv2(R, ker); dshift = (nO-nT)/2; O = circshift(O, -dshift);
            Output = O(1:nT(1), 1:nT(2));
    """
    nT = R.shape
    O = signal.convolve2d(R, ker, mode="full")
    nO = O.shape
    dshift = ((nO[0] - nT[0]) // 2, (nO[1] - nT[1]) // 2)
    O = np.roll(O, (-dshift[0], -dshift[1]), axis=(0, 1))
    return O[: nT[0], : nT[1]]


def test_separable_matches_direct():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((100, 100))
    for frac in (0.02, 0.03):
        s = sigma_pixels_from_fraction((100, 100), frac)
        k1, k2 = gaussian_kernel_1d(s[0]), gaussian_kernel_1d(s[1])
        fast = conv_field(field, k1, k2)
        slow = conv_field_reference(field, np.outer(k1, k2))
        assert np.abs(fast - slow).max() < 1e-12, frac


def test_matches_matlab_conv2_field():
    """Our 'same' convolution reproduces conv2_field.m exactly."""
    rng = np.random.default_rng(1)
    field = rng.standard_normal((100, 100))
    for frac in (0.02, 0.03):
        s = sigma_pixels_from_fraction((100, 100), frac)
        k1, k2 = gaussian_kernel_1d(s[0]), gaussian_kernel_1d(s[1])
        ker = np.outer(k1, k2)
        assert ker.shape[0] % 2 == 1 and ker.shape[1] % 2 == 1
        ours = conv_field(field, k1, k2)
        theirs = matlab_conv2_field_linear(field, ker)
        assert np.abs(ours - theirs).max() < 1e-12, frac


def test_kernel_normalisation_and_shape():
    """Kernel is 2K-1 taps wide, unit sum, and separable == mvnpdf sample."""
    for frac, n in ((0.02, 100), (0.03, 100)):
        sigma = n * frac
        K = int(np.ceil(2.5 * sigma))
        k1 = gaussian_kernel_1d(sigma)
        assert k1.size == 2 * K - 1
        assert abs(k1.sum() - 1.0) < 1e-15

        # MATLAB: K = mvnpdf(x, [0 0], Sigma.^2); K = K/sum(K(:))
        x = np.arange(-(K - 1), K, dtype=float)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        pdf = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2) / (2 * np.pi * sigma**2)
        pdf = pdf / pdf.sum()
        assert np.abs(gaussian_kernel_2d((sigma, sigma)) - pdf).max() < 1e-15


def test_1d_kernel_degenerate_dimension():
    """A singleton dimension gives the identity kernel [1.0]."""
    s = sigma_pixels_from_fraction((500, 1), 0.02)
    assert s == (10.0, 0.02)
    k2 = gaussian_kernel_1d(s[1])
    # sigma = 0.02 px -> K = ceil(0.05) = 1 -> single tap, unit weight
    assert k2.size == 1 and k2[0] == 1.0


def test_lfp_at_matches_compute_lfp():
    """Single-site readout must equal the array-wide computation everywhere.

    Regression guard: an earlier figure helper reversed the sliced kernel,
    which is a no-op at the array centre but mirrors the readout to the
    opposite side of the sheet near a boundary.  It passed a centre-location
    spot check and produced a figure in which focal onset was invisible.
    """
    from las_model.lfp import lfp_at, compute_lfp

    rng = np.random.default_rng(0)
    n_units, n_time = 512, 40
    epsc = rng.random((n_units, n_time))
    ipsc = -rng.random((n_units, n_time))
    ref = compute_lfp(epsc, ipsc)

    e_t, i_t = epsc.T.copy(), ipsc.T.copy()
    for loc in (0, 1, 17, n_units // 2, n_units - 2, n_units - 1):
        got = lfp_at(e_t, i_t, loc)
        fin = np.isfinite(got) & np.isfinite(ref[loc])
        assert np.allclose(got[fin], ref[loc][fin], atol=1e-9), f"lfp_at mismatch at loc={loc}"
        assert np.array_equal(np.isnan(got), np.isnan(ref[loc])), f"NaN prefix differs at loc={loc}"


def test_lfp_kernel_matches_paper():
    """sigma_LFP = 0.025 is in NORMALISED units -> e-folding = 0.025*N neurons."""
    from las_model.lfp import lfp_kernel

    n_units = 2000
    k = lfp_kernel(n_units)  # default must be the paper reading
    centre = len(k) // 2
    efold = int(np.argmax(k[centre:] < np.exp(-1)))
    expected = 0.025 * n_units
    assert abs(efold - expected) <= 2, f"e-folding {efold}, paper implies {expected}"


def test_wave_direction_rejects_radial_field():
    """A radially symmetric activation map has no wave direction (r2 == 0)."""
    import numpy as np
    from las_model.wave2d import wave_direction
    n = 50
    yy, xx = np.mgrid[0:n, 0:n]
    T = np.hypot(yy - 24.5, xx - 24.5)          # perfectly radial
    ang, spd, r2, k = wave_direction(T, (24.5, 24.5), 2.5)
    assert np.isnan(ang) and np.isnan(spd), (ang, spd)
    assert r2 < 1e-9, r2


def test_wave_direction_recovers_planar_field():
    """A planar activation ramp returns its true direction and speed."""
    import numpy as np
    from las_model.wave2d import wave_direction
    n = 50
    yy, xx = np.mgrid[0:n, 0:n]
    T = 0.5 * xx + 0.0 * yy                      # gradient 0.5 ms/unit along +x
    ang, spd, r2, k = wave_direction(T, (24.5, 24.5), 4.0)
    assert abs(ang) < 1e-6, ang                  # +x direction
    assert abs(spd - 2.0) < 1e-6, spd            # 1/0.5
    assert r2 > 1 - 1e-9, r2


def test_sigma_from_diffusion():
    """sigma = sqrt(2 D / dt), the conversion run_exp6 documents."""
    import numpy as np
    from las_model.run import sigma_from_diffusion
    assert abs(sigma_from_diffusion(200.0, 1.0) - 20.0) < 1e-12
    assert abs(sigma_from_diffusion(200.0, 2.0) - np.sqrt(200.0)) < 1e-12


def test_mnist_drive_schedule_matches_wc():
    """active_row_at must be bit-identical to wc_model.active_row_at.

    The WC study's row schedule is the thing we are deliberately NOT changing;
    if these ever diverge the two studies stop being comparable on stimulus
    delivery, which is the whole point of porting it.
    """
    from las_model.mnist_drive import active_row_at
    def wc(step, dt, image_duration, row_duration=None, n_rows=28):
        if row_duration is None:
            row_duration = image_duration / n_rows
        return min(int(((step * dt) % image_duration) / row_duration), n_rows - 1)
    for dt in (0.5, 1.0, 2.0):
        nst = int(round(250.0 / dt))
        mine = [active_row_at(s, dt, 250.0) for s in range(nst)]
        assert mine == [wc(s, dt, 250.0) for s in range(nst)], dt
        cnt = np.bincount(mine, minlength=28)
        assert cnt.sum() == nst, (dt, cnt.sum())
        assert (cnt > 0).all(), "some row never visited at dt=%s" % dt
        assert set(np.unique(cnt)) <= {nst // 28, nst // 28 + 1}, cnt
    # global counter and image-local counter must agree over a 10 s run
    nst = 250
    assert ([active_row_at(s, 1.0, 250.0) for s in range(10000)]
            == [active_row_at(s % nst, 1.0, 250.0) for s in range(10000)])


def test_mnist_weights_preserve_wc_sign_ratio():
    """Normalised bounds must keep WC's negative fraction, not just its shape."""
    from las_model.mnist_drive import (sample_mnist_weights, MNIST_WEIGHT_MIN,
                                       MNIST_WEIGHT_MAX, WC_WEIGHT_MIN,
                                       WC_WEIGHT_MAX)
    got = MNIST_WEIGHT_MIN / MNIST_WEIGHT_MAX
    want = WC_WEIGHT_MIN / WC_WEIGHT_MAX
    assert abs(got - want) < 1e-12, (got, want)
    w = sample_mnist_weights(np.random.default_rng(0))
    wc_frac = -WC_WEIGHT_MIN / (WC_WEIGHT_MAX - WC_WEIGHT_MIN)
    assert abs((w < 0).mean() - wc_frac) < 0.03, ((w < 0).mean(), wc_frac)
    assert abs(MNIST_WEIGHT_MAX - 1.0) < 1e-12, "max|w| must be 1 for scale to read in threshold units"


def test_sequential_masks_one_row_without_rescaling():
    """Sequential must SLICE the same weights, never boost them (WC behaviour)."""
    from las_model import params as P
    from las_model.mnist_drive import MNISTDrive
    p = P.exp1_template((50, 50))
    img = np.ones((28, 28))
    rng = np.random.default_rng(0)
    a = MNISTDrive(p, rng=np.random.default_rng(0), scale=10.0, sequential=False)
    b = MNISTDrive(p, rng=np.random.default_rng(0), scale=10.0, sequential=True)
    assert np.array_equal(a.weights, b.weights), "modes must share the weight vector"
    a.set_image(img); b.set_image(img); b.set_row(7)
    fa, fb = a._field[a.mask], b._field[b.mask]
    assert (fb != 0).sum() == 28, (fb != 0).sum()
    # the live row must be IDENTICAL to the same row of the simultaneous field
    fa2, fb2 = a._field.copy(), b._field.copy()
    r0, c0 = a.origin
    assert np.allclose(fa2[r0 + 7, c0:c0 + 28], fb2[r0 + 7, c0:c0 + 28])
    assert abs((fb != 0).sum() / 784.0 - 1.0 / 28) < 1e-9


def test_chloride_stays_positive_in_operating_range():
    """Cl_in must not reach 0: E_Cl = 26.7*log(Cl_in/Cl_ex) yields NaN there.

    Negative pixel weights inject hyperpolarising current, which Liou's own
    protocol never does (Fig-1C is +200 pA), so this failure mode is introduced
    by the WC-style signed weights and is NOT guarded in model.py -- Cl_in is
    unbounded below.  Measured: scale 60 leaves min Cl_in ~0.49 mM, scale 80
    goes non-positive.  This test pins the documented safe ceiling.
    """
    from las_model import params as P
    from las_model.run_mnist import run_mnist
    # A solid saturated block is a HARDER case than real digits (equal ink
    # fraction, but every negative weight fires at full strength over a
    # contiguous patch), and it goes non-finite by scale 60 without the clamp.
    # So there is no image-independent safe scale: the clamp, not a ceiling,
    # is what keeps the equations finite.
    imgs = np.zeros((4, 28, 28)); imgs[:, 8:20, 8:20] = 1.0
    r = run_mnist(imgs, scale=60.0, T_end=2000.0, seed=0)
    assert np.all(np.isfinite(r["Cl_in"])), "clamp failed to keep Cl_in finite"
    assert r["cl_floor_hits"] > 0, "this case is supposed to exercise the clamp"
    # and the clamp must be reported, so a sweep can reject the run
    r2 = run_mnist(imgs, scale=5.0, T_end=1000.0, seed=0)
    assert r2["cl_floor_hits"] == 0, "clamp fired at a benign scale"


def test_cl_floor_can_be_disabled():
    """The unguarded MATLAB behaviour must remain reachable."""
    from las_model import params as P
    from las_model.model import MeanFieldModel
    p = P.exp1_template((50, 50))
    m = MeanFieldModel(p, seed=0)
    assert m.cl_floor is not None and m.cl_floor_hits == 0
    m.cl_floor = None
    m.Cl_in = np.full_like(np.asarray(m.Cl_in, dtype=float), -1.0)
    m.Cl_in = np.maximum(m.Cl_in, -1.0)
    assert np.all(np.asarray(m.Cl_in) < 0), "with cl_floor=None nothing clamps Cl_in"


def test_drive_is_in_threshold_units():
    """scale=1 must put a saturated pixel at max|w| exactly at threshold current."""
    from las_model import params as P
    from las_model.mnist_drive import MNISTDrive, MNIST_WEIGHT_MAX
    from las_model.mnist_input import threshold_current
    p = P.exp1_template((50, 50))
    d = MNISTDrive(p, rng=np.random.default_rng(0), scale=1.0)
    d.weights = np.full((28, 28), MNIST_WEIGHT_MAX)
    d.set_image(np.ones((28, 28)))
    assert abs(d._field[d.mask].max() - threshold_current(p)) < 1e-9


def _main():
    """Run EVERY test_* in this module.

    The previous runner listed six calls explicitly and sat in the MIDDLE of
    the file, so every test defined below it -- including all the MNIST-input
    and LFP tests -- was silently skipped while the suite reported success.
    Auto-discovery is used instead so a new test cannot be added without being
    run.
    """
    import sys
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


def test_circular_conv_matches_matlab_fold():
    """conv_field(topology='circular') == conv2_field.m's circular branch."""
    import numpy as np
    from scipy import signal as sig
    from las_model.kernels import conv_field, gaussian_kernel_1d, sigma_pixels_from_fraction

    def matlab_circular(R, ker):
        nT = np.array(R.shape); O = sig.convolve2d(R, ker, mode='full')
        nO = np.array(O.shape); nRound = np.ceil(nO / nT).astype(int)
        dshift = (nO - nT) // 2
        pad = nRound * nT - nO
        if pad.any():
            O = np.pad(O, ((0, max(0, pad[0])), (0, max(0, pad[1]))))
        O = np.roll(np.roll(O, -dshift[0], 0), -dshift[1], 1)
        out = np.zeros(tuple(nT))
        for i in range(nRound[0]):
            for j in range(nRound[1]):
                out += O[i*nT[0]:(i+1)*nT[0], j*nT[1]:(j+1)*nT[1]]
        return out

    rng = np.random.default_rng(0)
    for n, frac in [(50, 0.02), (50, 0.03), (50, 0.10), (32, 0.05)]:
        F = rng.standard_normal((n, n))
        s = sigma_pixels_from_fraction((n, n), frac)
        k1, k2 = gaussian_kernel_1d(s[0]), gaussian_kernel_1d(s[1])
        got = conv_field(F, k1, k2, topology='circular')
        want = matlab_circular(F, np.outer(k1, k2))
        assert np.abs(got - want).max() < 1e-12, (n, frac, np.abs(got-want).max())


def test_torus_has_uniform_drive_and_conserves_mass():
    """A torus has no boundary: every unit gets identical recurrent drive."""
    import numpy as np
    from las_model.model import MeanFieldModel
    from las_model.params import exp1_template
    from las_model.kernels import conv_field, gaussian_kernel_1d

    O = MeanFieldModel(exp1_template((50, 50)), seed=0)
    O.add_standard_recurrent_connections(topology='circular')
    d = O.proj[0].project(np.ones((50, 50)))
    assert d.max() - d.min() < 1e-9, (d.min(), d.max())

    O.add_standard_recurrent_connections(topology='linear')
    dl = O.proj[0].project(np.ones((50, 50)))
    assert dl.min() / dl.max() < 0.6, dl.min() / dl.max()

    F = np.random.default_rng(1).standard_normal((50, 50))
    k = gaussian_kernel_1d(1.0)
    assert abs(conv_field(F, k, k, 'circular').sum() - F.sum()) < 1e-9


def test_torus_is_translation_equivariant():
    """Rolling the stimulus rolls the solution: the torus is homogeneous."""
    import numpy as np
    from las_model.run import run_exp1_torus

    kw = dict(T_end=3000.0, write_cycle=100.0, n=(50, 50), progress_every=0)
    a = run_exp1_torus(stim_x=(0.5, 0.5), **kw)['rate']
    b = run_exp1_torus(stim_x=(0.2, 0.8), **kw)['rate']
    assert np.abs(a - np.roll(np.roll(b, 15, 1), 35, 2)).max() < 1e-6


def test_movie_renders_without_clipping():
    """Movie writer produces a file and the default colour scale clips nothing."""
    import os, numpy as np
    from las_model.movie import activity_movie

    rng = np.random.default_rng(0)
    fake = dict(rate=rng.random((40, 12, 12)).astype(np.float32) * 30.0,
                t=np.arange(40) * 10.0, f_max=100.0)
    out = activity_movie(fake, '_test_movie.mp4', stride=2, fps=10, dpi=60)
    try:
        assert os.path.getsize(out['path']) > 1000, out
        assert out['clipped_fraction'] == 0.0, out
        assert out['n_frames'] == 20, out
        # stride=2 on a 10 ms write cycle -> 20 ms per kept frame; 20*10/1000 = 0.2x
        assert abs(out['speed_x'] - 0.2) < 1e-9, out
    finally:
        if os.path.exists(out['path']):
            os.remove(out['path'])


def test_mnist_embed_wraps_at_the_seam():
    """embed() must place modularly: a torus has no edge to truncate against.
    A plain slice assignment silently drops the overhang (real bug, fixed)."""
    import numpy as np
    from las_model.mnist_input import embed
    img = np.ones((28, 28))
    F = embed(img, n=50, origin=(40, 40))
    assert F.sum() == 28 * 28, ("digit truncated at seam: %.0f of 784" % F.sum())
    Fc = embed(img, n=50, origin=(11, 11))
    assert np.allclose(F, np.roll(np.roll(Fc, 29, 0), 29, 1))


def test_mnist_threshold_current_tracks_parameters():
    """scale is in multiples of threshold, so it must follow E_L and g_L."""
    import numpy as np
    from las_model import params as P
    from las_model.mnist_input import threshold_current
    p = P.exp1_template((8, 8))
    assert abs(threshold_current(p) - 52.0) < 1e-9, threshold_current(p)
    p.E_L = np.full((8, 8), -68.0)          # 10 mV further from phi0
    assert abs(threshold_current(p) - 92.0) < 1e-9, threshold_current(p)


def test_mnist_stimulus_is_translation_equivariant_on_torus():
    """The response to a digit must be a rigid shift of the response to the
    same digit placed elsewhere -- including across the wrap seam."""
    import numpy as np
    from las_model import params as P, model as M
    from las_model.mnist_input import embed
    rng = np.random.default_rng(0)
    img = (rng.random((28, 28)) > 0.85).astype(float)

    def run(origin):
        p = P.exp1_template((50, 50)); p.mask = None
        O = M.MeanFieldModel(p, seed=0)
        O.add_standard_recurrent_connections(sigma_E_frac=0.02, sigma_I_frac=0.03,
                                             topology='circular')
        F = embed(img * 104.0, 50, origin)
        O.ext.append(M.ExternalInput(deterministic=lambda a, b, t: F))
        for _ in range(60):
            O.step(1.0)
        return O.R_f / p.f_max

    Ra, Rc = run((11, 11)), run((40, 40))
    d = np.abs(Rc - np.roll(np.roll(Ra, 29, 0), 29, 1)).max()
    assert d < 1e-12, "not equivariant across seam: %.2e" % d


def test_mnist_row_major_matches_wc_convention():
    """Pixel (r, c) must land at the unit that WC-MNIST indexes as 28*r+c,
    so input-weight vectors transfer between the two studies unreordered."""
    import numpy as np
    from las_model.mnist_input import embed, patch_origin
    r0, c0 = patch_origin(50, 28)
    for (r, c) in [(0, 0), (7, 13), (27, 27)]:
        img = np.zeros((28, 28)); img[r, c] = 1.0
        rr, cc = np.argwhere(embed(img, 50) > 0)[0]
        assert (rr - r0, cc - c0) == (r, c), ((r, c), (rr, cc))


def test_meanfield_psc_matches_the_conductance_used_next_step():
    """MeanFieldModel.psc() must reconstruct the conductance the equations use.

    Input_E is decayed at the END of step() and the next arrival is written to
    _proj_value, which is not folded in until the TOP of the next step.  psc()
    adds it back.  Verified: recorded gE equals the conductance entering the
    membrane equation on the FOLLOWING step (measured 2.8e-17).
    """
    import numpy as np
    from las_model.model import MeanFieldModel, ExternalInput
    import las_model.params as P
    p_ = P.exp1_template((20, 20)); p_.mask = None
    O = MeanFieldModel(p_, seed=1)
    O.add_standard_recurrent_connections(sigma_E_frac=0.02, sigma_I_frac=0.03,
                                         topology="circular")
    O.ext = [ExternalInput(deterministic=lambda x1, x2, t: 200.0 *
                           (np.hypot((x1 - 1) / 20 - 0.5, (x2 - 1) / 20 - 0.5) < 0.15),
                           tmax=200.0)]
    for _ in range(120):
        O.step(1.0)
    gE_rec = O.Input_E.copy()
    for pr, val in zip(O.proj, O._proj_value):
        if pr.type == "E":
            gE_rec = gE_rec + val / p_.tau_syn_E
    epsc, _ = O.psc()
    O.step(1.0)
    gE_used = O.Input_E / np.exp(-1.0 / p_.tau_syn_E)   # undo end-of-step decay
    assert np.abs(gE_rec - gE_used).max() < 1e-12, "psc() conductance is not the one used"
    assert epsc.min() >= -1e-12, "EPSC must be non-negative (V < E_Esyn)"


def test_lfp_2d_delay_is_milliseconds_not_samples():
    """delay_ms must be converted by the RECORDING interval, not used raw.

    Liou's LFP.m hardcodes dT=6 because his recorder runs at dt=1 ms, so ms and
    samples coincide there.  Recording every 2 ms and passing 6 as samples
    would impose a 12 ms delay.
    """
    import numpy as np
    from las_model.lfp import lfp_2d
    rng = np.random.default_rng(0)
    e = rng.random((60, 25)); i_ = -rng.random((60, 25))
    _, dT1 = lfp_2d(e, i_, [12], (5, 5), dt_record=1.0)
    _, dT2 = lfp_2d(e, i_, [12], (5, 5), dt_record=2.0)
    assert dT1 == 6, f"at dt=1ms the delay should be 6 samples, got {dT1}"
    assert dT2 == 3, f"at dt=2ms the delay should be 3 samples, got {dT2}"


def test_lfp_2d_kernel_uses_normalised_sigma_and_wraps():
    """sigma_LFP is a FRACTION of extent (like sigma_E/sigma_I), and periodic."""
    import numpy as np
    from las_model.lfp import lfp_kernel_2d
    w = lfp_kernel_2d((50, 50), sigma=0.025)((25, 25)).reshape(50, 50)
    assert abs(w[25, 25] - 1.0) < 1e-12
    L = 0.025 * 50
    assert abs(w[25, 26] - np.exp(-1.0 / L)) < 1e-12, "e-folding must be sigma*n units"
    wc = lfp_kernel_2d((50, 50), sigma=0.025, periodic=True)((0, 0)).reshape(50, 50)
    assert abs(wc[0, 49] - wc[0, 1]) < 1e-12, "kernel must wrap on the torus"


def test_lfp_2d_electrode_stride_does_not_alias_on_the_grid():
    """Stride 83 on a 50-wide sheet must visit distinct rows AND columns.

    The WC study hit exactly this: stride 28 on a 28-wide image put every
    electrode in column 0, which carries no ink.  gcd(stride, row_len) == 1 is
    the condition that the electrodes walk diagonally instead.
    """
    from math import gcd
    els = [83 * k for k in range(30)]
    assert gcd(83, 50) == 1, "stride aliases onto a subset of columns"
    assert len({u // 50 for u in els}) == 30, "electrodes do not span distinct rows"
    assert len({u % 50 for u in els}) == 30, "electrodes do not span distinct columns"


def test_psc_docstring_does_not_claim_a_fixed_ipsc_sign():
    """Guard against the false claim being reinstated.

    An earlier docstring said ``ipsc < 0`` unconditionally.  Measurement
    refutes it (positive in 2.3% of samples on the 80 s focal run), and a
    reader who trusts a fixed sign will mis-handle the Lambda formula.  This
    test fails if the unconditional claim reappears.
    """
    import inspect
    from las_model.model import MeanFieldModel
    doc = inspect.getdoc(MeanFieldModel.psc) or ""
    assert "ipsc < 0``." not in doc, "psc() docstring reasserts an unconditional ipsc < 0"
    assert "NOT always negative" in doc, "psc() must document that ipsc can be positive"


def test_ipsc_sign_report_counts_positive_samples():
    """ipsc_sign_report must find wrong-sign samples, not assume there are none."""
    import numpy as np
    from las_model.lfp import ipsc_sign_report
    i_ = -np.ones((10, 4))
    assert ipsc_sign_report(i_)["frac_positive"] == 0.0
    i_[7, :2] = 5.0
    rep = ipsc_sign_report(i_)
    assert abs(rep["frac_positive"] - 2 / 40) < 1e-12
    assert abs(rep["max_frac_units_at_one_time"] - 0.5) < 1e-12


def test_lfp_2d_is_unfiltered_by_default_and_filter_is_opt_in():
    """The raw broadband Lambda must remain available; filtering is a display choice.

    Liou applies no filter, so unfiltered is the faithful default.  A 1 Hz
    high-pass also annihilates DC (in the WC study a saturated network rendered
    as a flat trace), so the unfiltered trace must stay reachable.
    """
    import numpy as np
    from las_model.lfp import lfp_2d
    rng = np.random.default_rng(3)
    n_t = 4000
    e = 100.0 + rng.random((n_t, 25))          # strong DC offset
    i_ = -rng.random((n_t, 25))
    raw, _ = lfp_2d(e, i_, [12], (5, 5), dt_record=2.0, report_ipsc_sign=False)
    filt, _ = lfp_2d(e, i_, [12], (5, 5), dt_record=2.0, filter_hz=1.0,
                     report_ipsc_sign=False)
    fin_r = np.isfinite(raw[:, 0]); fin_f = np.isfinite(filt[:, 0])
    assert abs(np.mean(raw[fin_r, 0])) > 1.0, "default output should retain DC"
    assert abs(np.mean(filt[fin_f, 0])) < 0.01 * abs(np.mean(raw[fin_r, 0])), \
        "1 Hz high-pass should remove the DC offset"
    assert not np.allclose(raw[fin_r, 0], filt[fin_f, 0])


def test_highpass_is_zero_phase():
    """filtfilt (forward-backward) must introduce no group delay.

    A causal lfilter would shift the peak later in time; zero-phase must not.
    """
    import numpy as np
    from las_model.lfp import highpass_eeg
    fs = 500.0
    t = np.arange(3000) / fs
    x = np.sin(2 * np.pi * 8 * t)[:, None]
    y = highpass_eeg(x, fs_hz=fs, cutoff_hz=1.0, order=4)
    mid = slice(500, 2500)
    lag = int(np.argmax(np.correlate(y[mid, 0], x[mid, 0], mode="same")) - len(x[mid]) // 2)
    assert abs(lag) <= 1, f"zero-phase filter shifted the signal by {lag} samples"


def test_patch_projection_geometry():
    """Pixel 1 -> rows/cols 1-3; pixel 2 -> cols 3-5; pixel 29 -> rows 3-5 (1-indexed)."""
    import numpy as np
    from las_model.patch_projection import build_projection
    W = build_projection((60, 60), weights=np.ones((28, 28, 3, 3)))
    for pix, exp_r, exp_c in [(0, [1, 2, 3], [1, 2, 3]),
                              (1, [1, 2, 3], [3, 4, 5]),
                              (28, [3, 4, 5], [1, 2, 3])]:
        u = np.asarray(W[:, pix].todense()).ravel().reshape(60, 60)
        ys, xs = np.nonzero(u)
        assert sorted(set(ys + 1)) == exp_r, (pix, sorted(set(ys + 1)))
        assert sorted(set(xs + 1)) == exp_c, (pix, sorted(set(xs + 1)))


def test_patch_weights_are_independent_per_connection():
    """7056 independent weights, not 784 shared across each pixel's 9 targets."""
    import numpy as np
    from las_model.patch_projection import sample_patch_weights
    w = sample_patch_weights(np.random.default_rng(0))
    assert w.shape == (28, 28, 3, 3)
    assert len(np.unique(w)) == w.size
    assert np.isclose(np.abs(w).max(), 1.0)


def test_patch_coverage_and_multiplicity():
    """60x60 leaves a 3-wide undriven stripe; 56x56 tiles exactly."""
    from las_model.patch_projection import coverage_report
    r60 = coverage_report((60, 60))
    assert r60["extent"] == 57 and not r60["tiles_exactly"]
    assert r60["n_undriven"] == 351
    assert set(r60["multiplicity_counts"]) == {0, 1, 2, 4}
    r56 = coverage_report((56, 56))
    assert r56["tiles_exactly"] and r56["n_undriven"] == 0


def test_sequential_patch_masks_image_row_not_network_row():
    """An active image row drives its full 3-unit band, preserving overlap.

    The band's absolute position depends on the drive's centring offset
    (``PatchDrive`` centres the 2*28 patch grid on the sheet), so the
    expectation is derived from ``d.offset`` rather than hard-coded: what is
    being pinned is that ALL THREE rows of the patch are driven, not where.
    """
    import numpy as np
    import las_model.params as P
    from las_model.mnist_drive import PatchDrive
    d = PatchDrive(P.exp1_template((60, 60)), rng=np.random.default_rng(0),
                   scale=1.0, n=60, sequential=True)
    d.set_row(14)
    d.set_image(np.ones((28, 28)))
    ys = np.nonzero(d._field.any(1))[0]
    base = d.offset[0] + 2 * 14
    assert list(ys) == [base, base + 1, base + 2], (ys, d.offset)
    assert d.offset == (2, 2), d.offset          # (60 - 56) // 2


def test_n56_drive_offset_is_zero():
    """The locked n=56 configuration must be unchanged by centring.

    At n=56 the patch grid already fills the sheet, so the offset is (0, 0)
    and every geometry result measured at n=56 still holds.  Larger sheets
    (n=84) centre the digit and leave an undriven surround for propagation.
    """
    import numpy as np
    import las_model.params as P
    from las_model.mnist_drive import PatchDrive
    d56 = PatchDrive(P.exp1_template((56, 56)), rng=np.random.default_rng(0), n=56)
    assert d56.offset == (0, 0), d56.offset
    d84 = PatchDrive(P.exp1_template((84, 84)), rng=np.random.default_rng(0), n=84)
    assert d84.offset == (14, 14), d84.offset
    d84.set_image(np.ones((28, 28)))
    ys = np.nonzero(np.asarray(d84._field).any(1))[0]
    # 28 patch origins at stride 2 (0..54) plus patch width 3 spans 57 rows
    assert ys.min() == 14 and ys.max() == 14 + 2 * 27 + 2, (ys.min(), ys.max())


def test_physiological_mask_rejects_clamped_runs():
    from las_model.run_mnist import physiological_mask, assert_physiological
    good = {"cl_floor_frac_units": 0.0, "cl_floor_hits": 0}
    bad = {"cl_floor_frac_units": 0.003, "cl_floor_hits": 2754}
    assert list(physiological_mask([good, bad])) == [True, False]
    assert_physiological(good)
    try:
        assert_physiological(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("assert_physiological did not raise on a clamped run")



def test_default_grid_tiles_exactly():
    """n=56 default: patches tile the sheet with no undriven units."""
    import las_model.patch_projection as PP
    rep = PP.coverage_report()
    assert rep["n_undriven"] == 0, rep
    assert rep["tiles_exactly"], rep
    assert sorted(rep["multiplicity_counts"]) == [1, 2, 4], rep


def test_compensation_modes_scale_as_expected():
    """1/m divides by m exactly; sqrt divides by sqrt(m); None leaves raw sums."""
    import las_model.patch_projection as PP
    m = PP.multiplicity_map((56, 56)).ravel()
    W0 = PP.build_projection((56, 56), rng=np.random.default_rng(3), compensate=None)
    Wm = PP.build_projection((56, 56), rng=np.random.default_rng(3), compensate="multiplicity")
    Ws = PP.build_projection((56, 56), rng=np.random.default_rng(3), compensate="sqrt")
    r0 = np.asarray(np.abs(W0).sum(1)).ravel()
    assert np.allclose(np.asarray(np.abs(Wm).sum(1)).ravel(), r0 / m, rtol=1e-10)
    assert np.allclose(np.asarray(np.abs(Ws).sum(1)).ravel(), r0 / np.sqrt(m), rtol=1e-10)


def test_run_defaults_are_n56_and_80s():
    """Runner defaults encode the configuration decisions of 2026-08-13."""
    import inspect, las_model.run_mnist as RM
    d = inspect.signature(RM.run_mnist).parameters
    assert d["n"].default == 56, d["n"].default
    assert d["T_end"].default == 80000.0, d["T_end"].default
    assert d["compensate"].default == "sqrt", d["compensate"].default




def _mnist_images(n=200):
    """First n MNIST training images as a (n, 784) float array in [0, 1]."""
    import glob, struct
    hits = glob.glob("mnist_data/MNIST/raw/train-images-idx3-ubyte") + \
           glob.glob("~/dev/wc-mnist-study/mnist_data/MNIST/raw/train-images-idx3-ubyte")
    if not hits:
        import pytest
        pytest.skip("MNIST idx file not available")
    with open(hits[0], "rb") as fh:
        _, cnt, r, c = struct.unpack(">IIII", fh.read(16))
        buf = fh.read(n * r * c)
    return (np.frombuffer(buf, dtype=np.uint8).reshape(n, r * c).astype(np.float64) / 255.0)


def test_sqrt_compensation_flattens_multiplicity_classes():
    """sqrt(m) is the mode that flattens MNIST drive across multiplicity classes.

    Measured exponent p in drive ~ m**p is 0.56 (spread 0.49-0.63 over weight
    draws), so sqrt undercorrects slightly (m4/m1 ~ 1.09) while dividing by m
    overcorrects and inverts (~0.55).  Guards against the transposed-image bug
    that once made p look like 0.74: assert the image array is (n, 784).
    """
    import las_model.patch_projection as PP
    m = PP.multiplicity_map((56, 56)).ravel()
    images = _mnist_images(200)
    assert images.shape == (200, 784), images.shape
    out = {}
    for mode in (None, "sqrt", "multiplicity"):
        W = PP.build_projection((56, 56), rng=np.random.default_rng(0), compensate=mode)
        d = np.abs(np.asarray(W @ images.T)).mean(1)
        out[mode] = d[m == 4].mean() / d[m == 1].mean()
    assert 1.0 <= out["sqrt"] < 1.25, out
    assert out["multiplicity"] < 0.7, out
    assert out[None] > 2.0, out
    assert abs(out["sqrt"] - 1.0) < abs(out["multiplicity"] - 1.0), out


def test_measured_exponent_matches_fit():
    """MEASURED_EXPONENT reproduces the fit it documents."""
    import las_model.patch_projection as PP
    images = _mnist_images(200)
    m = PP.multiplicity_map((56, 56)).ravel()
    ps = []
    for sd in range(3):
        W = PP.build_projection((56, 56), rng=np.random.default_rng(sd), compensate=None)
        d = np.abs(np.asarray(W @ images.T)).mean(1)
        v = np.array([d[m == k].mean() for k in (1, 2, 4)])
        ps.append(np.polyfit(np.log([1, 2, 4]), np.log(v), 1)[0])
    assert abs(np.mean(ps) - PP.MEASURED_EXPONENT) < 0.06, (np.mean(ps), PP.MEASURED_EXPONENT)


def test_float_exponent_accepted_and_matches_named_modes():
    import las_model.patch_projection as PP
    m = np.maximum(PP.multiplicity_map((56, 56)).ravel(), 1)
    a = PP.build_projection((56, 56), rng=np.random.default_rng(1), compensate=0.5)
    b = PP.build_projection((56, 56), rng=np.random.default_rng(1), compensate="sqrt")
    assert np.allclose(a.toarray(), b.toarray())
    c = PP.build_projection((56, 56), rng=np.random.default_rng(1), compensate=PP.MEASURED_EXPONENT)
    d = PP.build_projection((56, 56), rng=np.random.default_rng(1), compensate="measured")
    assert np.allclose(c.toarray(), d.toarray())


def test_unknown_compensation_mode_raises():
    import las_model.patch_projection as PP
    try:
        PP.build_projection((56, 56), rng=np.random.default_rng(0), compensate="mean")
    except ValueError:
        return
    raise AssertionError("unknown compensate mode should raise ValueError")


def test_focal_probe_matches_exp1_reference():
    """run_mnist(probe=...) reproduces run_exp1_torus's focal stimulus exactly.

    The probe exists because the MNIST drive cannot nucleate a Liou seizure:
    it is spatially distributed and sign-balanced, so it raises the mean rate
    and engages global inhibition everywhere instead of loading chloride in
    one compact patch.  With scale=0 the probe must therefore be numerically
    identical to the paper's Exp1 stimulus, or susceptibility measured with it
    is not comparable to the published seizure.
    """
    import numpy as np
    import las_model.params as P
    from las_model.run_mnist import run_mnist
    from las_model.run import run_exp1_torus
    imgs = np.zeros((4, 28, 28), dtype=np.float32)
    a = run_mnist(imgs, scale=0.0, T_end=6000.0, n=56, projection=True, seed=0,
                  rec_every=10, probe=dict(t=(2.0, 5.0), Ic=200.0, r=0.05, x=(0.5, 0.5)))
    b = run_exp1_torus(T_end=6000.0, n=(56, 56), write_cycle=10.0, Ic=200.0,
                       stim_t=(2.0, 5.0), stim_x=(0.5, 0.5), stim_r=0.05)
    fmax = float(np.mean(P.exp1_template((56, 56)).f_max))
    rb = np.asarray(b["rate"]).reshape(len(b["rate"]), -1) / fmax
    ra = np.asarray(a["rate"])
    assert ra.shape == rb.shape, (ra.shape, rb.shape)
    assert abs(float(ra.max()) - float(rb.max())) < 1e-4, (ra.max(), rb.max())
    assert np.allclose(ra[-1], rb[-1], atol=1e-4), float(np.abs(ra[-1] - rb[-1]).max())


def test_probe_none_leaves_drive_untouched():
    """probe=None must be bit-identical to the pre-probe code path."""
    import numpy as np
    from las_model.run_mnist import run_mnist
    imgs = np.tile(np.linspace(0, 1, 784, dtype=np.float32).reshape(1, 28, 28), (4, 1, 1))
    a = run_mnist(imgs, scale=5.0, T_end=3000.0, n=56, projection=True, seed=0, rec_every=10)
    b = run_mnist(imgs, scale=5.0, T_end=3000.0, n=56, projection=True, seed=0, rec_every=10,
                  probe=None)
    assert np.array_equal(a["rate"], b["rate"])


def test_lfp30_stride_does_not_alias_on_the_84_sheet():
    """30 electrodes at stride 233 on 84x84 must span 30 distinct rows AND columns.

    Same trap as the n=50 ring (stride 83, gcd 1).  A stride sharing a factor
    with the row length collapses the array onto a subset of columns, which on
    a centred patch drive means sampling only driven or only naive tissue.
    """
    from math import gcd
    n = 84
    idx = np.arange(30) * 233
    assert idx.max() < n * n
    assert gcd(233, n) == 1, "stride aliases onto a subset of columns"
    rows, cols = idx // n, idx % n
    assert len(set(rows.tolist())) == 30 and len(set(cols.tolist())) == 30
    # both tissue classes sampled: driven band is rows/cols 14..69 at n=84
    inb = (np.abs(rows - 41.5) <= 28) & (np.abs(cols - 41.5) <= 28)
    assert inb.sum() == 14 and (~inb).sum() == 16, (int(inb.sum()), int((~inb).sum()))


def test_lfp30_stride_on_the_wrapped_56_sheet_samples_both_tissue_classes():
    """The pilot sheet is n=56 and callers WRAP EL30; re-check the trap there.

    Regression on a real defect (2026-08-14): the aliasing property was verified
    at n=84 only, while every pilot run executed at n=56 with ``EL30 % (n*n)``.
    Wrapping does not inherit the n=84 guarantee -- at n=56 the array spans 30
    distinct columns but only 27 distinct ROWS.  What the criterion actually
    needs is (a) no two electrodes on the same unit and (b) both tissue classes
    sampled, so those are what is asserted.
    """
    from math import gcd
    from las_model import readout as RO
    n = 56
    idx = np.asarray(RO.EL30) % (n * n)
    assert gcd(233, n * n) == 1
    assert len(set(idx.tolist())) == 30, "electrodes collide on the wrapped sheet"
    rows, cols = idx // n, idx % n
    assert len(set(cols.tolist())) == 30
    assert len(set(rows.tolist())) == 27      # documented, not accidental
    _, dmask = RO.driven_channels(n=n)
    assert int(dmask.sum()) == 13 and int((~dmask).sum()) == 17, \
        (int(dmask.sum()), int((~dmask).sum()))


def test_undriven_units_are_the_seizure_readout_not_the_driven_band():
    """The MNIST drive DOES seize; the readout must be units with zero input.

    Regression on a real inversion (2026-08-13).  Scoring burstiness/low-band
    power on the DRIVEN units concluded "the MNIST drive cannot nucleate a
    seizure" -- the seizure was there but hidden under the stimulus.  The
    sensitive readout is the naive border: at n=84 with a centred 56x56 driven
    band, 3920 units receive no input, and their active fraction is EXACTLY
    zero until recruitment starts.

    Also pins the logical trap: a matched no-probe control establishes that the
    probe is not the CAUSE.  It says nothing about whether an event occurred.
    """
    n = 84
    rows, cols = np.divmod(np.arange(n * n), n)
    driven = (np.abs(rows - 41.5) <= 28) & (np.abs(cols - 41.5) <= 28)
    assert driven.sum() == 3136, int(driven.sum())
    assert (~driven).sum() == 3920, int((~driven).sum())
    # the driven band must be a strict subset, i.e. a naive border must exist
    assert driven.sum() < n * n, "no undriven tissue -> no recruitment readout"
    # distance-outside-band bins used for the front fit must be inside the sheet
    d = np.maximum(np.abs(rows - 41.5), np.abs(cols - 41.5)) - 28.0
    for lo in range(0, 14, 2):
        m = (~driven) & (d >= lo) & (d < lo + 2)
        assert m.sum() > 0, ("empty distance bin", lo)


def test_ictality_criterion_is_driven_area_morphology_not_spread():
    """The seizure criterion is discharge morphology in the DRIVEN area and its evolution.

    User correction 2026-08-13: spread beyond the stimulated area is neither necessary nor
    sufficient; activity can spread, or outlast the stimulus, without being ictal. Earlier
    work here wrongly treated recruitment of zero-input tissue as the definition. Spread,
    outlasting, and refresh-unlocking are CONFOUND EXCLUSIONS only.

    Also pins the discrimination that does work for refresh entrainment: vary image_duration
    and check the discharge interval does not follow it. Phase concentration alone is not
    enough -- the genuine seizure scores R=0.45 at the refresh period by coincidence.
    """
    import numpy as np
    n, band = 84, 56
    el = np.arange(30) * 233                      # the pinned 30-electrode montage
    r, c = el // n, el % n
    lo, hi = (n - band) // 2, (n - band) // 2 + band
    drv = (r >= lo) & (r < hi) & (c >= lo) & (c < hi)
    assert drv.sum() == 14, f"driven-channel count drifted: {int(drv.sum())} (expected 14)"
    assert (~drv).sum() == 16, "surround channels retained as secondary descriptors"

    # the morphology metrics must be computable from driven channels alone, with no
    # reference to any undriven channel
    t = np.arange(0, 4.0, 0.005)
    ipi = 0.26
    x = np.zeros_like(t)
    for k in range(int(t[-1] / ipi)):
        x += 40000.0 * np.exp(-0.5 * ((t - (0.1 + k * ipi)) / 0.025) ** 2)
    from scipy import signal as sg
    pk, _ = sg.find_peaks(x / x.std(), height=2.0, distance=int(0.12 / 0.005))
    got = float(np.median(np.diff(t[pk])))
    assert abs(got - ipi) < 0.02, f"IPI recovery broken: {got} vs {ipi}"

    # a stimulus-paced train would scale with image_duration; the measured Liou train does not
    stim_ratio, measured_ratio = 400.0 / 250.0, 285.0 / 255.0
    assert measured_ratio < 1.25 < stim_ratio, (
        "regression: the discharge interval must not track image_duration "
        f"(measured x{measured_ratio:.2f} vs stimulus x{stim_ratio:.2f})")

def test_sigma_ratio_default_sits_at_the_net_centre_sign_flip():
    """The paper's sigma_I/sigma_E = 1.5 puts the net centre weight at ~0.

    Pins the geometry behind treating the ratio as an operating-point axis
    rather than a fixed structural choice: the net centre
    g_E*K_E(0) - g_I_local*K_I(0) - g_I_global/N crosses zero near ratio 1.55,
    and the default 1.5 lands just below it.  If this stops holding, the
    sweep's sigma_I_frac range no longer brackets the sign flip.
    """
    import numpy as np
    from las_model.model import sigma_pixels_from_fraction, gaussian_kernel_1d

    def net_centre(n, sI_frac, sE_frac=0.02, g_E=100.0, g_I_local=250.0, g_I_global=50.0):
        sE = sigma_pixels_from_fraction((n, n), sE_frac)
        sI = sigma_pixels_from_fraction((n, n), sI_frac)
        kE, kI = gaussian_kernel_1d(sE[0]), gaussian_kernel_1d(sI[0])
        cE, cI = kE[len(kE) // 2] ** 2, kI[len(kI) // 2] ** 2
        return g_E * cE - g_I_local * cI - g_I_global / (n * n)

    assert net_centre(56, 0.02) < -15.0, "ratio 1.0 must be strongly net-inhibited"
    assert -3.0 < net_centre(56, 0.03) < 0.0, "default ratio 1.5 must sit just below zero"
    assert net_centre(56, 0.04) > 0.0, "ratio 2.0 must be net-excitatory at the centre"

    lo, hi = 0.03, 0.04
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if net_centre(56, mid) < 0.0:
            lo = mid
        else:
            hi = mid
    assert 1.4 < (0.5 * (lo + hi)) / 0.02 < 1.8, "sign flip must fall in ratio 1.4-1.8"


def test_run_mnist_exposes_connectivity_via_rec_kwarg():
    """The connectivity pass-through EXISTS -- via rec=, not via flat kwargs.

    History worth keeping: an earlier version of this test asserted that
    sigma_E_frac/g_E/g_I_local/... were absent from run_mnist's signature and
    concluded a "blocker" was outstanding.  It passed VACUOUSLY -- the
    pass-through had already been added as a nested ``rec=`` dict, so the flat
    names were legitimately absent while the capability was present.  Four
    parameter screens monkeypatched MeanFieldModel needlessly as a result.

    Lesson pinned here: test the CAPABILITY, never the spelling.
    """
    import inspect
    from las_model import run_mnist as rm
    from las_model.model import MeanFieldModel

    params = inspect.signature(rm.run_mnist).parameters
    assert "rec" in params, "rec= is the connectivity pass-through; do not remove it"

    # every kwarg of the connectivity constructor must be reachable through rec=
    conn = inspect.signature(MeanFieldModel.add_standard_recurrent_connections).parameters
    reachable = set(conn) - {"self"}
    for name in ("sigma_E_frac", "sigma_I_frac", "g_E", "g_I_local", "g_I_global"):
        assert name in reachable, f"{name} vanished from the connectivity constructor"

    src = inspect.getsource(rm.run_mnist)
    assert "rkw.update(rec)" in src, "rec= must be merged into the connectivity kwargs"
    assert 'rkw = dict(topology="circular")' in src, (
        "circular topology is the locked configuration for the MNIST sheet"
    )


def test_input_I_is_read_by_the_chloride_equation():
    """No inhibitory conductance parameter in this model is 'just' excitability.

    g_I_global / g_I_local land in Input_I, and Input_I appears in Eq. 3
    (Cl_in_inf), so any inhibitory gain is also a chloride-loading -- i.e.
    seizure-fuel -- parameter.  This structural fact is why the sweep reports
    Cl_early alongside every excitability result; if the coupling is ever
    refactored away, this test should fail loudly rather than silently
    invalidating that reasoning.
    """
    import inspect
    from las_model.model import MeanFieldModel

    src = inspect.getsource(MeanFieldModel.step)
    cl_line = [ln for ln in src.split("\n") if "Vd_Cl" in ln and "tau_Cl" in ln]
    assert cl_line, "chloride equation not found in step()"
    assert "Input_I" in cl_line[0], (
        "Eq. 3 no longer reads Input_I: the 'inhibitory gain is also chloride fuel' "
        "argument in LIOU_SWEEP_DESIGN.md needs re-deriving"
    )
    assert "gI = self.Input_I / p.f_max" in src, "Input_I must also drive Eq. 1"
    assert "g_sum = p.g_L + gE + gI + gK" in src, "Input_I must also shunt via g_sum"


def test_phi0_enters_only_through_the_threshold():
    """phi0 is a pure additive threshold offset -- one path, unlike Input_I.

    Pins the asymmetry that makes phi0 the high-authority axis and g_I_global
    the low-authority one, so the tiering in LIOU_SWEEP_DESIGN.md stays
    grounded in the equations.
    """
    import inspect
    from las_model.params import mean_field_2d_standard_template
    from las_model.model import MeanFieldModel

    p = mean_field_2d_standard_template((8, 8))
    assert "self.phi0 + u * self.phi_inf_gain" in inspect.getsource(type(p).phi_inf)
    assert "phi0" not in inspect.getsource(MeanFieldModel.step), (
        "phi0 now appears directly in step(): it is no longer a single-path "
        "threshold offset"
    )

def test_image_duration_default_is_250ms():
    """The entrainment control depends on this value.

    Every 'intrinsic vs refresh-paced' argument in LIOU_SWEEP_DESIGN.md is
    computed against a 250 ms / 4 Hz image refresh.  If the default changes,
    the recorded IPI/duration ratios (notably the 2:1 lock at g_I_local=250)
    no longer mean what the document says they mean.
    """
    import las_model.run_mnist as RM

    assert float(RM.IMAGE_DURATION_MS) == 250.0, (
        "IMAGE_DURATION_MS changed: re-run the duration-doubling control before "
        "trusting any intrinsic-vs-paced claim in LIOU_SWEEP_DESIGN.md"
    )


def test_amplitude_is_the_entrainment_robust_metric_not_ipi():
    """Documents the measured metric choice for sweep scoring.

    Doubling image duration moved discharge amplitude by <10% at every
    g_I_local level tested while moving IPI by up to 48%.  This is why the
    sweep scores on amplitude and treats IPI as a descriptor.  Values are the
    measured ones from liou_tier2_gains.npz; the test pins the DECISION so a
    future refactor that starts scoring on IPI has to confront it.
    """
    measured = {  # g_I_local: (amp_pct_change, ipi_pct_change) on 250 -> 500 ms
        120.0: (2.9, -6.4),
        180.0: (-7.8, -11.8),
        250.0: (-9.6, -48.0),
    }
    for g, (d_amp, d_ipi) in measured.items():
        assert abs(d_amp) < 10.0, f"g_I_local={g}: amplitude was supposed to be robust"
    assert abs(measured[250.0][1]) > 40.0, "the 2:1 lock at the default is the reason IPI is not the primary score"
    assert max(abs(a) for a, _ in measured.values()) < min(
        abs(i) for g, (_, i) in measured.items() if g == 250.0
    ), "amplitude must be more duration-robust than IPI for the scoring choice to hold"

def test_pilot_draws_are_paired_across_configurations():
    """Both configurations must receive IDENTICAL parameter draws per index.

    The WC sweep is paired (same bias_e/bias_i at a given seed in every config),
    which is what makes the simultaneous-vs-sequential contrast a dynamics
    difference rather than a sampling difference.  The Liou pilot seeds draws
    from param_seed = 1000 + draw, independent of run_seed, so the pairing
    holds and a 20-draw pilot is a strict subset of the 500-draw sweep.
    """
    import numpy as np

    def draw(k):
        rng = np.random.default_rng(1000 + k)
        return [rng.uniform(lo, hi) for lo, hi in
                [(1500., 20000.), (10., 120.), (1500., 20000.), (3., 12.),
                 (120., 400.), (-48., -38.), (60., 130.), (1., 5.),
                 (7., 30.), (0.020, 0.045), (4., 37.)]]

    for k in (0, 7, 19):
        assert draw(k) == draw(k), "draw must be reproducible from param_seed alone"
    assert draw(0) != draw(1), "different draw indices must give different parameters"
    # first 20 draws of a 500-draw sweep are the pilot draws, unchanged
    assert [draw(k) for k in range(20)] == [draw(k) for k in range(20)]


def test_axis_off_axes_still_retain_tick_label_objects():
    """Guards the figure-overlap checker against a false positive.

    ax.axis('off') hides tick labels but matplotlib RETAINS the Text objects
    with live bounding boxes, so a checker that walks get_xticklabels() on
    every axes reports collisions against invisible text.  The checker must
    skip axes whose `axison` is False.  (Same class of trap as retained
    out-of-view tick labels.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.axis("off")
    fig.canvas.draw()
    labels = [t for t in ax.get_xticklabels() if t.get_text().strip()]
    assert labels, "matplotlib no longer retains tick labels on axis('off') axes"
    assert ax.axison is False, "axison is the flag the checker must test"
    assert all(t.get_visible() for t in labels), (
        "these Text objects still report visible=True, which is why the checker "
        "must test ax.axison rather than t.get_visible()"
    )
    plt.close(fig)

def test_lfp_montage_derives_dt_record_in_milliseconds():
    """dt_record is MILLISECONDS and must come from the run's own time axis.

    Regression for a real pilot bug: readout.lfp_montage hardcoded
    dt_record=0.005, meaning seconds.  lfp_2d consumes it twice --
    dT = round(delay_ms/dt_record) and fs = 1000/dt_record -- so the single
    wrong unit produced a 1200-sample (6 s) axonal delay instead of 1 sample
    (6 ms) AND told the 1 Hz high-pass that fs was 200 kHz, disabling it.
    Nothing raised; every one of 54 pilot runs carried it.  Median discharge
    amplitude moved ~23% and discharge count ~37%.
    """
    import numpy as np
    from las_model import readout as RO

    T, nunits, n = 400, 25, 5
    rng = np.random.default_rng(0)
    z = dict(EPSC=rng.normal(size=(T, nunits)), IPSC=rng.normal(size=(T, nunits)),
             t=np.arange(T) * 5.0)                      # 5 ms recording interval
    L, t_s = RO.lfp_montage(z, n=n)
    # 6 ms delay at 5 ms sampling -> exactly 1 NaN sample, not 1200
    assert int(np.sum(~np.isfinite(L[:, 0]))) == 1, "axonal delay is not 6 ms"
    assert np.isclose(t_s[1] - t_s[0], 0.005), "time axis must be returned in seconds"

    # a seconds-valued time axis must be REFUSED, not silently mis-scaled
    z_bad = dict(z, t=np.arange(T) * 0.005)
    try:
        RO.lfp_montage(z_bad, n=n)
    except ValueError as e:
        assert "MILLISECONDS" in str(e)
    else:
        raise AssertionError("a seconds-valued time axis was silently accepted")


def test_readout_reproduces_the_spec_anchor_detector():
    """readout.morph must be the detector the spec's anchor numbers came from.

    The morphology helpers lived only as kernel-local defs while every anchor
    value in LIOU_PILOT_SPEC.md was measured, so a kernel reset made those
    numbers unreproducible.  They are a module now; this pins the contract that
    made them comparable (3-peak minimum per channel, 120 ms refractory,
    2-sigma height, MEDIAN across driven channels).
    """
    import inspect
    import numpy as np
    from las_model import readout as RO

    src = inspect.getsource(RO.morph)
    assert "distance=int(0.12 / ds)" in src, "120 ms refractory changed"
    assert "if len(pk) < 3:" in src, "3-peak-per-channel minimum changed"
    assert "height=h" in src and "h=2.0" in inspect.signature(RO.morph).__str__()

    # a clean 4 Hz train must be recovered at the right rate and amplitude
    fs, dur = 200.0, 20.0
    t_ = np.arange(0, dur, 1 / fs)
    x = np.zeros_like(t_)
    for k in range(int(dur * 4)):
        x += 1000.0 * np.exp(-0.5 * ((t_ - (0.1 + k * 0.25)) / 0.012) ** 2)
    L = np.column_stack([x, x, x])
    m = RO.morph(L, t_, np.arange(3), 0.0, dur, per_scan=False)
    assert abs(m["ipi"] - 250.0) < 6.0, "IPI of a 4 Hz train misread: %.1f" % m["ipi"]
    assert m["n"] >= 3 * 70, "peaks dropped: n=%d" % m["n"]


def test_driven_channel_count_is_stable():
    """13 of the 30 montage electrodes fall in the driven area at n=56.

    Every amplitude in the spec is a median over exactly these channels; if the
    montage or the band changes, the anchor numbers are not comparable and must
    be re-measured rather than reinterpreted.
    """
    from las_model import readout as RO
    chs, mask = RO.driven_channels(n=56)
    assert len(chs) == 13, "driven channel count changed: %d" % len(chs)
    assert mask.sum() == 13 and len(mask) == 30



def test_morph_excludes_nonfinite_samples_rather_than_dropping_the_channel():
    """lfp_2d NaNs the first dT samples by construction (the axonal delay).  An
    ``isfinite(x).all()`` guard therefore dropped EVERY channel for any window
    starting at t=0 -- so every early-window metric came back NaN with no error
    raised, and the pilot's growth (late/early) column was entirely missing.
    Assert a window containing the delay NaN still yields finite metrics."""
    import numpy as np
    from las_model import readout as RO
    rng = np.random.default_rng(0)
    T, C = 4000, 60
    L = rng.normal(0, 100.0, size=(T, C))
    # plant discharges so there is something to detect
    for k in range(40, T, 60):
        L[k, :] += 4000.0
    t = np.arange(T) * 0.005
    L[0, :] = np.nan          # the delay artifact
    ch = np.arange(30)
    m = RO.morph(L, t, ch, 0.0, t[-1] * 0.55, per_scan=False)
    assert m["n"] > 0, "window containing the delay NaN detected nothing"
    assert np.isfinite(m["amp"]), "amp is NaN despite only one non-finite sample"
    # and the value must match a window that simply starts after the NaN
    m2 = RO.morph(L, t, ch, t[1], t[-1] * 0.55, per_scan=False)
    assert abs(m["amp"] - m2["amp"]) / m2["amp"] < 0.05


def test_all_nan_channel_is_still_rejected():
    """The fix must not silently accept a channel with no usable data."""
    import numpy as np
    from las_model import readout as RO
    T = 2000
    L = np.full((T, 40), np.nan)
    L[:, 5:] = 0.0
    t = np.arange(T) * 0.005
    m = RO.morph(L, t, np.arange(30), 0.0, t[-1], per_scan=False)
    assert m["n"] == 0 and not np.isfinite(m["amp"])


def test_lfp_amplitude_ranks_are_not_preserved_under_dt_record_error():
    """Documents WHY the pilot had to be re-run rather than rescaled.  I argued
    the dt_record bug applied 'the same transform to every run', so rankings
    would survive and only absolute amplitudes needed restating.  Measured on
    the 54 stored runs, Spearman rho was 0.948 (not 1.0) with one run moving 15
    places -- and decisively, the zero-discharge SET changed: both
    depolarization-block anchors reported n=0 with the broken high-pass and
    several hundred discharges with it working.  A disabled filter is not a
    monotone rescaling; never rescale, always re-run."""
    assert 0.90 < 0.9483 < 1.0


def test_asymmetry_distinguishes_sawtooth_from_sinusoid():
    """The rise/decay asymmetry feature must separate the two morphologies it exists to
    separate.  A symmetric sinusoid -> ratio ~1; a fast-rise/slow-decay sawtooth -> >1."""
    import numpy as np
    fs = 200.0
    t = np.arange(0, 8.0, 1.0 / fs)
    sine = np.sin(2 * np.pi * 4.0 * t)
    saw = -2.0 * ((t * 4.0) % 1.0) + 1.0   # fast rise at wrap, slow linear decay
    def ratio(x):
        d = np.diff(x); up, dn = d[d > 0], -d[d < 0]
        return float(np.mean(up) / np.mean(dn))
    assert 0.8 < ratio(sine) < 1.25, ratio(sine)
    assert ratio(saw) > 2.0, ratio(saw)


def test_entrainment_lock_error_flags_integer_multiples_of_image_period():
    """A discharge train at an integer multiple of the 250 ms image period must be flagged
    as stimulus-locked; one at a non-integer multiple must not.  This distinction is what
    keeps refresh-entrained runs out of the ICTAL class."""
    import numpy as np
    IMG_MS = 250.0
    for ipi, want_locked in ((500.0, True), (750.0, True), (1250.0, True),
                             (330.0, False), (610.0, False)):
        err = abs(ipi / IMG_MS - round(ipi / IMG_MS))
        assert (err < 0.05) is want_locked, (ipi, err)


def test_blind_relabel_is_a_bijection_and_hides_anchor_identity():
    """The blinded zoom sheet's labels must not encode which runs are anchors -- the
    pilot's acceptance test is agreement between an UNCUED eyeball read and the automated
    verdict, so a leaked 'anchor' prefix or distinct colour invalidates it."""
    import numpy as np
    keys = ["anchor_a", "anchor_b"] + ["draw%02d" % i for i in range(8)]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(keys))
    m = {keys[i]: "R%02d" % (j + 1) for j, i in enumerate(order)}
    assert len(set(m.values())) == len(keys)
    assert all(not v.startswith("anchor") and "anchor" not in v for v in m.values())
    assert set(m) == set(keys)

def test_amp_label_never_rounds_a_nonzero_scale_to_zero():
    """The scale bar / clip annotation must stay interpretable at small amplitude.

    Regression: the label used "%.0f x 10^3" unconditionally, so a run whose
    shared-gain scale bar was 907 units printed "0x10^3" and a clip limit of 916
    printed "+/-0x10^3" -- exactly the low-amplitude runs where the reader needs
    the number to separate quiet from active.
    """
    import render_adjudication_figs as RAF
    for v in (907.0, 916.0, 430.0, 12.0, 3.2, 0.4):
        lab = RAF._amp_label(v)
        digits = lab.replace("$\\times$10$^3$", "").strip()
        assert float(digits) != 0.0, (v, lab)
    assert RAF._amp_label(2.1e4).startswith("21")          # big -> integer kilo
    assert RAF._amp_label(6300).startswith("6.3")          # mid -> one decimal
    assert "10$^3$" not in RAF._amp_label(430)             # small -> plain units


def test_adjudication_page_is_self_contained_and_blind():
    """The scoring page must carry no cue to the answer and no external file deps."""
    import build_adjudication_html as B, os, re
    if not os.path.isdir("adj_figs_webp"):
        return
    p, _nruns, _nb = B.build(out="_test_page.html")
    h = open(p).read()
    os.remove(p)
    # self-contained: no external src/href other than data: URIs built in JS
    assert not re.search(r'<(img|script|link)[^>]*\s(src|href)="(?!data:)', h), "external asset ref"
    # blind: no run key, no parameter name, no automated verdict leaks into the page
    for leak in ("anchor_", "g_I_local", "gKmax", "ICTAL", "ENTRAINED", "auto2", "lock_err"):
        assert leak not in h, leak
    # every verdict has a distinct key and the export writes the blind id
    assert h.count('type="radio"') == len(B.VERDICTS)
    assert len({k for _v, k, _c, _h in B.VERDICTS}) == len(B.VERDICTS)
    assert "blind_id" in h


def test_strip_lanes_share_one_time_axis_and_gutter_stays_in_register():
    """Heatmap + recruitment must scroll WITH the montage, not beside it.

    Requested 2026-08-14: those two lanes used to be a separate whole-run figure
    whose x-axis bore no relation to the montage's scroll position.  They are now
    axes of the strip figure sharing axS's x-axis.  Two things can silently break
    that and both are checked here:

      * a lane created without ``sharex=axS`` -- then it keeps its own autoscaled
        limits and the alignment is wrong by a run-dependent amount;
      * a change to the strip's lane stack that is not mirrored in the gutter --
        the page scales the gutter by the strip's height, so unequal natural
        heights throw every pinned label off the lane it names.
    """
    import glob, os
    import render_adjudication_figs as RAF
    from PIL import Image
    npz = sorted(glob.glob("pilot_out_v2/*.npz"))
    if not npz:
        return
    parts, _meta = RAF.render_run(npz[0], "_test_figs", label="AXTEST")
    # the standalone heat panel is gone; its content lives in the strip
    assert set(parts) == {"strip", "gutter", "diag"}, parts
    hs = Image.open(parts["strip"]).size[1]
    hg = Image.open(parts["gutter"]).size[1]
    assert hs == hg, "strip/gutter natural heights diverged: %d vs %d" % (hs, hg)
    for p in parts.values():
        os.remove(p)
    # The sharex wiring is checked at the SOURCE, not on the rendered image,
    # because the figure is closed inside render_run and the failure it guards
    # against is invisible in a single panel: axRec without sharex autoscales to
    # the bin-centre range (half a bin in from each end), which reads as a
    # plausible curve that is misaligned with the montage by a run-dependent
    # fraction of a second -- exactly the error the merge was meant to remove.
    src = open(RAF.__file__, encoding="utf-8").read()
    for lane in ("axH", "axRec", "axR", "axC"):
        assert "%s = figW.add_subplot(" % lane in src, lane
        decl = src.split("%s = figW.add_subplot(" % lane, 1)[1].split(")", 1)[0]
        assert "sharex=axS" in decl, "%s does not share the montage x-axis" % lane


def test_field_envelope_preserves_peak_recruitment_within_one_bin():
    """The stored full field is an 80 ms MAX envelope, not the raw field.

    That reduction is what makes 12 runs affordable (5 MB vs 200 MB each), but
    it is only legitimate if the quantity the sheet panel is read for --
    simultaneous recruited fraction -- survives it.  Max-pooling can only
    move a threshold crossing to the bin that contains it, so peak recruitment
    must be preserved exactly and its timing to within one bin.  A MEAN
    envelope would fail this: it dilutes a brief discharge below threshold.
    """
    import numpy as np
    from las_model import pilot_worker as PW

    rng = np.random.default_rng(0)
    T, U, dt_s = 4000, 64, 0.005
    t = np.arange(T, dtype=np.float32) * dt_s
    rate = rng.uniform(0.0, 0.02, size=(T, U)).astype(np.float32)
    # a brief, fully-synchronous discharge: 3 frames (15 ms) inside one 80 ms bin
    rate[1234:1237, :] = 0.9
    # and a sparser event recruiting a known fraction
    rate[2500:2503, : U // 4] = 0.8

    env, tb, b = PW.field_envelope(rate, t, bin_s=0.080)
    assert b == 16, b
    assert env.shape == (T // b, U), env.shape

    raw_peak = float((rate > PW.FIELD_THR).mean(axis=1).max())
    env_peak = float((env.astype(np.float32) > PW.FIELD_THR).mean(axis=1).max())
    assert env_peak == raw_peak == 1.0, (env_peak, raw_peak)

    # the quarter-sheet event survives at its true amplitude
    frac = (env.astype(np.float32) > PW.FIELD_THR).mean(axis=1)
    assert frac.max() == 1.0
    assert np.isclose(sorted(set(frac.tolist()))[-2], 0.25), sorted(set(frac.tolist()))[-3:]

    # timing preserved to within one bin
    raw_k = int((rate > PW.FIELD_THR).mean(axis=1).argmax())
    env_k = int(frac.argmax())
    assert abs(env_k - raw_k // b) <= 1, (env_k, raw_k // b)

    # A MEAN envelope loses brief events -- guards against someone
    # "simplifying" max to mean.  Sized so the dilution is unambiguous: ONE
    # frame at 0.9 in a 16-frame bin averages to 0.056, below FIELD_THR=0.10,
    # while the max envelope keeps it at 0.9.  (The 3-frame discharge above
    # averages to 0.169 and would survive a mean, which is why the control
    # needs its own shorter event rather than reusing it.)
    brief = np.full((T, U), 0.001, dtype=np.float32)
    brief[1600, :] = 0.9
    env_b, _tb, bb = PW.field_envelope(brief, t, bin_s=0.080)
    assert float((env_b.astype(np.float32) > PW.FIELD_THR).mean(axis=1).max()) == 1.0
    nb = T // bb
    mean_env = brief[: nb * bb].reshape(nb, bb, U).mean(axis=1)
    assert float((mean_env > PW.FIELD_THR).mean(axis=1).max()) == 0.0, \
        float(mean_env.max())


def test_field_recording_is_opt_in_and_does_not_perturb_the_default_path():
    """field_every=0 must leave run_mnist's outputs exactly as before.

    The 54-run adjudication set is already scored; if adding full-field
    recording changed the default path, those runs would no longer reproduce.
    """
    import inspect
    import numpy as np
    from las_model import run_mnist as RM

    sig = inspect.signature(RM.run_mnist)
    assert sig.parameters["field_every"].default == 0

    rng = np.random.default_rng(0)
    imgs = rng.uniform(0, 1, size=(4, 28, 28))
    # T_end long enough that the slow states develop SPATIAL structure: at
    # T_end=60 ms only one field frame is recorded, at t=0, where Cl_in is
    # still uniform at its equilibrium -- the per-unit check below would then
    # pass vacuously on a recorder that broadcast the field mean.
    kw = dict(T_end=600.0, n=12, seed=0, rec_every=5, sequential=False)
    a = RM.run_mnist(imgs, **kw)
    b = RM.run_mnist(imgs, field_every=100, **kw)

    for k in ("t", "rate", "Cl_in", "g_K", "V"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    assert "Cl_in_field" not in a and "Cl_in_field" in b
    # per-unit fields carry one row per recorded step, one column per unit
    assert b["Cl_in_field"].shape[1] == 12 * 12
    assert b["Cl_in_field"].shape[0] == b["t_field"].shape[0] > 0
    # and they are per-unit, not a broadcast of the field mean
    assert b["Cl_in_field"].std(axis=1).max() > 0.0


def test_normalised_rate_converts_to_hz_with_the_millisecond_f_max():
    """f_max is per-MILLISECOND; Hz needs the 1e3.

    T_refractory = 5 ms, so f_max = 1/5 = 0.2 per ms = 200 Hz.  The stored rate
    field is normalised to f_max, so a fully-recruited unit is 1.0 -> 200 Hz.
    Dropping the 1e3 reports it as 0.2 Hz, which is off by 1000x and lands in a
    range (sub-Hz) that looks superficially plausible on a figure -- the paper's
    Figure 6 colour scale runs 0-80 Hz, so a correct seizure must reach tens of
    Hz, not tenths.
    """
    import numpy as np
    from las_model.params import mean_field_2d_standard_template

    p = mean_field_2d_standard_template((4, 4))
    f_max = float(np.mean(p.f_max))
    assert np.isclose(f_max, 0.2), f_max          # per ms
    assert np.isclose(f_max * 1e3, 200.0)         # Hz

    # a unit at 40% of maximum is 80 Hz -- the top of the paper's Fig 6 scale
    assert np.isclose(0.4 * f_max * 1e3, 80.0)


def test_fullfield_page_images_are_in_exact_pixel_register():
    """The LFP strip and the 3136-unit sheet share one scroll box.

    That box has a single scrollLeft, so a given x is the same instant in both
    images ONLY if the two images have identical natural widths and each gutter
    matches its partner's natural height.  matplotlib rounds figsize*dpi, so
    neither holds automatically -- render_fullfield_page measures the strip and
    builds the raster to match, then resamples each gutter.  All four
    relationships are checked here because each one fails silently:

      * unequal widths  -> the sheet reports a different time than the LFP at
        the same scroll position, and nothing on screen says so;
      * unequal heights -> every pinned label sits on the wrong lane.

    Also asserts the sheet has EXACTLY one row per unit.  Any spatial pooling
    would make "full resolution" false while still producing a plausible image.
    """
    import glob
    import os
    import numpy as np
    from PIL import Image

    npz = sorted(glob.glob("fullfield_out/*.npz"))
    if not npz:
        return
    import render_fullfield_page as RFP

    parts, meta = RFP.render_run(npz[0], "_test_ff")
    # six parts since 2026-08-16: the whole-sheet OVERVIEW (sheetfit) and its
    # gutter were added so both panes fit a 1080x1920 portrait screen at once.
    assert set(parts) == {"strip", "gutter", "sheet", "sheetgut",
                          "sheetfit", "sheetfitgut"}, parts
    sz = {k: Image.open(v).size for k, v in parts.items()}
    assert sz["strip"][0] == sz["sheet"][0], \
        "strip/sheet widths differ: %s vs %s" % (sz["strip"][0], sz["sheet"][0])
    assert sz["strip"][1] == sz["gutter"][1], \
        "LFP gutter out of register: %s vs %s" % (sz["strip"][1], sz["gutter"][1])
    assert sz["sheet"][1] == sz["sheetgut"][1], \
        "sheet gutter out of register: %s vs %s" % (sz["sheet"][1], sz["sheetgut"][1])

    with np.load(npz[0], allow_pickle=True) as f:
        n_units = int(f["env"].shape[1])
        n = int(f["n"])
    assert n_units == n * n, (n_units, n)
    assert sz["sheet"][1] == n_units, \
        "sheet is %d px tall for %d units -- not one row per unit" % (
            sz["sheet"][1], n_units)

    for p in parts.values():
        os.remove(p)

    # The colour scale must span the full physical range: clipping at the
    # paper's 80 Hz put 98% of the depolarization-block runs above the ceiling,
    # which rendered them as a flat field carrying no information.
    assert RFP.HZ_MAX >= 200.0, RFP.HZ_MAX
    assert RFP.HZ_PAPER == 80.0, RFP.HZ_PAPER
    # and the page must pin the sheet's height rather than let it scale with
    # the width, or 3136 rows collapse to a few hundred screen pixels
    src = open("build_fullfield_html.py", encoding="utf-8").read()
    assert "sheet.style.height = sh" in src, "sheet height is left to height:auto"


def test_fullfield_scoring_options_cover_every_automated_verdict():
    """Every sheet_verdict must map to a radio-button option.

    The user's five options rename three of the four automated verdicts
    ("SEIZURE (Liou-like)" -> "FOCAL SEIZURE (Liou-like)", etc.), so the mapping
    is load-bearing: an unmapped verdict makes every run of that class score as a
    disagreement, which looks like the reader disagreeing rather than a wiring
    bug.  build() raises on an unmapped verdict; this checks the map is actually
    complete against the CSV, and that OTHER stays absent from it -- OTHER is for
    morphologies not yet encountered and must never count as agreement with an
    automated call.
    """
    import csv
    import os
    import build_fullfield_html as BFH

    assert "OTHER" in BFH.OPTIONS, BFH.OPTIONS
    assert "OTHER" not in BFH.VERDICT_TO_OPTION.values()
    assert set(BFH.VERDICT_TO_OPTION.values()) <= set(BFH.OPTIONS)
    assert set(BFH.OPT_COLOUR) == set(BFH.OPTIONS)

    if not os.path.exists("fullfield_verdicts.csv"):
        return
    with open("fullfield_verdicts.csv", newline="", encoding="utf-8") as f:
        seen = {r["sheet_verdict"] for r in csv.DictReader(f) if r.get("sheet_verdict")}
    missing = seen - set(BFH.VERDICT_TO_OPTION)
    assert not missing, "unmapped sheet_verdict(s): %s" % sorted(missing)


def test_fullfield_page_does_not_leak_its_own_classification():
    """The reader's call must be uncued.

    This project's acceptance criterion is agreement between an UNCUED human read
    and the automated verdict, so the page must not display its own verdict --
    nor the montage class or the sheet metrics, which are derived from the same
    pipeline and leak the answer just as effectively.  They ship in the JSON
    payload (the CSV export needs them) but must only reach the DOM behind the
    explicit reveal, and every vote must record whether the reveal was already
    open when it was cast.
    """
    import build_fullfield_html as BFH

    src = BFH.HTML
    assert "revealed_before_scoring" in src
    assert "let i = 0, revealed = false;" in src, "reveal must default to closed"
    # the verdict reaches the DOM only inside showMine(), which is reveal-gated
    body = src.split("function showMine()", 1)
    assert len(body) == 2, "showMine() missing"
    before, after = body[0], "function showMine()" + body[1]
    # draw() must not write r.verdict / r.cls / r.peak into the header.  Comments
    # are stripped first: the source deliberately NAMES those fields in a comment
    # explaining why they are withheld, and matching raw text would fail on the
    # documentation rather than on a real leak.
    import re as _re
    draw = before.split("function draw()", 1)[1] if "function draw()" in before else ""
    draw_code = _re.sub(r"/\*.*?\*/", "", draw, flags=_re.S)
    draw_code = _re.sub(r"//[^\n]*", "", draw_code)
    for banned in ("r.verdict", "r.cls", "r.peak"):
        assert banned not in draw_code, "draw() leaks %s before reveal" % banned
    gated = after.split("function showMine()", 1)[1].split("function score(", 1)[0]
    assert "if (!revealed)" in gated, "showMine() is not reveal-gated"


def test_fullfield_page_javascript_actually_parses():
    """The emitted page must be valid JS, not merely valid Python.

    Shipped broken on 2026-08-15: the JS lives inside a Python triple-quoted
    template, and a single-backslash ``\\n`` in the CSV-quoting regex was consumed
    by the PYTHON parser and emitted as a raw newline.  That split the regex
    literal, which is a syntax error, and a syntax error anywhere in the script
    means the whole script is discarded -- so ``draw()`` never ran and the page
    rendered as four lines of alt text with the header stuck on its template
    placeholders.  Nothing on the Python side noticed: the template was a
    perfectly valid string and build() returned normally.

    Two things are checked, because the first alone would not have caught it:
      * the emitted script parses (esprima), and
      * the CSV-quoting regex survived as an ESCAPE rather than a raw newline.
    """
    import json as _json
    import build_fullfield_html as BFH

    try:
        import esprima
    except ImportError:
        return                      # parser unavailable; the build-time guard still runs

    # Format with every placeholder the template declares, discovered rather
    # than hardcoded: a new field (pool, added 2026-08-16) otherwise makes this
    # test fail with KeyError on a page that is perfectly fine.
    import string as _string
    fields = {f for _, f, _, _ in _string.Formatter().parse(BFH.HTML) if f}
    supply = dict(runs="[]", pxps=BFH.PX_PER_S, win=BFH.WINDOW_S, n=0,
                  opts=_json.dumps(BFH.OPTIONS),
                  ocol=_json.dumps(BFH.OPT_COLOUR), stamp="test", btns="",
                  pool=4, gutw=214)
    missing = fields - set(supply)
    assert not missing, "template gained placeholder(s) %s -- add them here" % sorted(missing)
    html = BFH.HTML.format(**{k: supply[k] for k in fields})
    js = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    esprima.parseScript(js)         # raises on the 2026-08-15 defect

    # the regex must contain a two-character escape, not a literal newline
    i = js.find("return /[")
    assert i != -1, "CSV-quoting regex not found"
    lit = js[i:js.index(".test(s)", i)]
    assert "\\n" in lit, "escape was swallowed: %r" % lit
    assert "\n" not in lit, "raw newline inside the regex literal: %r" % lit


def test_whole_sheet_overview_is_pooled_not_resampled():
    """The fit-to-screen sheet must be a MAX-POOLED raster, not a browser squash.

    Requested 2026-08-15: see the full vertical extent of both panes at once on a
    1080x1920 portrait display, which leaves ~870 px per pane for 3136 unit rows.
    Letting the browser scale the full image there is nearest-neighbour: it
    samples ~800 rows and discards the rest.  Measured over all 12 runs that
    turns 21 time bins containing real activity completely blank, and the losses
    concentrate in the SPARSE runs (9 bins in anchor_gKmax120_seq, 7 in
    anchor_gIlocal600_seq) -- exactly where one missed discharge changes the
    verdict.  A max pool over the same rows loses none, which is the property
    asserted here on real run data.

    Also checks the overview and its gutter come out at equal heights, since the
    page scales the gutter by the image's factor.
    """
    import glob
    import os
    import numpy as np
    from PIL import Image

    npz = sorted(glob.glob("fullfield_out/*.npz"))
    if not npz:
        return
    import render_fullfield_page as RFP

    # pick the SPARSEST available run: that is where resampling loses events
    def _active(p):
        with np.load(p, allow_pickle=True) as f:
            e = np.asarray(f["env"], np.float32)
            return float((e * float(f["f_max"]) * 1e3 > 20.0).mean())
    target = min(npz, key=_active)

    with np.load(target, allow_pickle=True) as f:
        hz = np.asarray(f["env"], np.float32).T * float(f["f_max"]) * 1e3
        n = int(f["n"])
    act = hz > 20.0
    pool = int(np.ceil(n * n / float(RFP.FIT_ROWS)))
    assert pool >= 2, "no pooling needed; test assumptions stale"

    nrow = (hz.shape[0] // pool) * pool
    pooled = hz[:nrow].reshape(-1, pool, hz.shape[1]).max(axis=1) > 20.0
    keep = np.round(np.linspace(0, hz.shape[0] - 1, pooled.shape[0])).astype(int)
    nn = act[keep]

    lost_pool = int((act.any(0) & ~pooled.any(0)).sum())
    lost_nn = int((act.any(0) & ~nn.any(0)).sum())
    assert lost_pool == 0, "max pool lost %d active time bins" % lost_pool
    assert lost_nn > 0, (
        "nearest-neighbour lost nothing on %s, so this test no longer "
        "demonstrates why pooling is required" % os.path.basename(target))

    # The renderer must actually USE a max pool.  Asserting the property on a
    # locally-computed pool (above) would still pass if render_run switched to
    # mean-pooling -- verified by doing exactly that, which is the plausible
    # "simplification".  So compare the emitted raster against both candidates:
    # a mean pool is dimmer everywhere activity is sparse, so its brightest
    # pixel falls short of the max pool's.
    src = open(RFP.__file__, encoding="utf-8").read()
    assert ".max(axis=1)" in src.split("hz_pool =", 1)[1].split("\n", 1)[0], \
        "overview raster is not max-pooled"

    parts, _m = RFP.render_run(target, "_test_fit")
    assert "sheetfit" in parts and "sheetfitgut" in parts, sorted(parts)
    hf = Image.open(parts["sheetfit"]).size
    hg = Image.open(parts["sheetfitgut"]).size
    assert hf[1] == hg[1], "overview/gutter heights differ: %d vs %d" % (hf[1], hg[1])
    assert hf[1] <= RFP.FIT_ROWS, hf[1]
    assert hf[0] == Image.open(parts["strip"]).size[0], "overview width off the strip"
    for p in parts.values():
        os.remove(p)


def test_fullfield_panes_share_one_horizontal_origin():
    """The LFP and sheet x-axes must line up EXACTLY on screen.

    Requested 2026-08-16.  Two distinct things have to hold, and only the first
    was ever true:

    1. RASTER LEVEL -- a second mark must land on the same pixel column in the
       strip and in both sheet rasters.  Checked by locating the rules in the
       rendered images and comparing to round(s * W / T_s).

    2. PAGE LEVEL -- the panes must share ONE grid column.  They did not: the
       page had a nested grid for the montage row, so the gutter column width
       was computed separately per pane.  Both gutters are 214 px naturally but
       are shown at different heights, and with ``width:auto`` each width follows
       its own aspect ratio -- 161 px for the LFP gutter against 237 px for the
       sheet's.  The panes' t=0 therefore sat 76 px apart, which at the default
       zoom is 0.88 s of apparent time shift: the same x was a different instant
       in each pane, which is exactly what the reviewer saw.
    """
    import glob
    import os
    import numpy as np
    from PIL import Image

    npz = sorted(glob.glob("fullfield_out/*.npz"))
    if not npz:
        return
    import build_fullfield_html as BFH
    import render_fullfield_page as RFP

    # ---- 2. page level: one grid, explicit equal gutter widths --------------
    src = BFH.HTML
    assert "#striprow" not in src.split("/*", 1)[0] or "grid-template-columns" in src
    assert 'gut.style.width = GUTW + "px"' in src, \
        "LFP gutter width is not pinned -- width:auto reintroduces the 76 px offset"
    assert 'sgut.style.width = GUTW + "px"' in src, "sheet gutter width is not pinned"
    assert 'style.width = "auto"' not in src, "a gutter still uses width:auto"

    # ---- 1. raster level: rules on identical columns ------------------------
    parts, meta = RFP.render_run(npz[0], "_test_align")
    W = Image.open(parts["strip"]).size[0]
    T_s = float(meta["T_s"])
    expect = np.array([round(s * W / T_s) for s in range(1, int(T_s) + 1)])

    def _bright(p):
        a = np.asarray(Image.open(p).convert("L")).astype(np.int16).mean(0)
        return np.where(a > np.median(a) + 1.5)[0]

    def _dark(p):
        a = np.asarray(Image.open(p).convert("L")).astype(np.int16)
        band = a[: int(a.shape[0] * 0.72)].mean(0)
        return np.where(band < np.median(band) - 0.25)[0]

    for name, cols in (("strip", _dark(parts["strip"])),
                       ("sheet", _bright(parts["sheet"])),
                       ("sheetfit", _bright(parts["sheetfit"]))):
        assert len(cols), "no time rules found in %s" % name
        off = max(abs(cols[np.argmin(np.abs(cols - x))] - x) for x in expect)
        assert off == 0, "%s second marks are off by up to %d px" % (name, off)

    # every image spans the same pixel width, or the shared axis is meaningless
    widths = {k: Image.open(parts[k]).size[0]
              for k in ("strip", "sheet", "sheetfit")}
    assert len(set(widths.values())) == 1, widths

    # ---- 3. the sheet DATA sits where the axis says ------------------------
    # The rule check above cannot see a shifted raster, because _time_rules
    # draws the marks after the shift so they move with it -- verified by
    # shifting the raster deliberately, which passed the rule check.  So compare
    # the image's column-brightness profile against the data profile under the
    # raster's own pixel->bin mapping.  A one-bin lag drops the correlation
    # measurably (0.961 -> 0.937 on anchor_default_seq), which is the signal.
    with np.load(npz[0], allow_pickle=True) as f:
        hz = np.asarray(f["env"], np.float32).T * float(f["f_max"]) * 1e3
    prof = np.asarray(Image.open(parts["sheet"]).convert("L"),
                      dtype=np.float32).mean(0)
    nb = hz.shape[1]
    take = np.clip((np.arange(W) * nb) // W, 0, nb - 1)
    dp = hz.mean(0)
    # Scan a range of lags and require the BEST fit to be zero.  Comparing only
    # against a one-bin lag was not enough: a deliberate 58 px (6-bin) shift
    # still passed that version, because at 6 bins both candidates are equally
    # wrong.  The argmax over a window is agnostic to the shift's size.
    cs = {L: np.corrcoef(dp[np.clip(take + L, 0, nb - 1)], prof)[0, 1]
          for L in range(-10, 11)}
    best = max(cs, key=cs.get)
    assert best == 0, (
        "sheet raster is time-shifted by ~%+d bins (corr %.4f there vs %.4f at "
        "zero lag)" % (best, cs[best], cs[0]))
    assert cs[0] > 0.90, cs[0]

    for p in parts.values():
        os.remove(p)


def test_sheet_movie_frame_rates_are_on_the_right_side_of_nyquist():
    """The two movie tiers exist for a measured reason; keep them honest.

    The discharge rhythm on the ictal runs is ~16.7 Hz (median IPI 60 ms,
    measured from rate_trace).  The stored envelope is 80 ms bins, i.e. 0.75
    samples per cycle -- BELOW Nyquist, so an overview movie cannot show
    discharges and any flicker in it is aliasing.  The detail tier samples the
    raw field every 20 ms, giving 3 samples per cycle.

    This asserts the arithmetic against real data rather than trusting the
    constants, because the failure mode is silent: a movie built at 80 ms looks
    perfectly fine and simply lacks the phenomenon it appears to show.
    """
    import os
    import numpy as np
    from scipy import signal

    p = "fullfield_out/anchor_default_seq.npz"
    if not os.path.exists(p):
        return
    import render_movies as RM

    with np.load(p, allow_pickle=True) as f:
        rt = np.asarray(f["rate_trace"], np.float32)
        t = np.asarray(f["t"], np.float32)
        t_env = np.asarray(f["t_env"], np.float32)
    dt_s = float(t[1] - t[0])
    pk, _ = signal.find_peaks(rt, height=0.3 * float(rt.max()),
                              distance=max(1, int(0.03 / dt_s)))
    assert len(pk) > 50, len(pk)
    ipi_s = float(np.median(np.diff(t[pk])))
    f_hz = 1.0 / ipi_s
    assert 10.0 < f_hz < 30.0, f_hz            # sanity: it is a fast rhythm

    env_dt = float(t_env[1] - t_env[0])
    assert ipi_s / env_dt < 2.0, (
        "the envelope now resolves the rhythm (%.2f samples/cycle); the "
        "overview caveat in the page is stale" % (ipi_s / env_dt))

    detail_dt = 4 * dt_s                        # stride=4 on the raw field
    assert ipi_s / detail_dt >= 2.0, (
        "detail tier is below Nyquist at %.1f ms/frame (%.2f samples/cycle)"
        % (detail_dt * 1e3, ipi_s / detail_dt))

    # the page must state the caveat, not just the module
    import build_fullfield_html as BFH
    assert "Nyquist" in BFH.HTML, "the aliasing caveat is missing from the page"
    assert "6 s window" in BFH.HTML, "detail movie's limited window is not stated"

    # playback-speed claims must match fps x frame interval
    assert abs(RM.FPS * 0.080 - 2.0) < 1e-9
    assert abs(RM.FPS * 0.020 - 0.5) < 1e-9


def test_fullfield_page_builds_without_movies():
    """Movies are optional; scoring must not depend on them.

    Rendering all 13 takes ~30 s of encoding, and the page's primary job is
    blind scoring from the montage and sheet.  A missing ff_movies/ must
    degrade to a hidden movie pane, not a broken build.
    """
    import os
    import build_fullfield_html as BFH

    if not os.path.isdir("ff_figs_webp"):
        return
    got = BFH._movies("ff_movies_does_not_exist", "anchor_default_seq")
    assert got == {}, got
    out, n, _checked = BFH.build(out="_test_nomov.html",
                                 mov_dir="ff_movies_does_not_exist")
    page = open(out, encoding="utf-8").read()
    assert '"mov": {}' in page or '"mov":{}' in page, "movie dict not empty"
    assert 'id="movwrap"' in page, "movie pane markup should still be present"
    os.remove(out)


def test_confusable_scoring_options_are_not_adjacent():
    """Lookalike verdicts must sit in different groups, with undo available.

    A real mis-click cost two runs on 2026-08-16: "FOCAL entrained, not a
    seizure" and "FOCAL SEIZURE (Liou-like)" share a leading word and sat at
    keys 1 and 3 with an unrelated option between them, so a scan for "FOCAL"
    hit the wrong one -- on the first two runs scored, before the key positions
    were familiar.  The options are now grouped by the distinction being judged
    (seizure / not a seizure), so a slip between the two crosses a visible
    divider, and `u` undoes the last call.

    Checked here because both properties are easy to lose in a later edit and
    neither fails loudly: a regrouped list still renders, and a missing undo is
    invisible until someone needs it.
    """
    import build_fullfield_html as BFH

    # every option appears exactly once, in exactly one group
    flat = [o for _lbl, opts in BFH.OPT_GROUPS for o in opts]
    assert flat == BFH.OPTIONS, (flat, BFH.OPTIONS)
    assert len(set(flat)) == len(flat), flat

    _home = {o: gi for gi, (_l, opts) in enumerate(BFH.OPT_GROUPS) for o in opts}
    home_of = _home.__getitem__

    # NO TWO OPTIONS IN DIFFERENT GROUPS MAY SHARE A LEADING WORD.  Applied
    # 2026-08-16 after the grouping above proved to treat a symptom: the
    # entrained option was renamed from "FOCAL entrained, not a seizure" to
    # "ENTRAINED, not a seizure" so it no longer collided with "FOCAL SEIZURE
    # (Liou-like)" at a glance.
    #
    # NARROWED 2026-08-19 from "no two options" to "no two options in different
    # groups".  The user's renamed subtypes both begin "SEIZURE:" -- deliberately,
    # since both ARE seizures and the shared prefix states that.  What made the
    # original collision costly was that the two lookalikes sat on OPPOSITE sides
    # of the seizure/not-seizure line, so a slip changed the answer to the
    # question the study asks.  A slip between two same-group options changes only
    # the subtype, which the second scoring pass showed is the genuinely
    # ambiguous axis anyway (the only two calls that moved between passes were
    # within the seizure class).  Cross-group collisions remain forbidden; the
    # within-group pair is instead required to differ at its FIRST DISTINGUISHING
    # WORD, checked below.
    firsts = {}
    for o in BFH.OPTIONS:
        gi = home_of(o)
        w = o.split()[0].strip(",:").upper()
        prev = firsts.get(w)
        assert prev is None or prev[1] == gi, (
            "options %r and %r are in different groups and share the leading "
            "word %r -- a slip between them changes the seizure call"
            % (prev[0], o, w))
        firsts[w] = (o, gi)

    # Options sharing a leading word must diverge immediately after it, so the
    # eye has a distinguishing token in a fixed position rather than having to
    # read to the end of a long label.
    from collections import defaultdict
    byfirst = defaultdict(list)
    for o in BFH.OPTIONS:
        byfirst[o.split()[0].strip(",:").upper()].append(o)
    for w, opts in byfirst.items():
        if len(opts) < 2:
            continue
        seconds = [o.split()[1].strip(",:").upper() for o in opts]
        assert len(set(seconds)) == len(seconds), (
            "options %r share both their first and second word" % opts)

    # options that differ only by whether they ARE a seizure stay in different
    # groups, so a slip between them also crosses a visible divider
    home = {o: gi for gi, (_l, opts) in enumerate(BFH.OPT_GROUPS) for o in opts}
    seiz = [o for o in BFH.OPTIONS if "SEIZURE" in o.upper()
            and "not a seizure" not in o]
    notseiz = [o for o in BFH.OPTIONS if "not a seizure" in o]
    assert seiz and notseiz, (seiz, notseiz)
    assert {home[o] for o in seiz}.isdisjoint({home[o] for o in notseiz}), \
        "seizure and non-seizure options share a group"

    # the group headers must reach the page, or the divider is invisible
    src = BFH.HTML
    assert "gsep" in src, "group divider CSS/markup missing"
    assert "function undo()" in src, "undo was removed"
    assert 'k === "u"' in src, "undo has no key binding"


def test_score_join_maps_human_verdicts_onto_automated_classes():
    import join_adjudication_scores as J
    assert J.agree("seizure", "ICTAL") is True
    assert J.agree("seizure", "ENTRAINED") is False
    assert J.agree("entrained", "ENTRAINED") is True
    assert J.agree("no-seizure", "QUIET_OR_BLOCK") is True
    assert J.agree("no-seizure", "ACTIVE_NONICTAL") is True
    assert J.agree("no-seizure", "ICTAL") is False
    assert J.agree("unsure", "ICTAL") is None      # abstention is not a disagreement
    # every automated class the pilot can emit is reachable from some human verdict
    emitted = {"ICTAL", "ENTRAINED", "ACTIVE_NONICTAL", "QUIET_OR_BLOCK", "NO_DISCHARGE"}
    covered = {"ICTAL", "ENTRAINED"} | set(J.MAP["no-seizure"])
    assert emitted <= covered, emitted - covered


def test_lfp_montage_sigma_defaults_to_lfp_2d_and_is_settable():
    """sigma= must reach lfp_2d, and omitting it must change nothing.

    The default path is load-bearing: every scored artifact (54 pilot runs, 12
    full-field re-runs) was produced with no sigma argument, so if adding the
    parameter altered the default output those runs would stop reproducing.
    Widening it must also actually change the output -- a pass-through that is
    silently ignored would let a "sigma 0.11" sweep run at 0.025.
    """
    import numpy as np
    import inspect
    from las_model import readout as RO
    from las_model import lfp as LFP

    assert "sigma" in inspect.signature(RO.lfp_montage).parameters

    n, T = 56, 400
    rng = np.random.default_rng(0)
    z = dict(t=np.arange(T, dtype=float) * 5.0,
             EPSC=rng.normal(size=(T, n * n)),
             IPSC=rng.normal(size=(T, n * n)))

    d_none = RO.lfp_montage(z, n=n)[0]
    d_expl = RO.lfp_montage(z, n=n, sigma=0.025)[0]
    wide = RO.lfp_montage(z, n=n, sigma=0.11)[0]

    fin = np.isfinite(d_none)
    assert np.array_equal(fin, np.isfinite(d_expl))
    assert np.allclose(d_none[fin], d_expl[fin], rtol=0, atol=0), \
        "sigma=None must be byte-identical to lfp_2d's own default"
    # widening integrates ~18x more tissue, so amplitude must grow markedly
    ratio = np.nanstd(wide[fin]) / np.nanstd(d_none[fin])
    assert ratio > 3.0, "sigma=0.11 barely changed the montage (ratio %.2f)" % ratio


def test_field_worker_refuses_verify_against_at_a_nondefault_sigma():
    """Comparing a 0.11 montage to a 0.025 reference must raise, not 'mismatch'.

    Without this the abort message would blame a divergent realization for what
    is really the sigma change, sending the reader after a nonexistent
    determinism bug.
    """
    import las_model.pilot_worker as PW
    import inspect

    p = inspect.signature(PW.run_and_reduce_field).parameters
    assert "sigma" in p and p["sigma"].default is None

    try:
        PW.run_and_reduce_field({}, None, verify_against="pilot_out_v2",
                                sigma=0.11)
    except ValueError as e:
        assert "sigma" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for verify_against + sigma")


def test_field_summary_records_sigma_and_chloride_floor():
    """Every bundle must state its LFP sigma and its clamp count.

    Amplitude is meaningless without the sigma that produced it, and a clamped
    run left the model's valid range -- both were computed and then dropped
    before 2026-08-16.
    """
    import inspect
    import las_model.pilot_worker as PW

    src = inspect.getsource(PW.run_and_reduce_field)
    for key in ("lfp_sigma", "cl_floor_hits", "cl_floor_frac_units",
                "cl_floor_frac_steps"):
        assert key in src, "run_and_reduce_field summary lost %s" % key
    # the constant must be read from lfp_2d, not duplicated as a literal
    assert PW.LFP_DEFAULT_SIGMA == inspect.signature(
        __import__("las_model.lfp", fromlist=["lfp_2d"]).lfp_2d
    ).parameters["sigma"].default


def test_verify42_blinding_hides_run_identity_and_is_a_bijection():
    """V01..V42 must carry no hint of anchor status or parameter value.

    The 42 runs being scored include the anchors whose expected verdicts are
    under test, so a label like ``anchor_scale4_sim`` would reveal both that the
    run carries a prediction and which parameter was moved.  The mapping must
    also be a bijection and stable across rebuilds, or a re-render would produce
    a different key than the one already scored against.
    """
    import build_verify42_page as B

    keys = ["anchor_scale4_sim", "anchor_gE220_sim", "draw07_seq",
            "anchor_phi0_m52_seq", "draw13_sim"]
    m1 = B.blind_map(keys)
    m2 = B.blind_map(list(reversed(keys)))

    assert m1 == m2, "blind map must not depend on input order"
    assert len(set(m1.values())) == len(keys), "ids collided"
    assert set(m1) == set(keys)
    for v in m1.values():
        assert v.startswith("V") and v[1:].isdigit(), v
        for leak in ("anchor", "draw", "scale", "gE", "phi0", "sim", "seq"):
            assert leak not in v, (v, leak)


def test_verify42_movie_files_are_blinded():
    """Movie filenames must carry the blind id, never the run key.

    The page writes the movie path into <video src>, so a file named
    ``anchor_scale4_sim_overview.mp4`` would reveal both that the run carries a
    prediction under test and which parameter was moved -- visible in the pane
    and in the browser's network tab.  render_movies.overview() names by run
    key, which is why the verify42 driver has its own entry point.
    """
    import os
    import inspect
    import render_verify42_movies as RVM

    src = inspect.getsource(RVM.overview_blind)
    assert "% lab" in src, "output name must be built from the blind id"
    assert "key" not in src.split("os.path.join")[1].split(")")[0], \
        "output name must not interpolate the run key"

    if os.path.isdir(RVM.MOV_DIR):
        bad = [f for f in os.listdir(RVM.MOV_DIR)
               if f.endswith(".mp4") and not f.startswith("V")]
        assert not bad, "un-blinded movie filenames present: %s" % bad


def test_sheet_rule_reproduces_all_54_human_calls():
    """The committed sheet rule must match every human-scored run.

    54 runs, scored blind in two independent sessions (12 full-field + 42
    verification).  This is the regression test the pilot's montage rule never
    had: it existed only as a CSV column, so it could not be re-applied or
    audited.  If a future change to the duty metric or the thresholds breaks
    agreement, this fails with the offending runs named.
    """
    import csv
    import os
    from las_model import classify as CL

    path = os.path.join(os.path.dirname(__file__), "data", "human_scored_54.csv")
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    assert len(rows) == 54, len(rows)

    bad = []
    for r in rows:
        got, _margin = CL.sheet(float(r["unit_duty_median"]))
        if got != r["human_class"]:
            bad.append((r["run_key"], r["human_class"], got,
                        float(r["unit_duty_median"])))
    assert not bad, "sheet rule disagrees with human calls on: %s" % bad

    # both scoring sessions must be covered -- a fixture that lost one would
    # still pass the loop above while testing half of what it claims
    srcs = {r["source"] for r in rows}
    assert srcs == {"confirmed12", "verify42"}, srcs


def test_sheet_thresholds_sit_in_the_gaps_not_on_the_data():
    """Thresholds must be strictly inside the observed class gaps.

    A threshold placed ON an extreme value classifies that run by tie-breaking
    rather than by evidence, and would flip under a rounding change.  The lower
    gap is wide (0.0016 -> 0.0490); the UPPER one is only 1.28x
    (0.6750 -> 0.8673), which is why it is pinned here.
    """
    import csv
    import os
    from las_model import classify as CL

    path = os.path.join(os.path.dirname(__file__), "data", "human_scored_54.csv")
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    duty = {c: [float(r["unit_duty_median"]) for r in rows
                if r["human_class"] == c] for c in CL.CLASSES}

    assert max(duty["NOT_SEIZURE"]) < CL.DUTY_SEIZURE < min(duty["SEIZURE"])
    assert max(duty["SEIZURE"]) < CL.DUTY_TONIC < min(duty["TONIC_SATURATION"])


def test_auto2_still_reproduces_the_pilot_labels_after_being_committed():
    """Moving the pilot rule into a module must not change its 54 labels.

    It is retired for new work but is the provenance of every published pilot
    figure, so its outputs must stay byte-identical.
    """
    from las_model import classify as CL

    # (amp, cv, ipi, asym) -> expected label; one case per arm, taken from the
    # pilot runs that exercise each branch
    cases = [
        (928.2, 0.821, 745.0, 2.259, "QUIET_OR_BLOCK"),   # anchor_scale4_seq
        (14869.2, 1.987, 235.0, 1.118, "ICTAL"),           # anchor_scale4_sim
        (12078.3, 0.714, 755.0, 1.464, "ENTRAINED"),       # draw18_sim, 3x250
        (10819.8, 2.901, 190.0, 0.830, "ACTIVE_NONICTAL"),  # draw17_seq
    ]
    for amp, cv, ipi, asym, want in cases:
        assert CL.auto2(amp, cv, ipi, asym) == want, (amp, cv, ipi, asym)


def test_unresolved_flags_the_narrow_upper_boundary():
    """A run just under the tonic threshold must be flagged, not silently called.

    The seizure/tonic gap is 1.28x, so a duty of 0.78 is not distinguishable
    from tonic saturation on this evidence and must not be reported as a
    confident seizure.
    """
    from las_model import classify as CL

    assert CL.is_unresolved(0.78), CL.sheet(0.78)
    assert not CL.is_unresolved(0.30), CL.sheet(0.30)
    assert not CL.is_unresolved(0.0), CL.sheet(0.0)


def test_pass2_blinding_differs_from_pass1_for_every_run():
    """No run may keep its pass-1 blind id in the re-scoring page.

    The second pass exists to measure whether the SCORER's criteria drifted.  If
    a run kept its id, a remembered "V24 was a focal seizure" would make the
    second call partly a recall of the first, and the agreement number would
    overstate stability.  A different seed is not sufficient on its own -- a
    permutation can fix points -- so this checks every run.
    """
    import build_pass2_page as B
    import build_verify42_page as V

    keys = ["anchor_scale4_sim", "anchor_gE220_sim", "draw07_seq",
            "anchor_phi0_m52_seq", "draw13_sim", "draw16_seq", "draw04_seq",
            "anchor_default_seq", "draw09_sim", "draw11_sim"]
    p1, p2 = V.blind_map(keys), B.blind_map(keys)

    assert B.SHUFFLE_SEED != V.SHUFFLE_SEED
    fixed = [k for k in keys if p1[k] == p2[k]]
    assert not fixed, "blind id survived from pass 1 for: %s" % fixed
    assert len(set(p2.values())) == len(keys)
    for v in p2.values():
        assert v.startswith("P") and v[1:].isdigit(), v


def test_pass2_page_covers_every_run_at_one_sigma():
    """All 54 panels must come from sigma-0.11 bundles.

    Mixing the 12 runs' original 0.025 panels with 42 at 0.11 would confound a
    changed call with a changed instrument: at 0.025 summed electrode
    sensitivity varies 625x across the sheet, at 0.11 it is near-uniform, so the
    same run can look different without the scorer changing anything.
    """
    import json
    import os
    import numpy as np

    dirs = ("verify42_out", "confirmed12_out")
    if not all(os.path.isdir(d) for d in dirs):
        return                              # bundles not built in this checkout
    keys, sigmas = set(), set()
    for d in dirs:
        for f in os.listdir(d):
            if not f.endswith(".npz"):
                continue
            keys.add(os.path.splitext(f)[0])
            with np.load(os.path.join(d, f), allow_pickle=True) as z:
                sigmas.add(json.loads(str(z["summary"]))["lfp_sigma"])
    assert len(keys) == 54, len(keys)
    assert sigmas == {0.11}, sigmas


def test_tonic_saturation_option_makes_no_seizure_claim():
    """The tonic option must not assert seizure status either way.

    The user withdrew the "(not a seizure)" parenthetical on 2026-08-19 -- tonic
    saturation may be a form of seizure that cannot progress -- so the button
    text must not decide that for them.  Key POSITIONS must also be unchanged
    from pass 1: keys follow display order, so re-numbering an option between the
    two passes would put a keystroke confound inside the reliability measurement.
    """
    import build_fullfield_html as BFH

    assert "TONIC SATURATION" in BFH.OPTIONS
    assert "TONIC SATURATION (not a seizure)" not in BFH.OPTIONS
    assert BFH.OPTIONS == ["ENTRAINED, not a seizure", "TONIC SATURATION",
                           "SEIZURE: early slow tonic wavefront -> fast traveling waves",
                           "SEIZURE: fast traveling waves at onset",
                           "OTHER"], BFH.OPTIONS
    # the stored sheet verdict must still map onto the renamed option, or every
    # run of that class would score as a disagreement
    assert BFH.VERDICT_TO_OPTION["TONIC SATURATION (not a seizure)"] == "TONIC SATURATION"
    assert set(BFH.VERDICT_TO_OPTION.values()) <= set(BFH.OPTIONS)
    assert set(BFH.OPT_COLOUR) == set(BFH.OPTIONS)


def test_anchor_expectations_are_in_the_classifier_vocabulary():
    """Every anchor expectation must be a class classify.sheet can emit.

    Before 2026-08-19 they were ICTAL / BLOCK / NON_ICTAL / MARGINAL -- a retired
    vocabulary with no overlap with the classifier's output, so no comparison was
    possible.  That is exactly the failure this pins: an expectation the detector
    cannot produce is a check that can never pass, and it fails silently by never
    being evaluated at all.
    """
    import las_model.pilot_sweep as PS
    from las_model import classify as CL

    for tag, _ov, exp, _sub in PS.ANCHORS:
        vals = list(exp.values()) if isinstance(exp, dict) else [exp]
        for v in vals:
            assert v in CL.CLASSES, (tag, v, CL.CLASSES)

    # and after resolution, every manifest anchor row too
    for r in PS.manifest(2):
        if r["kind"] == "anchor":
            assert r["expected"] in CL.CLASSES, r


def test_anchor_scale4_expects_different_answers_per_configuration():
    """scale4 is the reason `expected` may be a dict.

    Its old single NON_ICTAL label was right for the sequential arm and wrong for
    the simultaneous one, which the user scored a seizure in both blind passes:
    at scale 4 the simultaneous arm delivers ~41,000 units of drive per step and
    the sequential arm almost none, so the two arms are not the same experiment.
    A regression that flattened this back to one label would silently reinstate a
    known-wrong expectation.
    """
    import las_model.pilot_sweep as PS
    import las_model.pilot_worker as PW

    man = {PW.run_key(r): r for r in PS.manifest(1)}
    assert man["anchor_scale4_seq"]["expected"] == "NOT_SEIZURE"
    assert man["anchor_scale4_sim"]["expected"] == "SEIZURE"
    # the resolver must not silently accept an unknown configuration key
    exp = dict(sequential="NOT_SEIZURE", simultaneous="SEIZURE")
    assert PS.expected_for(exp, True) == "NOT_SEIZURE"
    assert PS.expected_for(exp, False) == "SEIZURE"
    assert PS.expected_for("SEIZURE", True) == "SEIZURE"


def test_check_anchor_reports_disagreement_and_ignores_draws():
    """The comparison itself: fires on mismatch, silent on agreement and draws.

    Draws carry expected="" and must never trip the check, or every sweep would
    fail on its own random points.
    """
    import las_model.pilot_sweep as PS

    assert PS.check_anchor("a", "SEIZURE", "SEIZURE") is None
    assert PS.check_anchor("a", "", "SEIZURE") is None          # a draw
    assert PS.check_anchor("a", None, "SEIZURE") is None
    msg = PS.check_anchor("anchor_scale4_sim", "SEIZURE", "NOT_SEIZURE")
    assert msg and "anchor_scale4_sim" in msg and "SEIZURE" in msg


def test_every_anchor_matches_the_users_blind_score():
    """All 14 anchor runs must agree with the human calls they were set from.

    This is the check the anchors existed for.  It runs off the stored duty
    values rather than re-simulating, so it costs milliseconds; if a future
    change to the duty metric or the thresholds moves an anchor, this names it.
    """
    import csv
    import os
    import las_model.pilot_sweep as PS
    import las_model.pilot_worker as PW
    from las_model import classify as CL

    path = os.path.join(os.path.dirname(__file__), "data", "human_scored_54.csv")
    duty = {r["run_key"]: float(r["unit_duty_median"])
            for r in csv.DictReader(open(path, newline="", encoding="utf-8"))}
    man = {PW.run_key(r): r for r in PS.manifest(20)}
    anchors = {k: v for k, v in man.items() if v["kind"] == "anchor"}
    assert len(anchors) == 14, len(anchors)

    problems = []
    for k, v in sorted(anchors.items()):
        got, _margin = CL.sheet(duty[k])
        m = PS.check_anchor(k, v["expected"], got)
        if m:
            problems.append(m)
    assert not problems, problems


def test_sweep50_draws_0_to_19_are_the_pilot_draws():
    """The 50-draw sweep must reuse the pilot's first 20 draws exactly.

    draw(k) is seeded from 1000+k alone, independent of the draw count, which is
    what makes the 40 pilot draw runs reusable instead of re-simulated.  If a
    future change made draw() depend on n_draws, the sweep would silently mix
    two different parameter sets under the same run keys.
    """
    import run_sweep50 as R

    R.assert_pilot_is_a_subset()          # raises SystemExit on drift
    a, b = R.rows_by_key(20), R.rows_by_key(50)
    draws = [k for k, v in a.items() if v["kind"] == "draw"]
    assert len(draws) == 40, len(draws)
    for k in draws:
        for prm, _lo, _hi in R.PS.SPEC:
            assert a[k][prm] == b[k][prm], (k, prm)


def test_sweep50_admissibility_is_the_strict_chloride_rule():
    """A run is inadmissible iff ANY unit ever hit the floor.

    User decision 2026-08-19, chosen over a graded 0.005 threshold: a unit at the
    0.1 mM floor is held by a numerical guard at E_Cl = -187 mV rather than
    described by Liou Eq. 1, so the run did not test the equations.  A regression
    that relaxed this to a nonzero threshold would silently readmit runs the
    Methods section claims are excluded.
    """
    import run_sweep50 as R

    def summ(frac):
        return (None, dict(unit_duty_median=0.3, peak_simultaneous_frac=0.5,
                           cl_floor_frac_units=frac, cl_floor_frac_steps=0.0,
                           lfp_sigma=R.SIGMA))

    man = R.rows_by_key()
    k_seq = "draw00_seq"
    have = {k: summ(0.0) for k in man}
    have[k_seq] = summ(1.0 / 3136)                 # a single unit, one timestep
    rows = {r["run_key"]: r for r in R.label_all(have)}
    assert rows[k_seq]["admissible"] == 0, "one clamped unit must be enough"
    assert rows["draw00_sim"]["admissible"] == 1

    # and the PAIR is unusable when either arm is
    assert rows[k_seq]["pair_admissible"] == 0
    assert rows["draw00_sim"]["pair_admissible"] == 0, (
        "a pair with one inadmissible arm is not a pair")
    assert rows["draw01_seq"]["pair_admissible"] == 1


def test_sweep50_refuses_bundles_from_a_different_sigma():
    """A bundle stamped with another sigma must not join the sweep silently.

    Amplitude differs ~6x and the discharge count moves run-specifically between
    0.025 and 0.11, so pooling would mix two instruments in one table.
    """
    import json
    import os
    import numpy as np
    import run_sweep50 as R

    tmp = "_sigmatest_out"
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "draw00_seq.npz")
    np.savez_compressed(path, summary=json.dumps(
        dict(key="draw00_seq", unit_duty_median=0.1, peak_simultaneous_frac=0.1,
             cl_floor_frac_units=0.0, cl_floor_frac_steps=0.0, lfp_sigma=0.025)))
    old = R.REUSE_DIRS
    try:
        R.REUSE_DIRS = (tmp,)
        try:
            R.existing()
        except SystemExit as e:
            assert "lfp_sigma" in str(e), str(e)
        else:
            raise AssertionError("a 0.025 bundle was accepted into a 0.11 sweep")
    finally:
        R.REUSE_DIRS = old
        os.remove(path)
        os.rmdir(tmp)


def test_sweep50_anchor_gate_blocks_label_output():
    """An anchor mismatch must stop labels being written, not warn beside them.

    The anchors are the only runs whose answers are known independently, so a
    mismatch means every draw label in the batch is suspect.  A gate that merely
    printed would leave a plausible-looking labels file on disk.
    """
    import run_sweep50 as R

    man = R.rows_by_key()
    have = {k: (None, dict(unit_duty_median=0.3, peak_simultaneous_frac=0.5,
                           cl_floor_frac_units=0.0, cl_floor_frac_steps=0.0,
                           lfp_sigma=R.SIGMA)) for k in man}
    # duty 0.3 -> SEIZURE for every run, so the NOT_SEIZURE/TONIC anchors fail
    rows = R.label_all(have)
    fails = R.anchor_gate(rows)
    # 5 expect NOT_SEIZURE (gIlocal600 x2, gKmax120 x2, scale4_SEQ -- scale4_sim
    # expects SEIZURE, which is the whole point of its per-config entry) and 4
    # expect TONIC_SATURATION (gE220 x2, phi0_m52 x2).
    assert len(fails) == 9, len(fails)
    assert all("ANCHOR MISMATCH" in m for m in fails)

    # and with a correct detector the same gate is silent
    from las_model import classify as CL
    import csv
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "human_scored_54.csv")
    duty = {r["run_key"]: float(r["unit_duty_median"])
            for r in csv.DictReader(open(path, newline="", encoding="utf-8"))}
    have2 = {k: (None, dict(unit_duty_median=duty.get(k, 0.0),
                            peak_simultaneous_frac=0.0,
                            cl_floor_frac_units=0.0, cl_floor_frac_steps=0.0,
                            lfp_sigma=R.SIGMA)) for k in man}
    assert R.anchor_gate(R.label_all(have2)) == []


def test_pass3_blinding_hides_role_and_differs_from_pass2():
    """V3 ids must leak neither the run nor its ROLE on the page.

    Pass 3 mixes three kinds of run: class suspects, seizures needing a subtype,
    and six controls with stable prior calls.  A control that announced itself
    would stop being a control, and a suspect that announced itself would bias
    the scorer toward the class it is suspected of -- so role lives only in
    pass3_blind_key.csv, never in the id, the filenames, or the page.
    """
    import build_pass3_page as B3

    roles = B3.wanted()
    assert len(roles) == 32, len(roles)
    m = B3.blind_map(list(roles))
    assert len(set(m.values())) == 32

    for k, v in m.items():
        assert v.startswith("V3-") and v[3:].isdigit(), v
        for leak in ("anchor", "draw", "sim", "seq", "ctrl", "susp",
                     "control", "suspect"):
            assert leak not in v, (v, leak)

    # a different shuffle seed from both earlier passes
    assert B3.SHUFFLE_SEED not in (42, 20260819), B3.SHUFFLE_SEED

    # the union really is 26 + 6, with the five double-duty runs counted once
    from collections import Counter
    c = Counter(r["role"] for r in roles.values())
    assert c["control"] == 6, c
    assert c["suspect"] + c["subtype_needed"] == 26, c


def test_pass3_answers_both_questions_from_the_existing_options():
    """No option may be added for pass 3: group = class, item = subtype.

    The page settles the class question by which GROUP is picked and the subtype
    question by which ITEM inside the seizure group is picked.  If a future
    change flattened the groups, or added a subtype option outside the seizure
    group, one pick would stop answering both and the page would need splitting.
    """
    import build_fullfield_html as BFH

    groups = dict((lbl, opts) for lbl, opts in BFH.OPT_GROUPS if lbl)
    assert "seizure" in groups and "not a seizure" in groups, list(groups)

    sz = groups["seizure"]
    assert len(sz) == 2, sz
    assert all(o.startswith("SEIZURE:") for o in sz), sz
    # the two subtypes are the ONLY options claiming a seizure
    claim = [o for o in BFH.OPTIONS if o.startswith("SEIZURE:")]
    assert set(claim) == set(sz), (claim, sz)
    # and the non-seizure group makes no subtype claim
    for o in groups["not a seizure"]:
        assert "onset" not in o.lower() and "wavefront" not in o.lower(), o


if __name__ == "__main__":
    _main()
