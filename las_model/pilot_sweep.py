"""Liou-MNIST pilot sweep driver.

54 runs = 20 paired random draws x {simultaneous, sequential}
        +  7 known-verdict anchors x {simultaneous, sequential}

Parameters reach the model by two routes and BOTH are needed:
  * params-object fields (tau_Cl, g_K_max, tau_K, Cl_in_eq, phi0, beta, tau_syn_I)
    are (n,n) arrays on a mutable dataclass -- set the whole array.
  * connectivity (g_E, g_I_local, g_I_global, sigma_E_frac, sigma_I_frac) goes
    through run_mnist's rec= dict.  NO monkeypatching required.
"""
import dataclasses as dc
import numpy as np

import las_model.params as LPAR
import las_model.run_mnist as RM

PARAM_FIELDS = ("tau_Cl", "g_K_max", "tau_K", "Cl_in_eq", "phi0", "beta", "tau_syn_I")
CONN_FIELDS = ("g_E", "g_I_local", "g_I_global", "sigma_E_frac", "sigma_I_frac")

SPEC = (
    ("tau_Cl", 1500.0, 20000.0),
    ("g_K_max", 10.0, 120.0),
    ("tau_K", 1500.0, 20000.0),
    ("Cl_in_eq", 3.0, 12.0),
    ("g_I_local", 120.0, 400.0),
    ("phi0", -48.0, -38.0),
    ("g_E", 60.0, 130.0),
    ("beta", 1.0, 5.0),
    ("tau_syn_I", 7.0, 30.0),
    ("sigma_I_frac", 0.020, 0.045),
    ("scale", 4.0, 37.0),
)

# Known-verdict anchors: (tag, overrides, expected).
#
# REWRITTEN 2026-08-19.  The previous expectations were in a retired vocabulary
# (ICTAL / BLOCK / NON_ICTAL / MARGINAL) that classify.sheet does not emit, so
# there was nothing they could be compared against -- and nothing did compare
# them: `expected` was written into every run summary and never checked.  The
# anchors therefore tested nothing.  check_anchor() below supplies the missing
# comparison, and these labels are now in the classifier's own three classes.
#
# SOURCE OF TRUTH: the user's blind scores, twice, on all seven anchors in both
# configurations (pass 1 and pass 2, kappa 0.95 overall, and identical on every
# anchor).  Not the original design intent -- two of the old labels were wrong:
#
#   anchor_scale4    said NON_ICTAL for both configs.  Correct for _seq
#                    (entrained) and WRONG for _sim (scored a seizure both
#                    times).  The spec's rationale, "weak drive, below
#                    nucleation", holds only for the sequential arm: at scale 4
#                    the simultaneous arm delivers ~41,000 units of |drive| per
#                    step where the sequential arm delivers almost none.  Hence
#                    the per-config mapping -- the reason `expected` may now be
#                    a dict as well as a string.
#   anchor_gIlocal600 said MARGINAL, a placeholder with no automated
#                    counterpart, so it could never pass or fail.  Scored
#                    entrained both times, so it is now a real check.
#
# The other five keep their meaning under new names: BLOCK -> TONIC_SATURATION
# (the user's term for what they scored), NON_ICTAL -> NOT_SEIZURE, ICTAL ->
# SEIZURE.
#
# `subtype` records the user's onset call where they scored a seizure.  Nothing
# checks it yet -- no committed metric separates the two subtypes -- so it is
# documentation for the onset measurement that will, and check_anchor() ignores
# it deliberately rather than by omission.
ANCHORS = (
    # The two configurations of the default anchor got DIFFERENT onset calls
    # (2026-08-29 corpus): _seq slow onset, _sim fast onset.  Recorded as a
    # config-keyed dict, resolved by expected_for(), rather than the single
    # "slow onset" that stood here -- that was true of _seq only and would have
    # sent the eventual onset metric chasing a disagreement that is really a
    # stale expectation.  This is the ONLY anchor whose subtype splits by
    # configuration; the class does not split for it.
    ("anchor_default", {}, "SEIZURE",
     {"sequential": "slow onset", "simultaneous": "fast onset"}),
    ("anchor_gIlocal120", {"g_I_local": 120.0}, "SEIZURE", "fast onset"),
    ("anchor_gE220", {"g_E": 220.0}, "TONIC_SATURATION", None),
    ("anchor_phi0_m52", {"phi0": -52.0}, "TONIC_SATURATION", None),
    ("anchor_scale4", {"scale": 4.0},
     {"sequential": "NOT_SEIZURE", "simultaneous": "SEIZURE"}, "slow onset"),
    ("anchor_gIlocal600", {"g_I_local": 600.0}, "NOT_SEIZURE", None),
    ("anchor_gKmax120", {"g_K_max": 120.0}, "NOT_SEIZURE", None),
)


def expected_for(exp, sequential):
    """Resolve an ANCHORS `expected` entry for one configuration.

    A plain string applies to both configurations; a dict keyed
    'sequential'/'simultaneous' gives them different expectations, which
    anchor_scale4 needs.
    """
    if isinstance(exp, str) or exp is None:
        return exp
    return exp["sequential" if sequential else "simultaneous"]


def check_anchor(key, expected, got):
    """Compare one anchor's computed class against its expectation.

    Returns None on agreement, else a message.  THIS IS THE CHECK THE ANCHORS
    EXISTED FOR AND NEVER HAD: without it a detector that is simply wrong
    produces a full sweep of plausible-looking labels, and the seven runs whose
    answers are known contribute nothing.  Callers should treat a non-None
    return as fatal to the sweep, not as a warning -- if the anchors disagree,
    every draw label in the same batch is suspect.
    """
    if not expected:                      # draws carry no expectation
        return None
    if got == expected:
        return None
    return ("ANCHOR MISMATCH %s: expected %s, classifier says %s. The anchors "
            "are the detector's own sanity check, so every draw in this batch "
            "is suspect until this is resolved -- fix the detector, or re-score "
            "this anchor and update ANCHORS with the new call."
            % (key, expected, got))

BASE = {
    "tau_Cl": 5000.0, "g_K_max": 40.0, "tau_K": 5000.0, "Cl_in_eq": 6.0,
    "phi0": -45.0, "beta": 2.5, "tau_syn_I": 15.0,
    "g_E": 100.0, "g_I_local": 250.0, "g_I_global": 50.0,
    "sigma_E_frac": 0.02, "sigma_I_frac": 0.03, "scale": 17.0,
}


def draw(k, n_draws_ignored=None):
    """Parameter draw k.  Seeded from 1000+k ONLY, so it is independent of the
    run seed and identical across configurations (paired design, as in WC)."""
    rng = np.random.default_rng(1000 + k)
    d = dict(BASE)
    for name, lo, hi in SPEC:
        d[name] = float(rng.uniform(lo, hi))
    d["g_I_global"] = float((50.0, 300.0)[k % 2])   # 2-level control arm
    return d


def build_params(n, d):
    """exp1_template with the Tier-1/2 params-object fields overridden.

    Fields are (n,n) arrays; assign the full array so no unit is left at the
    template value."""
    pp = LPAR.exp1_template((n, n))
    for name in PARAM_FIELDS:
        if name in d:
            cur = getattr(pp, name)
            setattr(pp, name, np.full_like(np.asarray(cur, dtype=float), float(d[name])))
    return pp


def run_one(images, d, *, sequential, n=56, T_end=80000.0, seed=0, rec_every=5,
            field_every=0):
    pp = build_params(n, d)
    rec = {k: float(d[k]) for k in CONN_FIELDS if k in d}
    return RM.run_mnist(
        images, scale=float(d["scale"]), T_end=T_end, n=n, projection=True,
        seed=seed, rec_every=rec_every, sequential=bool(sequential),
        params=pp, rec=rec, record_lfp_currents=True, field_every=field_every,
    )


def manifest(n_draws=20):
    """The full run list: draws first, then anchors, each in both configs."""
    rows = []
    for k in range(n_draws):
        d = draw(k)
        for seq in (False, True):
            rows.append(dict(tag="draw%02d" % k, kind="draw", expected="",
                             sequential=seq, seed=k, **d))
    for tag, ov, exp, subtype in ANCHORS:
        d = dict(BASE); d.update(ov)
        for seq in (False, True):
            # expected_for resolves the per-config case (anchor_scale4); every
            # other anchor gets the same string in both configurations.
            rows.append(dict(tag=tag, kind="anchor",
                             expected=expected_for(exp, seq),
                             expected_subtype=subtype,
                             sequential=seq, seed=0, **d))
    return rows
