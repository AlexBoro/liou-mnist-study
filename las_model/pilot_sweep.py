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

# known-verdict anchors: (tag, overrides, expected_verdict)
ANCHORS = (
    ("anchor_default", {}, "ICTAL"),
    ("anchor_gIlocal120", {"g_I_local": 120.0}, "ICTAL"),
    ("anchor_gE220", {"g_E": 220.0}, "BLOCK"),
    ("anchor_phi0_m52", {"phi0": -52.0}, "BLOCK"),
    ("anchor_scale4", {"scale": 4.0}, "NON_ICTAL"),
    ("anchor_gIlocal600", {"g_I_local": 600.0}, "MARGINAL"),
    ("anchor_gKmax120", {"g_K_max": 120.0}, "NON_ICTAL"),
)

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
    for tag, ov, exp in ANCHORS:
        d = dict(BASE); d.update(ov)
        for seq in (False, True):
            rows.append(dict(tag=tag, kind="anchor", expected=exp,
                             sequential=seq, seed=0, **d))
    return rows
