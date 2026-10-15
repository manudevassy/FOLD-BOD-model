"""
fold_bod.py — Fractional-Order Logistic Decay (FOLD) model for BOD
==================================================================

Fit the FOLD and classical first-order BOD models to any BOD time series and
compute basic wastewater-treatment design outputs.

Reference:
    Devassy, M. (2026). Improved BOD Modelling Using the FOLD Approach for
    Wastewater Treatment Design. Water and Environment Journal.
    https://doi.org/10.1111/wej.70098

Quick use:
    >>> from fold_bod import fit_fold, fold
    >>> t = [1, 2, 3, 4, 5]
    >>> y = [211, 322, 383, 423, 450]
    >>> L0, k, alpha, gamma = fit_fold(t, y)
"""
import numpy as np
from scipy.optimize import least_squares

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def first_order(t, L0, k):
    """Classical first-order BOD:  BOD(t) = L0 [1 - exp(-k t)]."""
    t = np.asarray(t, float)
    return L0 * (1.0 - np.exp(-k * t))


def fold(t, L0, k, alpha, gamma):
    """FOLD model:  BOD(t) = L0 [1 - (1 + k t^alpha / gamma)^(-gamma)].

    L0    ultimate BOD (mg/L)
    k     rate parameter (units depend on alpha)
    alpha fractional-order exponent, 0 < alpha <= 1
    gamma saturation parameter, gamma > 0
    Reduces to first-order when alpha = 1 and gamma -> infinity,
    and to second-order when alpha = 1 and gamma = 1.
    """
    t = np.asarray(t, float)
    return L0 * (1.0 - (1.0 + k * np.power(t, alpha) / gamma) ** (-gamma))


# ---------------------------------------------------------------------------
# Fitting settings — change these if your data need different limits
# ---------------------------------------------------------------------------
SETTINGS = {
    "FO_L0_bounds": (1.0, 5000.0),
    "FO_k_bounds": (1e-4, 5.0),
    "FOLD_L0_upper_factor": 5.0 / 3.0,   # L0 between max(BOD) and this x max(BOD)
    "FOLD_k_bounds": (1e-5, 10.0),
    "FOLD_alpha_bounds": (0.05, 1.0),
    "FOLD_gamma_bounds": (0.01, 10.0),
}


def _check(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.shape != y.shape:
        raise ValueError("time and BOD must have the same number of values")
    if len(t) < 3:
        raise ValueError("at least 3 data points are needed")
    if np.any(t <= 0):
        raise ValueError("times must be greater than zero (do not include t = 0)")
    return t, y


def fit_first_order(t, y):
    """Fit the first-order model. Returns (L0, k)."""
    t, y = _check(t, y)
    lo = [SETTINGS["FO_L0_bounds"][0], SETTINGS["FO_k_bounds"][0]]
    hi = [SETTINGS["FO_L0_bounds"][1], SETTINGS["FO_k_bounds"][1]]
    best = None
    for L0g in (1.0, 1.2, 1.5):
        for kg in (0.1, 0.3, 0.6):
            x0 = np.clip([y.max() * L0g, kg], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            r = least_squares(lambda p: y - first_order(t, *p), x0, bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
    return tuple(best.x)


def fit_fold(t, y, fixed=None):
    """Fit the FOLD model by multi-start bounded least squares.

    fixed : optional dict, e.g. {"alpha": 1.0}, to hold a parameter constant.
    Returns (L0, k, alpha, gamma).
    """
    t, y = _check(t, y)
    fixed = fixed or {}
    names = ["L0", "k", "alpha", "gamma"]
    lo_all = [y.max(), SETTINGS["FOLD_k_bounds"][0],
              SETTINGS["FOLD_alpha_bounds"][0], SETTINGS["FOLD_gamma_bounds"][0]]
    hi_all = [y.max() * SETTINGS["FOLD_L0_upper_factor"], SETTINGS["FOLD_k_bounds"][1],
              SETTINGS["FOLD_alpha_bounds"][1], SETTINGS["FOLD_gamma_bounds"][1]]
    free = [i for i, n in enumerate(names) if n not in fixed]
    lo = np.array([lo_all[i] for i in free])
    hi = np.array([hi_all[i] for i in free])

    def full(p_free):
        p = [None] * 4
        for j, i in enumerate(free):
            p[i] = p_free[j]
        for n, v in fixed.items():
            p[names.index(n)] = v
        return p

    best = None
    for L0g in (1.02, 1.3, 1.6):
        for kg in (0.05, 0.2, 0.5, 1.0):
            for ag in (0.4, 0.7, 1.0):
                for gg in (0.5, 2.0, 10.0):
                    start = [y.max() * L0g, kg, ag, gg]
                    x0 = np.clip([start[i] for i in free], lo + 1e-9, hi - 1e-9)
                    r = least_squares(lambda p: y - fold(t, *full(p)), x0, bounds=(lo, hi))
                    if best is None or r.cost < best.cost:
                        best = r
    return tuple(full(best.x))


def at_bound(t, y, params):
    """Names of FOLD parameters that ended on a fitting limit (less reliable)."""
    t, y = _check(t, y)
    L0, k, a, g = params
    flags = []
    if abs(L0 - y.max() * SETTINGS["FOLD_L0_upper_factor"]) < 1e-3 * L0:
        flags.append("L0 (upper limit)")
    if abs(g - SETTINGS["FOLD_gamma_bounds"][1]) < 1e-3:
        flags.append("gamma (upper limit)")
    if abs(a - SETTINGS["FOLD_alpha_bounds"][0]) < 1e-3:
        flags.append("alpha (lower limit)")
    return flags


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------


def goodness_of_fit(y_obs, y_pred, n_params):
    """Return dict with R2, RMSE, AIC and BIC."""
    y_obs = np.asarray(y_obs, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_obs)
    rss = float(np.sum((y_obs - y_pred) ** 2))
    r2 = 1.0 - rss / float(np.sum((y_obs - y_obs.mean()) ** 2))
    rss_n = max(rss / n, 1e-12)
    return {"R2": r2, "RMSE": np.sqrt(rss / n),
            "AIC": n * np.log(rss_n) + 2 * n_params,
            "BIC": n * np.log(rss_n) + n_params * np.log(n)}


# ---------------------------------------------------------------------------
# Design (Water and Environment Journal article, Section 2.5)
# ---------------------------------------------------------------------------
DESIGN = {
    "Q": 35000.0,           # flow (m3/d)
    "S_eff": 20.0,          # target effluent BOD (mg/L)
    "loading_rate": 500.0,  # BOD loading for activated sludge (g BOD/m3/d)
    "O2_per_BOD": 1.5,      # kg O2 per kg BOD
    "O2_per_HPday": 44.0,   # kg O2 per HP per day
}


def design_activated_sludge(L0, k, d=None):
    """Activated sludge reactor: retention time (d), volume (m3), blower HP."""
    d = {**DESIGN, **(d or {})}
    t_as = np.log(L0 / d["S_eff"]) / k
    v = max(d["Q"] * L0 / d["loading_rate"], d["Q"] * t_as)
    hp = d["O2_per_BOD"] * d["Q"] * L0 / 1000.0 / d["O2_per_HPday"]
    return {"t_days": t_as, "V_m3": v, "HP": hp}


def design_aerated_lagoon(L0, k, d=None):
    """Aerated lagoon: retention time (d), volume (m3), blower HP."""
    d = {**DESIGN, **(d or {})}
    t_al = (L0 - d["S_eff"]) / (d["S_eff"] * k)
    hp = d["O2_per_BOD"] * d["Q"] * L0 / 1000.0 / d["O2_per_HPday"]
    return {"t_days": t_al, "V_m3": t_al * d["Q"], "HP": hp}
