"""Uplift (CATE) models and their evaluation.

S-learner : one model on [X, T]; uplift = f(X, 1) - f(X, 0)
T-learner : separate models for treated and control; uplift = f1(X) - f0(X)
X-learner : imputes individual effects from the other arm's model, fits them, and blends
            with the propensity (Kunzel et al., 2019) - helps with the 85/15 imbalance
TO        : transformed-outcome regression, Z = Y (T - e) / (e (1 - e)), E[Z | X] = CATE
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _clf(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                          min_samples_leaf=200, random_state=seed)


def _reg(seed):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                         min_samples_leaf=200, random_state=seed)


def fit_predict(name: str, X, t, y, X_new, seed=0) -> np.ndarray:
    t = np.asarray(t).astype(int)
    if name == "S-learner":
        m = _clf(seed).fit(np.column_stack([X, t]), y)
        one, zero = np.ones(len(X_new)), np.zeros(len(X_new))
        return (m.predict_proba(np.column_stack([X_new, one]))[:, 1]
                - m.predict_proba(np.column_stack([X_new, zero]))[:, 1])
    if name == "T-learner":
        m1, m0 = _clf(seed).fit(X[t == 1], y[t == 1]), _clf(seed).fit(X[t == 0], y[t == 0])
        return m1.predict_proba(X_new)[:, 1] - m0.predict_proba(X_new)[:, 1]
    if name == "X-learner":
        m1, m0 = _clf(seed).fit(X[t == 1], y[t == 1]), _clf(seed).fit(X[t == 0], y[t == 0])
        d1 = y[t == 1] - m0.predict_proba(X[t == 1])[:, 1]
        d0 = m1.predict_proba(X[t == 0])[:, 1] - y[t == 0]
        g1, g0 = _reg(seed).fit(X[t == 1], d1), _reg(seed).fit(X[t == 0], d0)
        e = t.mean()                                     # randomised: constant propensity
        return e * g0.predict(X_new) + (1 - e) * g1.predict(X_new)
    if name == "Transformed outcome":
        e = t.mean()
        z = y * (t - e) / (e * (1 - e))
        return _reg(seed).fit(X, z).predict(X_new)
    raise ValueError(name)


MODELS = ["S-learner", "T-learner", "X-learner", "Transformed outcome"]


def qini_curve(uplift, y, t, n_points=100) -> pd.DataFrame:
    """Qini curve: incremental outcomes when targeting the top-k by predicted uplift,
    Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)."""
    order = np.argsort(-np.asarray(uplift))
    y, t = np.asarray(y, float)[order], np.asarray(t).astype(bool)[order]
    nt, nc = np.cumsum(t), np.cumsum(~t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * ~t)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points + 1).astype(int))[1:]
    q = yt[idx] - yc[idx] * nt[idx] / np.maximum(nc[idx], 1)
    frac = (idx + 1) / len(y)
    return pd.DataFrame({"frac": frac, "qini": q, "random": frac * q[-1]})


def qini_coefficient(uplift, y, t) -> float:
    """Area between the Qini curve and the random-targeting line, per treated user."""
    c = qini_curve(uplift, y, t)
    trap = getattr(np, "trapezoid", None) or np.trapz          # numpy >= 2 / < 2
    return float(trap(c["qini"] - c["random"], c["frac"]) / max(np.asarray(t).sum(), 1))


def capture_at(uplift, y, t, top=0.3) -> float:
    """Share of all incremental outcomes captured by treating only the top `top` fraction."""
    c = qini_curve(uplift, y, t, n_points=1000)
    return float(np.interp(top, c["frac"], c["qini"]) / c["qini"].iloc[-1])


def decile_table(uplift, y, t) -> pd.DataFrame:
    d = pd.DataFrame({"u": uplift, "y": y, "t": np.asarray(t).astype(int)})
    d["decile"] = pd.qcut(-d["u"].rank(method="first"), 10, labels=range(1, 11))
    g = d.groupby("decile", observed=True)
    return pd.DataFrame({"predicted": g["u"].mean(),
                         "observed": g.apply(lambda x: x.loc[x.t == 1, "y"].mean() - x.loc[x.t == 0, "y"].mean(),
                                             include_groups=False),
                         "n": g.size()})
