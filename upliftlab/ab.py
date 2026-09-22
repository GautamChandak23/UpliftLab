"""Randomised-experiment checks and average treatment effect (ATE) estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from . import FEATURES

Z = stats.norm.ppf(0.975)


def srm_test(n_treat: int, n_control: int, expected_treat_share: float) -> dict:
    """Sample-ratio mismatch: chi-square test of the observed split vs the design split.
    A tiny p-value (< 0.001) means the assignment or logging is broken - stop and investigate."""
    n = n_treat + n_control
    exp = np.array([expected_treat_share, 1 - expected_treat_share]) * n
    chi2, p = stats.chisquare([n_treat, n_control], exp)
    return {"treat_share": n_treat / n, "chi2": float(chi2), "p_value": float(p)}


def smd_table(df: pd.DataFrame, features=FEATURES, t="treatment") -> pd.DataFrame:
    """Standardised mean difference per pre-treatment feature; |SMD| > 0.1 flags imbalance."""
    g = df.groupby(t)[list(features)]
    m, v = g.mean(), g.var()
    smd = (m.loc[1] - m.loc[0]) / np.sqrt((v.loc[1] + v.loc[0]) / 2)
    return pd.DataFrame({"mean_control": m.loc[0], "mean_treat": m.loc[1], "smd": smd})


def diff_in_means(y, t) -> dict:
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return {"control_rate": y0.mean(), "treat_rate": y1.mean(), "ate": ate, "se": se,
            "ci": (ate - Z * se, ate + Z * se), "relative_lift": ate / y0.mean()}


def poisson_bootstrap_ci(y, t, B=300, seed=0, max_n=2_000_000) -> tuple:
    """Poisson bootstrap (weights ~ Poisson(1)): streams well, the standard trick at scale."""
    rng = np.random.default_rng(seed)
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    if len(y) > max_n:
        idx = rng.choice(len(y), max_n, replace=False)
        y, t = y[idx], t[idx]
    est = []
    for _ in range(B):
        w = rng.poisson(1.0, len(y))
        est.append((w * y)[t].sum() / w[t].sum() - (w * y)[~t].sum() / w[~t].sum())
    return tuple(np.percentile(est, [2.5, 97.5]))


def cupac(df: pd.DataFrame, outcome: str, features=FEATURES, t="treatment", folds=3, seed=0) -> dict:
    """CUPED with an ML covariate (CUPAC).

    Criteo has no pre-experiment metric, so the covariate is an out-of-fold prediction of
    the outcome from the pre-treatment features, fitted WITHOUT the treatment flag. Because
    assignment is random, the covariate is independent of treatment and the adjusted
    estimator stays unbiased while its variance drops by roughly corr(Y, covariate)^2.
    """
    X, y = df[list(features)].to_numpy(), df[outcome].to_numpy(float)
    m = np.empty(len(y))
    for a, b in KFold(folds, shuffle=True, random_state=seed).split(X):
        m[b] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                             random_state=seed).fit(X[a], y[a]).predict(X[b])
    theta = np.cov(y, m)[0, 1] / m.var(ddof=1)
    y_adj = y - theta * (m - m.mean())
    raw, adj = diff_in_means(y, df[t]), diff_in_means(y_adj, df[t])
    adj["variance_reduction"] = 1 - (adj["se"] / raw["se"]) ** 2
    adj["ci_width_reduction"] = 1 - adj["se"] / raw["se"]
    adj["theta"] = theta
    return {"raw": raw, "cupac": adj}


def mde(base_rate: float, n_treat: int, n_control: int, alpha=0.05, power=0.8) -> float:
    """Minimum detectable absolute effect for a two-proportion test."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * np.sqrt(base_rate * (1 - base_rate) * (1 / n_treat + 1 / n_control))
