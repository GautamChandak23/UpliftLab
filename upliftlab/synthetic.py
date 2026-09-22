"""Criteo-like synthetic data (same columns) with a KNOWN heterogeneous treatment effect.

Used for tests and to check that every estimator recovers the truth before it is trusted
on the real 14M-row file, where the truth is unknown.
"""
import numpy as np
import pandas as pd

from . import FEATURES


def make(n=200_000, treat_share=0.85, base_visit=0.04, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 12))
    t = (rng.random(n) < treat_share).astype(int)
    logit0 = np.log(base_visit / (1 - base_visit)) + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    tau_logit = 0.35 * np.maximum(X[:, 3], 0) - 0.1          # only users with f3 > 0.29 respond
    p0 = 1 / (1 + np.exp(-logit0))
    p1 = 1 / (1 + np.exp(-(logit0 + tau_logit)))
    visit = (rng.random(n) < np.where(t == 1, p1, p0)).astype(int)
    conversion = (visit & (rng.random(n) < 0.06 + 0.02 * (X[:, 4] > 0))).astype(int)
    exposure = (t & (rng.random(n) < 0.03)).astype(int)
    df = pd.DataFrame(X, columns=FEATURES)
    df["treatment"], df["conversion"], df["visit"], df["exposure"] = t, conversion, visit, exposure
    df.attrs["true_cate_visit"] = p1 - p0
    return df
