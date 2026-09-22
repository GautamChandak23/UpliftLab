import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from upliftlab import FEATURES, ab, synthetic, uplift  # noqa: E402

DF = synthetic.make(120_000, seed=1)


def test_srm_flags_broken_split_only():
    assert ab.srm_test(85_100, 14_900, 0.85)["p_value"] > 0.01
    assert ab.srm_test(86_000, 14_000, 0.85)["p_value"] < 0.001


def test_randomised_features_are_balanced():
    assert ab.smd_table(DF)["smd"].abs().max() < 0.05


def test_ate_ci_covers_truth():
    truth = DF.attrs["true_cate_visit"].mean()
    r = ab.diff_in_means(DF["visit"], DF["treatment"])
    assert r["ci"][0] < truth < r["ci"][1]
    lo, hi = ab.poisson_bootstrap_ci(DF["visit"], DF["treatment"], B=100)
    assert lo < truth < hi


def test_cupac_reduces_variance_without_bias():
    c = ab.cupac(DF, "visit")
    assert c["cupac"]["se"] < c["raw"]["se"]
    assert abs(c["cupac"]["ate"] - DF.attrs["true_cate_visit"].mean()) < 3 * c["cupac"]["se"]


def test_learners_rank_users_better_than_random():
    X, t, y = DF[FEATURES].to_numpy(), DF["treatment"].to_numpy(), DF["visit"].to_numpy()
    tr = np.random.default_rng(0).random(len(DF)) < 0.7
    true = DF.attrs["true_cate_visit"][~tr]
    rnd = uplift.qini_coefficient(np.random.default_rng(1).random((~tr).sum()), y[~tr], t[~tr])
    for name in ["T-learner", "X-learner"]:
        u = uplift.fit_predict(name, X[tr], t[tr], y[tr], X[~tr])
        assert np.corrcoef(u, true)[0, 1] > 0.3
        assert uplift.qini_coefficient(u, y[~tr], t[~tr]) > rnd


def test_oracle_qini_beats_model_and_capture_is_share():
    t, y = DF["treatment"].to_numpy(), DF["visit"].to_numpy()
    true = DF.attrs["true_cate_visit"]
    assert uplift.qini_coefficient(true, y, t) > uplift.qini_coefficient(-true, y, t)
    assert np.isclose(uplift.capture_at(true, y, t, 1.0), 1.0)
