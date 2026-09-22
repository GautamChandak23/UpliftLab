"""In-memory analysis on a sample (or synthetic data): health checks, ATE with bootstrap and
CUPAC, then four uplift learners compared on a held-out split.

python scripts/run_analysis.py --synthetic
python scripts/run_analysis.py --parquet data/criteo_sample.parquet --outcome visit
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from upliftlab import FEATURES, ab, synthetic, uplift  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--parquet")
    ap.add_argument("--outcome", default="visit")
    ap.add_argument("--n", type=int, default=300_000, help="synthetic rows")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = synthetic.make(a.n) if a.synthetic else pd.read_parquet(a.parquet)
    y, t = df[a.outcome].to_numpy(), df["treatment"].to_numpy()
    rep = {"rows": len(df)}

    rep["srm"] = ab.srm_test(int(t.sum()), int((1 - t).sum()), 0.85)
    rep["max_abs_smd"] = float(ab.smd_table(df)["smd"].abs().max())
    c = ab.cupac(df, a.outcome)
    rep["ate_raw"] = {k: c["raw"][k] for k in ("control_rate", "treat_rate", "ate", "se", "relative_lift")}
    rep["ate_bootstrap_ci"] = ab.poisson_bootstrap_ci(y, t)
    rep["ate_cupac"] = {k: c["cupac"][k] for k in ("ate", "se", "variance_reduction", "ci_width_reduction")}
    rep["mde_abs"] = ab.mde(c["raw"]["control_rate"], int(t.sum()), int((1 - t).sum()))

    rng = np.random.default_rng(0)
    tr = rng.random(len(df)) < 0.7
    X = df[FEATURES].to_numpy()
    res = {}
    for name in uplift.MODELS:
        u = uplift.fit_predict(name, X[tr], t[tr], y[tr], X[~tr])
        res[name] = {"qini_coef": uplift.qini_coefficient(u, y[~tr], t[~tr]),
                     "top30_capture": uplift.capture_at(u, y[~tr], t[~tr], 0.3)}
        if a.synthetic:
            res[name]["corr_with_true_cate"] = float(np.corrcoef(u, df.attrs["true_cate_visit"][~tr])[0, 1])
        if name == "T-learner":
            uplift.decile_table(u, y[~tr], t[~tr]).to_csv(os.path.join(a.out, "deciles_tlearner.csv"))
    res["random"] = {"qini_coef": uplift.qini_coefficient(rng.random((~tr).sum()), y[~tr], t[~tr])}
    rep["uplift_models"] = res
    print(json.dumps(rep, indent=2, default=float))
    json.dump(rep, open(os.path.join(a.out, "analysis.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
