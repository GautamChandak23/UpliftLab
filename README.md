# UpliftLab — who should actually get the ad?

The Criteo uplift dataset (~14M users, 12 anonymised features, 85% treated / 15% control,
`visit` and `conversion` outcomes) asks two questions: did the campaign work on average,
and *for whom*? Targeting everyone wastes budget on people who would have converted
anyway; uplift models rank users by the incremental effect of treating them.

```
spark/etl.py            PySpark: typed CSV read, duplicates, sample-ratio-mismatch test, arm outcome
                        rates, standardised mean differences per feature, parquet (full + 10% sample)
spark/tlearner_mllib.py T-learner with MLlib GBTs on the full data; decile/Qini table built in Spark
upliftlab/ab.py         SRM, SMD, difference in means, Poisson bootstrap CI, CUPAC variance reduction, MDE
upliftlab/uplift.py     S-, T-, X-learner, transformed-outcome; Qini curve/coefficient, top-k capture, deciles
scripts/run_analysis.py the in-memory analysis on the sample parquet (or --synthetic)
tests/                  6 tests on synthetic data with a known effect (CI coverage, unbiased CUPAC, ranking)
```

## Run

```bash
pip install -r requirements.txt          # needs Java 11+ for PySpark
python -m pytest tests
spark-submit spark/etl.py --input data/criteo-uplift-v2.1.csv.gz --out data
python scripts/run_analysis.py --parquet data/criteo_sample.parquet --outcome visit
python scripts/run_analysis.py --parquet data/criteo_sample.parquet --outcome conversion
spark-submit spark/tlearner_mllib.py --parquet data/criteo_full.parquet --outcome visit
```

The two Spark scripts were written without a Spark runtime available, so they have only
been syntax-checked. Run them on a 100k-row slice first, and fix anything that breaks
before you trust the full run. Everything under `upliftlab/` is tested.

## Things to be able to explain

* **SRM first.** If the treated share is not 85% within noise, the experiment is broken
  and no effect estimate is trustworthy.
* **CUPAC without a pre-period.** The covariate is an out-of-fold prediction from
  pre-treatment features, fitted without the treatment flag, so it cannot absorb the
  effect. The variance saving is roughly corr(Y, prediction)^2.
* **85/15 imbalance.** The T-learner's control model sees far less data, which is why the
  X-learner exists.
* **Evaluation.** Uplift has no per-user ground truth (you never see both outcomes for
  the same user), so models are compared with Qini curves on a held-out split, never
  with accuracy.
