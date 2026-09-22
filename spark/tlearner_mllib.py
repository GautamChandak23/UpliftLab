"""T-learner with Spark MLlib GBTs on the FULL dataset + decile / Qini table.

    spark-submit spark/tlearner_mllib.py --parquet data/criteo_full.parquet --outcome visit

Trains one GBTClassifier on treated rows and one on control rows (70% split), scores the
30% holdout, buckets it into uplift deciles with approxQuantile + Bucketizer, and
aggregates per decile in Spark so nothing large is collected to the driver.
NOTE: developed without a Spark runtime - smoke-test on the sample parquet first.
"""
import argparse

import numpy as np
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import Bucketizer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

FEATURES = [f"f{i}" for i in range(12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outcome", default="visit")
    ap.add_argument("--max-iter", type=int, default=50)
    a = ap.parse_args()
    spark = SparkSession.builder.appName("upliftlab-tlearner").getOrCreate()
    df = VectorAssembler(inputCols=FEATURES, outputCol="features").transform(spark.read.parquet(a.parquet))
    train, test = df.randomSplit([0.7, 0.3], seed=42)

    def fit(arm):
        gbt = GBTClassifier(labelCol=a.outcome, featuresCol="features", maxIter=a.max_iter,
                            maxDepth=5, seed=42)
        return gbt.fit(train.filter(F.col("treatment") == arm))

    m1, m0 = fit(1), fit(0)
    drop = ["rawPrediction", "probability", "prediction"]
    s = m1.transform(test).withColumn("p1", vector_to_array("probability")[1]).drop(*drop)
    s = m0.transform(s).withColumn("p0", vector_to_array("probability")[1]).drop(*drop)
    s = s.withColumn("uplift", F.col("p1") - F.col("p0")).cache()

    cuts = sorted(set(s.approxQuantile("uplift", [i / 10 for i in range(1, 10)], 0.001)))
    s = Bucketizer(splits=[-float("inf")] + cuts + [float("inf")], inputCol="uplift",
                   outputCol="bucket").transform(s)
    nb = len(cuts) + 1
    tab = (s.groupBy("bucket")
             .agg(F.avg("uplift").alias("predicted"),
                  F.sum(F.col("treatment")).alias("n_t"), F.sum(1 - F.col("treatment")).alias("n_c"),
                  F.sum(F.col(a.outcome) * F.col("treatment")).alias("y_t"),
                  F.sum(F.col(a.outcome) * (1 - F.col("treatment"))).alias("y_c"))
             .toPandas())
    tab["decile"] = nb - tab["bucket"].astype(int)          # decile 1 = highest predicted uplift
    tab = tab.sort_values("decile")
    tab["observed"] = tab.y_t / tab.n_t - tab.y_c / tab.n_c
    ct, cc = tab.n_t.cumsum(), tab.n_c.cumsum()
    tab["qini"] = tab.y_t.cumsum() - tab.y_c.cumsum() * ct / cc
    tab["capture"] = tab["qini"] / tab["qini"].iloc[-1]
    print(tab[["decile", "predicted", "observed", "n_t", "n_c", "qini", "capture"]].to_string(index=False))
    print("top-30% capture of incremental", a.outcome, f"{np.interp(3, tab.decile, tab.capture):.1%}")
    spark.stop()


if __name__ == "__main__":
    main()
