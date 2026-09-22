"""PySpark ETL + experiment health checks on the full Criteo uplift file (~14M rows).

    spark-submit spark/etl.py --input data/criteo-uplift-v2.1.csv.gz --out data/

Writes:
  data/criteo_full.parquet     all rows, partitioned by treatment
  data/criteo_sample.parquet   a stratified sample for the in-memory modelling scripts
  data/health.json             row counts, duplicates, SRM test, arm outcome rates, SMD table

Low-RAM laptops (8 GB): keep the driver at ~3 GB, e.g. on Windows
    set PYSPARK_SUBMIT_ARGS=--driver-memory 3g pyspark-shell
The cached table spills to disk instead of exhausting memory, and the duplicate check
shuffles a 64-bit row hash instead of all 16 columns.
"""
import argparse
import json
import os
import shutil

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F, types as T
from scipy import stats

FEATURES = [f"f{i}" for i in range(12)]
LABELS = ["treatment", "conversion", "visit", "exposure"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument("--design-treat-share", type=float, default=0.85)
    ap.add_argument("--partitions", type=int, default=32)
    ap.add_argument("--writer", choices=["spark", "arrow"], default="arrow" if os.name == "nt" else "spark",
                    help="arrow (default on Windows): write the sample with pyarrow, skipping Hadoop's "
                         "native Windows libraries; spark: normal Spark parquet writes (full + sample)")
    a = ap.parse_args()

    spark = (SparkSession.builder.appName("upliftlab-etl")
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    schema = T.StructType([T.StructField(c, T.DoubleType()) for c in FEATURES + LABELS])
    df = spark.read.csv(a.input, header=True, schema=schema)
    # a .gz file cannot be split, so Spark reads it in ONE task: spread the rows out first,
    # and cache with MEMORY_AND_DISK so a low-RAM machine spills to disk instead of crashing
    df = (df.select(*FEATURES, *[F.col(c).cast("int").alias(c) for c in LABELS])
            .repartition(a.partitions)
            .persist(StorageLevel.MEMORY_AND_DISK))

    n = df.count()
    # duplicates via a 64-bit hash of each row: 8 bytes/row in the shuffle instead of 16 columns
    # (chance of a false collision among ~14M rows is about 1e-5)
    n_distinct = df.select(F.xxhash64(*FEATURES, *LABELS).alias("h")).distinct().count()
    arms = (df.groupBy("treatment")
              .agg(F.count("*").alias("n"), F.avg("visit").alias("visit_rate"),
                   F.avg("conversion").alias("conversion_rate"), F.avg("exposure").alias("exposure_rate"))
              .toPandas().set_index("treatment"))
    nt, nc = int(arms.loc[1, "n"]), int(arms.loc[0, "n"])
    chi2, p = stats.chisquare([nt, nc], [a.design_treat_share * n, (1 - a.design_treat_share) * n])

    agg = [F.avg(c).alias(f"{c}__mean") for c in FEATURES] + [F.variance(c).alias(f"{c}__var") for c in FEATURES]
    bal = df.groupBy("treatment").agg(*agg).toPandas().set_index("treatment")
    smd = {c: float((bal.loc[1, f"{c}__mean"] - bal.loc[0, f"{c}__mean"])
                    / ((bal.loc[1, f"{c}__var"] + bal.loc[0, f"{c}__var"]) / 2) ** 0.5) for c in FEATURES}

    health = {"rows": n, "duplicate_rows": n - n_distinct,
              "srm": {"treat_share": nt / n, "chi2": float(chi2), "p_value": float(p)},
              "arms": arms.reset_index().to_dict(orient="records"), "smd": smd,
              "max_abs_smd": max(abs(v) for v in smd.values())}
    json.dump(health, open(f"{a.out}/health.json", "w"), indent=2, default=float)
    print(json.dumps(health, indent=2, default=float))

    sample = df.sampleBy("treatment", fractions={0: a.sample_frac, 1: a.sample_frac}, seed=42)
    if a.writer == "spark":
        df.write.mode("overwrite").partitionBy("treatment").parquet(f"{a.out}/criteo_full.parquet")
        sample.write.mode("overwrite").parquet(f"{a.out}/criteo_sample.parquet")
    else:
        # On Windows, Spark's file committer needs Hadoop's native hadoop.dll/winutils; without them
        # every Spark write fails (UnsatisfiedLinkError ... NativeIO$Windows.access0). The ~1.4M-row
        # sample is small enough to move to Python through Arrow and write with pyarrow instead.
        # The full 14M-row table is not written in this mode (only tlearner_mllib.py needs it).
        path = f"{a.out}/criteo_sample.parquet"
        if os.path.isdir(path):
            shutil.rmtree(path)
        pdf = sample.toPandas()
        pdf.to_parquet(path, index=False)
        print(f"wrote {len(pdf):,} sample rows to {path} with pyarrow")
    spark.stop()


if __name__ == "__main__":
    main()
