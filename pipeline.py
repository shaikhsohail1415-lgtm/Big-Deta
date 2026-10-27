"""
E-commerce Sales Analytics & Customer Segmentation  --  PySpark pipeline

Stages
  1. Ingest      : read raw CSV with an explicit schema
  2. Clean       : drop guest checkouts, cancellations, bad quantities / prices
  3. Analyse     : Spark SQL aggregations (revenue trends, products, countries, time-of-day)
  4. Features    : RFM (Recency, Frequency, Monetary) per customer
  5. Model       : K-Means (Spark MLlib) with silhouette / elbow analysis
  6. Basket      : FP-Growth association rules (frequently bought together)
  7. Export      : small aggregated tables -> CSV files in this folder (read by the Streamlit app)

Usage
    python pipeline.py --input sample_online_retail.csv --output . --k 4
"""
import argparse
import os
import time

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (DoubleType, IntegerType, StringType, StructField,
                               StructType)

SCHEMA = StructType([
    StructField("InvoiceNo", StringType(), True),
    StructField("StockCode", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("Quantity", IntegerType(), True),
    StructField("InvoiceDate", StringType(), True),
    StructField("UnitPrice", DoubleType(), True),
    StructField("CustomerID", IntegerType(), True),
    StructField("Country", StringType(), True),
])

SEGMENT_NAMES = {
    3: ["Champions", "At-Risk", "Lost / Hibernating"],
    4: ["Champions", "Loyal Customers", "At-Risk", "Lost / Hibernating"],
    5: ["Champions", "Loyal Customers", "Potential Loyalists", "At-Risk", "Lost / Hibernating"],
}


def get_spark(app_name="EcommerceSegmentation"):
    spark = (SparkSession.builder
             .appName(app_name)
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1-2. Ingest & clean
# --------------------------------------------------------------------------- #
def load_and_clean(spark, path):
    raw = spark.read.csv(path, header=True, schema=SCHEMA)
    raw = raw.withColumn("InvoiceDate", F.to_timestamp("InvoiceDate", "yyyy-MM-dd HH:mm:ss"))
    raw_rows = raw.count()

    clean = (raw
             .dropna(subset=["CustomerID", "InvoiceDate"])              # guest checkouts
             .filter(~F.col("InvoiceNo").startswith("C"))               # cancellations
             .filter((F.col("Quantity") > 0) & (F.col("UnitPrice") > 0))
             .dropDuplicates()
             .withColumn("TotalPrice", F.round(F.col("Quantity") * F.col("UnitPrice"), 2))
             .cache())
    clean_rows = clean.count()
    log(f"Rows: raw={raw_rows:,}  clean={clean_rows:,}  removed={raw_rows - clean_rows:,}")
    return clean, raw_rows, clean_rows


# --------------------------------------------------------------------------- #
# 3. Exploratory analysis with Spark SQL
# --------------------------------------------------------------------------- #
def sales_analysis(spark, clean, raw_rows, clean_rows, out):
    clean.createOrReplaceTempView("sales")

    kpis = spark.sql("""
        SELECT ROUND(SUM(TotalPrice), 2)            AS total_revenue,
               COUNT(DISTINCT InvoiceNo)            AS total_orders,
               COUNT(DISTINCT CustomerID)           AS total_customers,
               COUNT(DISTINCT StockCode)            AS total_products,
               ROUND(SUM(TotalPrice) / COUNT(DISTINCT InvoiceNo), 2) AS avg_order_value,
               MIN(InvoiceDate)                     AS first_date,
               MAX(InvoiceDate)                     AS last_date
        FROM sales
    """).toPandas()
    kpis["raw_rows"] = raw_rows
    kpis["clean_rows"] = clean_rows
    kpis.to_csv(f"{out}/kpis.csv", index=False)

    spark.sql("""
        SELECT date_format(InvoiceDate, 'yyyy-MM')   AS month,
               ROUND(SUM(TotalPrice), 2)             AS revenue,
               COUNT(DISTINCT InvoiceNo)             AS orders,
               COUNT(DISTINCT CustomerID)            AS customers
        FROM sales GROUP BY 1 ORDER BY 1
    """).toPandas().to_csv(f"{out}/monthly_revenue.csv", index=False)

    spark.sql("""
        SELECT Description AS product,
               SUM(Quantity)               AS units_sold,
               ROUND(SUM(TotalPrice), 2)   AS revenue
        FROM sales GROUP BY Description ORDER BY revenue DESC LIMIT 50
    """).toPandas().to_csv(f"{out}/top_products.csv", index=False)

    spark.sql("""
        SELECT Country AS country,
               ROUND(SUM(TotalPrice), 2)   AS revenue,
               COUNT(DISTINCT CustomerID)  AS customers,
               COUNT(DISTINCT InvoiceNo)   AS orders
        FROM sales GROUP BY Country ORDER BY revenue DESC
    """).toPandas().to_csv(f"{out}/country_revenue.csv", index=False)

    spark.sql("""
        SELECT hour(InvoiceDate) AS hour,
               date_format(InvoiceDate, 'E') AS weekday,
               dayofweek(InvoiceDate)        AS weekday_num,
               ROUND(SUM(TotalPrice), 2)     AS revenue,
               COUNT(DISTINCT InvoiceNo)     AS orders
        FROM sales GROUP BY 1, 2, 3 ORDER BY 3, 1
    """).toPandas().to_csv(f"{out}/hourly_sales.csv", index=False)
    log("Sales analysis tables written")


# --------------------------------------------------------------------------- #
# 4. RFM features
# --------------------------------------------------------------------------- #
def build_rfm(clean):
    snapshot = clean.agg(F.max("InvoiceDate")).collect()[0][0]
    snapshot_lit = F.lit(snapshot) + F.expr("INTERVAL 1 DAY")
    rfm = (clean.groupBy("CustomerID")
           .agg(F.datediff(snapshot_lit, F.max("InvoiceDate")).alias("Recency"),
                F.countDistinct("InvoiceNo").alias("Frequency"),
                F.round(F.sum("TotalPrice"), 2).alias("Monetary"),
                F.first("Country").alias("Country")))
    return rfm.cache()


# --------------------------------------------------------------------------- #
# 5. K-Means
# --------------------------------------------------------------------------- #
def prepare_features(rfm):
    """Log-transform the skewed RFM values, then standardise."""
    logged = (rfm.withColumn("R_log", F.log1p("Recency"))
                 .withColumn("F_log", F.log1p("Frequency"))
                 .withColumn("M_log", F.log1p("Monetary")))
    assembler = VectorAssembler(inputCols=["R_log", "F_log", "M_log"], outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    return Pipeline(stages=[assembler, scaler]).fit(logged).transform(logged).cache()


def select_k(data, k_min=2, k_max=8, seed=42):
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(k=k, seed=seed, featuresCol="features", maxIter=50).fit(data)
        preds = model.transform(data)
        rows.append({"k": k,
                     "silhouette": round(evaluator.evaluate(preds), 4),
                     "wssse": round(model.summary.trainingCost, 2)})
        log(f"  k={k}  silhouette={rows[-1]['silhouette']}  wssse={rows[-1]['wssse']}")
    return pd.DataFrame(rows)


def name_clusters(profile, k):
    """Rank clusters by R (low = good), F (high = good), M (high = good) and label them."""
    p = profile.copy()
    p["score"] = (p["avg_recency"].rank(ascending=True)
                  + p["avg_frequency"].rank(ascending=False)
                  + p["avg_monetary"].rank(ascending=False))
    p = p.sort_values(["score", "avg_monetary"], ascending=[True, False]).reset_index(drop=True)
    names = SEGMENT_NAMES.get(k, [f"Segment {i + 1}" for i in range(k)])
    return {int(row.cluster): names[i] for i, row in p.iterrows()}


def cluster_customers(data, k, seed=42):
    model = KMeans(k=k, seed=seed, featuresCol="features", maxIter=100).fit(data)
    preds = model.transform(data).withColumnRenamed("prediction", "cluster")
    return preds


# --------------------------------------------------------------------------- #
# 6. Market-basket analysis
# --------------------------------------------------------------------------- #
def basket_analysis(clean, out, min_support=0.005, min_conf=0.3):
    baskets = (clean.groupBy("InvoiceNo")
               .agg(F.collect_set("Description").alias("items"))
               .filter(F.size("items") >= 2))
    n_baskets = baskets.count()
    model = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_conf).fit(baskets)

    rules = model.associationRules.toPandas()
    if rules.empty:
        pd.DataFrame(columns=["antecedent", "consequent", "confidence", "lift", "support"]) \
            .to_csv(f"{out}/association_rules.csv", index=False)
        log("No association rules found - try lowering min_support")
        return
    rules["antecedent"] = rules["antecedent"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequent"] = rules["consequent"].apply(lambda x: " + ".join(sorted(x)))
    rules = rules.sort_values("lift", ascending=False).head(100).round(4)
    rules.to_csv(f"{out}/association_rules.csv", index=False)
    log(f"FP-Growth: {n_baskets:,} baskets -> {len(rules)} rules saved")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()
    spark = get_spark()
    log(f"Spark {spark.version} started")

    clean, raw_rows, clean_rows = load_and_clean(spark, args.input)
    sales_analysis(spark, clean, raw_rows, clean_rows, args.output)

    rfm = build_rfm(clean)
    log(f"RFM table: {rfm.count():,} customers")
    data = prepare_features(rfm)

    log("Evaluating k ...")
    k_df = select_k(data, 2, args.k_max)
    k_df.to_csv(f"{args.output}/k_selection.csv", index=False)

    log(f"Fitting final K-Means with k={args.k}")
    preds = cluster_customers(data, args.k)
    pdf = preds.select("CustomerID", "Country", "Recency", "Frequency", "Monetary", "cluster").toPandas()

    profile = (pdf.groupby("cluster")
               .agg(customers=("CustomerID", "count"),
                    avg_recency=("Recency", "mean"),
                    avg_frequency=("Frequency", "mean"),
                    avg_monetary=("Monetary", "mean"),
                    total_revenue=("Monetary", "sum"))
               .reset_index())
    mapping = name_clusters(profile, args.k)
    pdf["segment"] = pdf["cluster"].map(mapping)
    profile["segment"] = profile["cluster"].map(mapping)
    profile["revenue_share_pct"] = (profile["total_revenue"] / profile["total_revenue"].sum() * 100)
    profile = profile.round(2).sort_values("avg_monetary", ascending=False)

    pdf.round(2).to_csv(f"{args.output}/rfm_segments.csv", index=False)
    profile.to_csv(f"{args.output}/cluster_profile.csv", index=False)
    log("Segments:\n" + profile[["segment", "customers", "avg_recency", "avg_frequency",
                                  "avg_monetary", "revenue_share_pct"]].to_string(index=False))

    if not args.skip_basket:
        basket_analysis(clean, args.output)

    if args.save_parquet:
        clean.write.mode("overwrite").parquet("clean_sales.parquet")
        log("Cleaned data saved as Parquet -> clean_sales.parquet")

    log(f"Done in {time.time() - t0:.1f}s. CSV outputs written to '{args.output}'")
    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PySpark e-commerce segmentation pipeline")
    ap.add_argument("--input", default="sample_online_retail.csv")
    ap.add_argument("--output", default=".", help="folder for output CSVs (default: current folder)")
    ap.add_argument("--k", type=int, default=4, help="number of clusters for the final model")
    ap.add_argument("--k_max", type=int, default=8, help="max k tried in silhouette/elbow search")
    ap.add_argument("--skip_basket", action="store_true", help="skip FP-Growth step")
    ap.add_argument("--save_parquet", action="store_true", help="also save cleaned data as Parquet")
    main(ap.parse_args())
