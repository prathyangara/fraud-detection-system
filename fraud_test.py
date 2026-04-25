import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lag, unix_timestamp, sum, avg, input_file_name
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("BatchFraudPipeline") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 55)
print("   BATCH MODE — FRAUD DETECTION PIPELINE")
print("=" * 55)

# ════════════════════════════════
# BATCH: All CSV files at once
# ════════════════════════════════
batch_files = ["january.csv", "february.csv", "march.csv"]

print(f"\n[1] BATCH EXTRACT — Loading {len(batch_files)} files...")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(batch_files) \
    .withColumn("source_file", input_file_name())

# Clean filename — just show month name
from pyspark.sql.functions import regexp_extract
df = df.withColumn("month",
    regexp_extract(col("source_file"), r"(\w+)\.csv", 1)
)

total = df.count()
print(f"    Total records loaded: {total}")

print("\n Per-file count:")
df.groupBy("month").count().orderBy("month").show()

# ════════════════════════════════
# TRANSFORM
# ════════════════════════════════
print("[2] TRANSFORM — Applying fraud rules to all batches...")

df = df.withColumn("txn_time", col("txn_time").cast("timestamp"))
df = df.withColumn("hour", col("txn_time").cast("string").substr(12, 2).cast("int"))
df = df.withColumn("txn_epoch", unix_timestamp(col("txn_time")))

# Rule 1: High Amount
df = df.withColumn("rule1_high_amount",
    when(col("amount") > 100000, "HIGH AMOUNT").otherwise("OK")
)

# Rule 2: Midnight
df = df.withColumn("rule2_midnight",
    when((col("hour") >= 23) | (col("hour") <= 5), "MIDNIGHT").otherwise("OK")
)

# Rule 3: Location Jump
w1 = Window.partitionBy("card_id").orderBy("txn_epoch")
df = df.withColumn("prev_city", lag("city", 1).over(w1))
df = df.withColumn("rule3_location_jump",
    when(
        (col("prev_city").isNotNull()) & (col("prev_city") != col("city")),
        "LOCATION JUMP"
    ).otherwise("OK")
)

# Rule 4: High Velocity
w2 = Window.partitionBy("card_id").orderBy("txn_epoch").rowsBetween(-2, 0)
df = df.withColumn("txn_count_3rows", count("txn_id").over(w2))
df = df.withColumn("rule4_velocity",
    when(col("txn_count_3rows") >= 3, "HIGH VELOCITY").otherwise("OK")
)

# Final Verdict
df = df.withColumn("fraud_status",
    when(
        (col("rule1_high_amount")   != "OK") |
        (col("rule2_midnight")      != "OK") |
        (col("rule3_location_jump") != "OK") |
        (col("rule4_velocity")      != "OK"),
        "FRAUD"
    ).otherwise("OK")
)

print("    Rules applied successfully!")

# ════════════════════════════════
# RESULTS
# ════════════════════════════════
print("\n[3] RESULTS:")

print("\n All Transactions:")
df.select(
    "txn_id", "card_id", "amount", "city", "month", "fraud_status"
).show(truncate=False)

print("\n Fraud by Month:")
df.groupBy("month", "fraud_status") \
  .count() \
  .orderBy("month", "fraud_status") \
  .show()

print("\n Overall Summary:")
df.groupBy("fraud_status").agg(
    count("txn_id").alias("transactions"),
    sum("amount").alias("total_amount")
).show()

# ════════════════════════════════
# LOAD — Save batch output
# ════════════════════════════════
print("[4] LOAD — Saving batch results...")

os.makedirs("output", exist_ok=True)

df.select(
    "txn_id", "card_id", "amount", "city",
    "channel", "month", "fraud_status",
    "rule1_high_amount", "rule2_midnight",
    "rule3_location_jump", "rule4_velocity"
).coalesce(1).write \
 .mode("overwrite") \
 .option("header", "true") \
 .csv("output/batch_results")

print("    output/batch_results/ saved!")

fraud_count = df.filter(col("fraud_status") == "FRAUD").count()
print("\n" + "=" * 55)
print("   BATCH PIPELINE COMPLETE!")
print("=" * 55)
print(f"   Files Processed  : {len(batch_files)}")
print(f"   Total Txns       : {total}")
print(f"   FRAUD Detected   : {fraud_count}")
print(f"   Fraud Rate       : {round(fraud_count/total*100, 1)}%")

spark.stop()
print("\n Batch Pipeline finished!")