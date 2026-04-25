import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"

import json
import random
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder \
    .appName("KafkaSimulator") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 55)
print("   KAFKA SIMULATE — Real-time Fraud Detection")
print("=" * 55)
print("   Transactions coming in one by one...\n")

cards    = ["card_001", "card_002", "card_003", "card_004", "card_005"]
cities   = ["Chennai", "Mumbai", "Delhi", "Bangalore", "Dubai", "London"]
channels = ["POS", "Online", "ATM"]

def detect_fraud(txn):
    reasons = []
    if txn["amount"] > 100000:
        reasons.append("HIGH AMOUNT")
    if txn["city"] in ["Dubai", "London"]:
        reasons.append("FOREIGN CITY")
    if txn["channel"] == "ATM" and txn["amount"] > 50000:
        reasons.append("HIGH ATM")
    return reasons

fraud_count = 0
ok_count    = 0

for i in range(1, 16):
    # Simulate incoming transaction
    txn = {
        "txn_id"  : i,
        "card_id" : random.choice(cards),
        "amount"  : random.randint(100, 999999),
        "city"    : random.choice(cities),
        "channel" : random.choice(channels),
    }

    reasons = detect_fraud(txn)

    if reasons:
        fraud_count += 1
        flag = "FRAUD"
        reason_str = ", ".join(reasons)
    else:
        ok_count += 1
        flag = "  OK "
        reason_str = "Clean transaction"

    print(f"   [{flag}] txn={i:02d} | "
          f"{txn['card_id']} | "
          f"₹{txn['amount']:>7} | "
          f"{txn['city']:<12} | "
          f"{txn['channel']:<7} | "
          f"{reason_str}")

    time.sleep(0.5)

print("\n" + "=" * 55)
print("   STREAMING COMPLETE!")
print("=" * 55)
print(f"   Total     : 15")
print(f"   FRAUD     : {fraud_count}")
print(f"   Clean     : {ok_count}")
print(f"   Fraud Rate: {round(fraud_count/15*100, 1)}%")

spark.stop()