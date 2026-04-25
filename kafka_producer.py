import json
import time
import random
from kafka import KafkaProducer

# Fake transaction generator
cards     = ["card_001", "card_002", "card_003", "card_004", "card_005"]
cities    = ["Chennai", "Mumbai", "Delhi", "Bangalore", "Dubai", "London"]
channels  = ["POS", "Online", "ATM"]

def generate_transaction(txn_id):
    return {
        "txn_id"  : txn_id,
        "card_id" : random.choice(cards),
        "amount"  : random.randint(100, 999999),
        "city"    : random.choice(cities),
        "channel" : random.choice(channels),
    }

print("=" * 50)
print("   KAFKA PRODUCER — Sending Transactions")
print("=" * 50)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

for i in range(1, 11):
    txn = generate_transaction(i)
    producer.send("transactions", txn)
    print(f"   Sent txn {i}: card={txn['card_id']} amount=₹{txn['amount']} city={txn['city']}")
    time.sleep(1)

producer.flush()
print("\n 10 transactions sent!")