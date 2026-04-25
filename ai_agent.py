import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import random

print("=" * 60)
print("   AI AGENT — SMART FRAUD DETECTION SYSTEM")
print("=" * 60)

# ════════════════════════════
# TRAIN ML MODEL
# ════════════════════════════
data = {
    "amount":          [5000,150000,800,999999,2000,3000,500000,
                        1200,75000,200,300,250,50000,8000,900,
                        450000,600,1500,800000,3500,250000,700,
                        400,550,900000,2500,125000,680,320,750],
    "hour":            [10,14,23,9,15,15,2,23,11,8,
                        8,8,11,10,23,3,12,13,1,16,
                        2,9,10,11,0,14,15,8,9,10],
    "is_foreign_city": [0,0,0,1,0,0,1,0,0,0,
                        0,0,0,1,0,1,0,0,1,0,
                        1,0,0,0,1,0,0,0,0,0],
    "channel":         ["POS","ATM","POS","Online","POS","POS","Online",
                        "Online","Online","POS","POS","POS","Online","Online","POS",
                        "Online","POS","POS","ATM","POS","Online","POS",
                        "POS","POS","Online","POS","ATM","POS","POS","POS"],
    "fraud":           [0,1,1,1,0,0,1,1,0,0,
                        0,1,0,1,1,1,0,0,1,0,
                        1,0,0,0,1,0,1,0,0,0]
}

df = pd.DataFrame(data)
le = LabelEncoder()
df["channel_encoded"] = le.fit_transform(df["channel"])

features = ["amount","hour","is_foreign_city","channel_encoded"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df[features], df["fraud"])

# ════════════════════════════
# AI AGENT FUNCTION
# ════════════════════════════
def ai_agent_analyze(txn):
    # ML Score
    channel_enc = le.transform([txn["channel"]])[0]
    X = [[txn["amount"], txn["hour"],
          txn["is_foreign"], channel_enc]]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    risk_score = int(prob[1] * 100)

    # AI Agent — Rule reasoning
    reasons  = []
    actions  = []

    if txn["amount"] > 100000:
        reasons.append(f"High amount ₹{txn['amount']:,} (above ₹1 lakh limit)")
        actions.append("VERIFY AMOUNT")

    if txn["hour"] >= 23 or txn["hour"] <= 5:
        reasons.append(f"Suspicious time — {txn['hour']}:00 (midnight transaction)")
        actions.append("CALL CUSTOMER")

    if txn["is_foreign"] == 1:
        reasons.append(f"Foreign city transaction detected")
        actions.append("CHECK TRAVEL HISTORY")

    if txn["channel"] == "ATM" and txn["amount"] > 50000:
        reasons.append(f"Large ATM withdrawal ₹{txn['amount']:,}")
        actions.append("VERIFY ATM LOCATION")

    # Final verdict
    if pred == 1:
        verdict = "FRAUD"
        final_action = "BLOCK TRANSACTION"
        emoji = "FRAUD"
    else:
        verdict = "OK"
        final_action = "APPROVE"
        emoji = "OK"

    # AI Explanation
    if verdict == "FRAUD":
        explanation = (
            f"This transaction shows {len(reasons)} suspicious "
            f"indicator(s). Risk score is {risk_score}/100. "
            f"Immediate action required: {final_action}."
        )
    else:
        explanation = (
            f"This transaction appears normal. "
            f"Risk score is {risk_score}/100. "
            f"Safe to approve."
        )

    return {
        "verdict"    : verdict,
        "risk_score" : risk_score,
        "reasons"    : reasons,
        "action"     : final_action,
        "explanation": explanation
    }

# ════════════════════════════
# TEST TRANSACTIONS
# ════════════════════════════
transactions = [
    {"txn_id":1, "card_id":"card_001", "amount":5000,
     "city":"Chennai",   "channel":"POS",    "hour":10, "is_foreign":0},
    {"txn_id":2, "card_id":"card_002", "amount":999999,
     "city":"Dubai",     "channel":"ATM",    "hour":2,  "is_foreign":1},
    {"txn_id":3, "card_id":"card_003", "amount":800,
     "city":"Chennai",   "channel":"POS",    "hour":23, "is_foreign":0},
    {"txn_id":4, "card_id":"card_004", "amount":500000,
     "city":"London",    "channel":"Online", "hour":3,  "is_foreign":1},
    {"txn_id":5, "card_id":"card_005", "amount":1500,
     "city":"Bangalore", "channel":"POS",    "hour":14, "is_foreign":0},
]

# ════════════════════════════
# PRINT RESULTS
# ════════════════════════════
fraud_count = 0
ok_count    = 0

for txn in transactions:
    result = ai_agent_analyze(txn)

    print(f"\n{'─'*60}")
    print(f" Transaction {txn['txn_id']} | {txn['card_id']} | "
          f"₹{txn['amount']:,} | {txn['city']} | {txn['hour']}:00")
    print(f"{'─'*60}")
    print(f" VERDICT     : {result['verdict']}")
    print(f" RISK SCORE  : {result['risk_score']}/100")
    print(f" ACTION      : {result['action']}")

    if result["reasons"]:
        print(f" REASONS     :")
        for r in result["reasons"]:
            print(f"   - {r}")
    else:
        print(f" REASONS     : No suspicious activity")

    print(f" EXPLANATION : {result['explanation']}")

    if result["verdict"] == "FRAUD":
        fraud_count += 1
    else:
        ok_count += 1

# ════════════════════════════
# FINAL SUMMARY
# ════════════════════════════
print(f"\n{'='*60}")
print(f"   AI AGENT SUMMARY")
print(f"{'='*60}")
print(f"   Total Analyzed : {len(transactions)}")
print(f"   FRAUD Blocked  : {fraud_count}")
print(f"   OK Approved    : {ok_count}")
print(f"   Detection Rate : {round(fraud_count/len(transactions)*100,1)}%")
print(f"{'='*60}")
print("\n AI Agent Complete!")