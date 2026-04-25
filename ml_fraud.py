import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("=" * 55)
print("   ML MODEL — FRAUD DETECTION")
print("=" * 55)

# ════════════════════════════════
# STEP 1: Training Data Create
# ════════════════════════════════
print("\n[1] Creating training data...")

data = {
    "amount":   [5000, 150000, 800, 999999, 2000, 3000, 500000,
                 1200, 75000, 200, 300, 250, 50000, 8000, 900,
                 450000, 600, 1500, 800000, 3500, 250000, 700,
                 400, 550, 900000, 2500, 125000, 680, 320, 750],
    "hour":     [10, 14, 23, 9, 15, 15, 2, 23, 11, 8,
                 8, 8, 11, 10, 23, 3, 12, 13, 1, 16,
                 2, 9, 10, 11, 0, 14, 15, 8, 9, 10],
    "is_foreign_city": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
                        1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "channel":  ["POS","ATM","POS","Online","POS","POS","Online",
                 "Online","Online","POS","POS","POS","Online","Online","POS",
                 "Online","POS","POS","ATM","POS","Online","POS",
                 "POS","POS","Online","POS","ATM","POS","POS","POS"],
    "fraud":    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
                 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

# Channel encode பண்றோம் (text → number)
le = LabelEncoder()
df["channel_encoded"] = le.fit_transform(df["channel"])

print(f"   Total training records: {len(df)}")
print(f"   FRAUD records: {df['fraud'].sum()}")
print(f"   CLEAN records: {(df['fraud']==0).sum()}")

# ════════════════════════════════
# STEP 2: Model Train பண்றோம்
# ════════════════════════════════
print("\n[2] Training ML model...")

features = ["amount", "hour", "is_foreign_city", "channel_encoded"]
X = df[features]
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("   Model trained successfully!")

# ════════════════════════════════
# STEP 3: Model Test பண்றோம்
# ════════════════════════════════
print("\n[3] Testing model accuracy...")

y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100
print(f"   Accuracy: {accuracy:.1f}%")

print("\n   Detailed Report:")
print(classification_report(y_test, y_pred,
      target_names=["CLEAN", "FRAUD"]))

# ════════════════════════════════
# STEP 4: New Transactions Predict
# ════════════════════════════════
print("\n[4] Predicting new transactions...")

new_transactions = pd.DataFrame({
    "amount":          [5000,  500000, 800,   999999, 1500],
    "hour":            [10,    2,      23,    9,      14  ],
    "is_foreign_city": [0,     1,      0,     1,      0   ],
    "channel":         ["POS", "Online","POS","ATM",  "POS"]
})

new_transactions["channel_encoded"] = le.transform(new_transactions["channel"])

predictions  = model.predict(new_transactions[features])
probabilities = model.predict_proba(new_transactions[features])

print("\n" + "=" * 65)
print(f"  {'TXN':<4} {'AMOUNT':<10} {'HOUR':<6} {'FOREIGN':<9} {'CHANNEL':<8} {'RESULT':<8} {'CONFIDENCE'}")
print("=" * 65)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    txn = new_transactions.iloc[i]
    result = "FRAUD" if pred == 1 else "OK"
    confidence = max(prob) * 100
    foreign = "YES" if txn["is_foreign_city"] == 1 else "NO"
    print(f"  {i+1:<4} ₹{txn['amount']:<9} {int(txn['hour']):<6} {foreign:<9} {txn['channel']:<8} {result:<8} {confidence:.1f}%")

print("=" * 65)

# ════════════════════════════════
# STEP 5: Feature Importance
# ════════════════════════════════
print("\n[5] Which features matter most?")
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print()
for _, row in importance.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"   {row['Feature']:<20} {bar} {row['Importance']:.3f}")

print("\n ML Model Complete!")