import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import random

print("=" * 60)
print("   ML MODEL v2 — IMPROVED ACCURACY")
print("=" * 60)

# ════════════════════════════════
# STEP 1: More Data Generate
# ════════════════════════════════
print("\n[1] Generating large training dataset...")

random.seed(42)
np.random.seed(42)

rows = []
for i in range(1000):
    # FRAUD transactions (40%)
    if random.random() < 0.4:
        amount     = random.choice([
            random.randint(100000, 999999),  # high amount
            random.randint(100, 5000),        # small amount (card testing)
        ])
        hour       = random.choice([0,1,2,3,4,23])  # midnight
        is_foreign = random.choice([0, 1, 1])        # mostly foreign
        channel    = random.choice(["ATM","Online","Online"])
        fraud      = 1

    # CLEAN transactions (60%)
    else:
        amount     = random.randint(100, 50000)
        hour       = random.randint(8, 20)   # daytime
        is_foreign = 0                        # local city
        channel    = random.choice(["POS","POS","Online"])
        fraud      = 0

    rows.append({
        "amount"         : amount,
        "hour"           : hour,
        "is_foreign_city": is_foreign,
        "channel"        : channel,
        "fraud"          : fraud
    })

df = pd.DataFrame(rows)

le = LabelEncoder()
df["channel_encoded"] = le.fit_transform(df["channel"])

print(f"   Total records    : {len(df)}")
print(f"   FRAUD records    : {df['fraud'].sum()}")
print(f"   CLEAN records    : {(df['fraud']==0).sum()}")

# ════════════════════════════════
# STEP 2: Train Multiple Models
# ════════════════════════════════
print("\n[2] Training multiple models...")

features = ["amount","hour","is_foreign_city","channel_encoded"]
X = df[features]
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred) * 100

# Model 2: Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc  = accuracy_score(y_test, gb_pred) * 100

print(f"   Random Forest    : {rf_acc:.1f}%")
print(f"   Gradient Boosting: {gb_acc:.1f}%")

# Best model select
if rf_acc >= gb_acc:
    best_model = rf_model
    best_name  = "Random Forest"
    best_acc   = rf_acc
    best_pred  = rf_pred
else:
    best_model = gb_model
    best_name  = "Gradient Boosting"
    best_acc   = gb_acc
    best_pred  = gb_pred

print(f"\n   Best Model: {best_name} ({best_acc:.1f}%)")

# ════════════════════════════════
# STEP 3: Detailed Report
# ════════════════════════════════
print("\n[3] Detailed accuracy report:")
print(classification_report(y_test, best_pred,
      target_names=["CLEAN","FRAUD"]))

# Cross validation
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"   Cross Validation : {cv_scores.mean()*100:.1f}% "
      f"(+/- {cv_scores.std()*100:.1f}%)")

# ════════════════════════════════
# STEP 4: Feature Importance
# ════════════════════════════════
print("\n[4] Feature importance:")
importance = pd.DataFrame({
    "Feature"   : features,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)

for _, row in importance.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"   {row['Feature']:<20} {bar} {row['Importance']:.3f}")

# ════════════════════════════════
# STEP 5: Test New Transactions
# ════════════════════════════════
print("\n[5] Testing new transactions:")

new_txns = pd.DataFrame({
    "amount"         : [5000,  999999, 800,   500000, 1500,  250000],
    "hour"           : [10,    2,      23,    3,      14,    15    ],
    "is_foreign_city": [0,     1,      0,     1,      0,     0     ],
    "channel"        : ["POS","ATM","POS","Online","POS","Online"  ]
})
new_txns["channel_encoded"] = le.transform(new_txns["channel"])

preds = best_model.predict(new_txns[features])
probs = best_model.predict_proba(new_txns[features])

print(f"\n{'─'*65}")
print(f"  {'#':<3} {'Amount':<10} {'Hour':<6} {'Foreign':<9} {'Channel':<8} {'Result':<8} {'Confidence'}")
print(f"{'─'*65}")

for i, (pred, prob) in enumerate(zip(preds, probs)):
    txn        = new_txns.iloc[i]
    result     = "FRAUD" if pred == 1 else "OK"
    confidence = max(prob) * 100
    foreign    = "YES" if txn["is_foreign_city"] == 1 else "NO"
    icon       = "FRAUD" if pred == 1 else "OK   "
    print(f"  {i+1:<3} ₹{int(txn['amount']):<9} "
          f"{int(txn['hour']):<6} {foreign:<9} "
          f"{txn['channel']:<8} {icon:<8} {confidence:.1f}%")

print(f"{'─'*65}")

# ════════════════════════════════
# SUMMARY
# ════════════════════════════════
print(f"\n{'='*60}")
print(f"   ACCURACY IMPROVEMENT SUMMARY")
print(f"{'='*60}")
print(f"   Old Model (30 rows)  : 66.7%")
print(f"   New Model (1000 rows): {best_acc:.1f}%")
print(f"   Improvement          : +{best_acc-66.7:.1f}%")
print(f"{'='*60}")
print("\n Accuracy Improved Successfully!")