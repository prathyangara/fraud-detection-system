from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Train Model
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

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection AI</title>
    <style>
        body { font-family: Arial; max-width: 700px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #d32f2f; text-align: center; }
        .form-box { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        input, select { width: 100%; padding: 10px; margin: 8px 0 16px 0; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; box-sizing: border-box; }
        button { width: 100%; padding: 15px; background: #d32f2f; color: white; border: none; border-radius: 5px; font-size: 18px; cursor: pointer; }
        button:hover { background: #b71c1c; }
        .result { margin-top: 20px; padding: 20px; border-radius: 10px; }
        .fraud { background: #ffebee; border: 2px solid #d32f2f; }
        .ok { background: #e8f5e9; border: 2px solid #2e7d32; }
        .verdict { font-size: 28px; font-weight: bold; text-align: center; }
        .fraud .verdict { color: #d32f2f; }
        .ok .verdict { color: #2e7d32; }
        label { font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <h1>Fraud Detection AI Agent</h1>
    <div class="form-box">
        <label>Amount</label>
        <input type="number" id="amount" value="5000">
        <label>Hour (0-23)</label>
        <input type="number" id="hour" value="10" min="0" max="23">
        <label>Channel</label>
        <select id="channel">
            <option value="POS">POS</option>
            <option value="Online">Online</option>
            <option value="ATM">ATM</option>
        </select>
        <label>Foreign City?</label>
        <select id="is_foreign">
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
        <button onclick="analyze()">Analyze Transaction</button>
        <div id="result"></div>
    </div>
    <script>
        async function analyze() {
            const data = {
                amount: parseInt(document.getElementById('amount').value),
                hour: parseInt(document.getElementById('hour').value),
                channel: document.getElementById('channel').value,
                is_foreign: parseInt(document.getElementById('is_foreign').value)
            };
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await res.json();
            const isfraud = result.verdict === 'FRAUD';
            const cls = isfraud ? 'fraud' : 'ok';
            const icon = isfraud ? 'FRAUD' : 'OK';
            let reasonsHtml = result.reasons.length > 0
                ? '<ul>' + result.reasons.map(r => '<li>' + r + '</li>').join('') + '</ul>'
                : '<p>No suspicious activity.</p>';
            document.getElementById('result').innerHTML =
                '<div class="result ' + cls + '">' +
                '<div class="verdict">' + icon + '</div>' +
                '<p><b>Risk Score:</b> ' + result.risk_score + '/100</p>' +
                '<p><b>Action:</b> ' + result.action + '</p>' +
                '<p><b>Reasons:</b></p>' + reasonsHtml +
                '<p><b>Explanation:</b> ' + result.explanation + '</p>' +
                '</div>';
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    txn = request.json
    channel_enc = le.transform([txn["channel"]])[0]
    X = pd.DataFrame(
        [[txn["amount"], txn["hour"], txn["is_foreign"], channel_enc]],
        columns=features
    )
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    risk_score = int(prob[1] * 100)

    reasons = []
    if txn["amount"] > 100000:
        reasons.append("High amount above 1 lakh")
    if txn["hour"] >= 23 or txn["hour"] <= 5:
        reasons.append("Midnight transaction")
    if txn["is_foreign"] == 1:
        reasons.append("Foreign city detected")
    if txn["channel"] == "ATM" and txn["amount"] > 50000:
        reasons.append("Large ATM withdrawal")

    verdict = "FRAUD" if pred == 1 else "OK"
    action = "BLOCK TRANSACTION" if pred == 1 else "APPROVE"

    if verdict == "FRAUD":
        explanation = (
            "Transaction shows " + str(len(reasons)) +
            " suspicious indicators. Risk score " +
            str(risk_score) + "/100. Action: " + action
        )
    else:
        explanation = (
            "Transaction appears normal. Risk score " +
            str(risk_score) + "/100. Safe to approve."
        )

    return jsonify({
        "verdict": verdict,
        "risk_score": risk_score,
        "action": action,
        "reasons": reasons,
        "explanation": explanation
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)