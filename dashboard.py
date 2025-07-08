# dashboard.py
from flask import Flask, render_template, request
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

@app.route("/")
def dashboard():
    try:
        df = pd.read_csv("threat_logs.csv",usecols=["timestamp", "type", "confidence", "chat", 
                                "user", "message", "user_id", "chat_id", "chat_type"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    except (FileNotFoundError, ValueError):
        df = pd.DataFrame(columns=["timestamp", "type", "confidence", "chat", 
                                 "user", "message", "user_id", "chat_id", "chat_type"])

    # Filter by scam type if requested
    selected_type = request.args.get("type")
    if selected_type:
        df = df[df["type"] == selected_type]

    total = len(df)
    threats = df["type"].notnull().sum()
    threats_last_24h = len(df[df["timestamp"] >= pd.Timestamp.now() - pd.Timedelta("1d")])
    chat_count = df["chat"].nunique()
    detection_rate = round((threats / total) * 100, 2) if total else 0

    critical = len(df[df["confidence"] > 0.9])
    high = len(df[(df["confidence"] > 0.75) & (df["confidence"] <= 0.9)])
    low = len(df[df["confidence"] <= 0.75])
    recent = df.sort_values("timestamp", ascending=False).head(10).to_dict(orient="records")
    
    # Prepare recent threats data for the new cards
    recent_threats = []
    for idx, row in df.sort_values("timestamp", ascending=False).head(5).iterrows():
        recent_threats.append({
            "type": row["type"],
            "message": row["message"],
            "user": row["user"],
            "user_id": row.get("user_id", "N/A"),  # You'll need to get this from your data
            "chat": row["chat"],
            "chat_type": row.get("chat_type", "private"),  # Get from CSV
        "chat_id": row.get("chat_id", "N/A"),  # Get from CSV
        "timestamp": row["timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
        "detection_time": (row["timestamp"] + pd.Timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S'),
        "confidence": row["confidence"],
        "message_id": idx
    })

    return render_template(
        "dashboard.html",
        total=total,
        threats=threats,
        threats_last_24h=threats_last_24h,
        chat_count=chat_count,
        detection_rate=detection_rate,
        types=df["type"].value_counts().to_dict(),
        critical=critical,
        high=high,
        low=low,
        recent=recent,
        recent_threats=recent_threats,
        selected_type=selected_type
    )
@app.template_filter('number_format')
def number_format(value):
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value

if __name__ == "__main__":
    app.run(debug=True)
