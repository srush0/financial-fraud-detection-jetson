import time
import sqlite3
import json
import joblib
import numpy as np
import pandas as pd
import requests

from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Edge Financial Fraud Scanner",
    version="2.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model_v2.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "decision_threshold.json"

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    threshold_config = json.load(file)

DECISION_THRESHOLD = float(threshold_config["threshold"])

FEATURES = [
    "amount",
    "hour_sin",
    "hour_cos",
    "frequency",
    "distance"
]

if list(model.feature_names_in_) != FEATURES:
    raise ValueError("Model feature configuration mismatch.")

DB_PATH = "data/transactions.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL,
                hour INTEGER,
                frequency INTEGER,
                distance REAL,
                prediction TEXT,
                fraud_probability REAL,
                latency_ms REAL,
                timestamp TEXT
            )
        """)


init_db()


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    frequency: int = Field(ge=0)
    distance: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Edge Fraud Scanner is running",
        "status": "active"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def analyze_transaction(transaction: Transaction):
    start = time.perf_counter()

    hour = transaction.hour

    data = pd.DataFrame([{
        "amount": transaction.amount,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "frequency": transaction.frequency,
        "distance": transaction.distance
    }], columns=FEATURES)

    probability = float(model.predict_proba(data)[0][1])
    prediction = int(probability >= DECISION_THRESHOLD)

    latency = (time.perf_counter() - start) * 1000
    label = "FRAUD" if prediction == 1 else "LEGITIMATE"

    return {
        "prediction": label,
        "fraud_probability": round(probability, 4),
        "decision_threshold": DECISION_THRESHOLD,
        "latency_ms": round(latency, 4)
    }


@app.post("/predict")
def predict(transaction: Transaction):
    result = analyze_transaction(transaction)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_history
            (amount, hour, frequency, distance, prediction,
             fraud_probability, latency_ms, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction.amount,
            transaction.hour,
            transaction.frequency,
            transaction.distance,
            result["prediction"],
            result["fraud_probability"],
            result["latency_ms"],
            datetime.now().isoformat()
        ))

    return result


@app.get("/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM prediction_history
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

    return [dict(row) for row in rows]


@app.post("/explain")
def explain_transaction(transaction: Transaction):
    result = analyze_transaction(transaction)

    prompt = f"""
You are a financial transaction analysis assistant.

A Random Forest machine learning model has already classified
the transaction. Your task is ONLY to explain its result.

Transaction details:
- Amount: {transaction.amount}
- Hour: {transaction.hour}
- Recent transaction frequency: {transaction.frequency}
- Distance from usual location: {transaction.distance} km

Model prediction: {result["prediction"]}
Fraud probability: {result["fraud_probability"]:.2%}
Decision threshold: {DECISION_THRESHOLD:.2%}

Explain:
1. Why the model may have produced this result.
2. Which transaction characteristics may be relevant.
3. What a human reviewer should check.

Important:
- Do not change the model's prediction.
- Do not claim that fraud is confirmed.
- Explain that this is a model-generated risk assessment.
- Do not invent customer history or facts.
- Keep the explanation under 120 words.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 180
                }
            },
            timeout=180
        )

        response.raise_for_status()

        return {
            **result,
            "ai_model": "gemma3:4b",
            "explanation": response.json()["response"]
        }

    except requests.exceptions.RequestException as error:
        return {
            **result,
            "ai_model": "gemma3:4b",
            "explanation": None,
            "error": f"Ollama connection failed: {str(error)}"
        }
