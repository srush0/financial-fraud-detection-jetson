import time
import sqlite3
import joblib

from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Edge Financial Fraud Scanner",
    version="1.1.0"
)

model = joblib.load("models/fraud_model.joblib")

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


@app.post("/predict")
def predict(transaction: Transaction):
    start = time.perf_counter()

    data = [[
        transaction.amount,
        transaction.hour,
        transaction.frequency,
        transaction.distance
    ]]

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    latency = (time.perf_counter() - start) * 1000

    label = "FRAUD" if prediction == 1 else "LEGITIMATE"

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
            label,
            probability,
            latency,
            datetime.now().isoformat()
        ))

    return {
        "prediction": label,
        "fraud_probability": round(probability, 4),
        "latency_ms": round(latency, 4)
    }


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
    import requests

    data = [[
        transaction.amount,
        transaction.hour,
        transaction.frequency,
        transaction.distance
    ]]

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    label = "FRAUD" if prediction == 1 else "LEGITIMATE"

    prompt = f"""
You are a financial transaction analysis assistant.

A Random Forest machine learning model has already classified
the transaction. Your job is ONLY to explain its result.

Transaction details:
- Amount: {transaction.amount}
- Hour: {transaction.hour}
- Recent transaction frequency: {transaction.frequency}
- Distance from usual location: {transaction.distance} km

Model prediction: {label}
Fraud probability: {probability:.2%}

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
            timeout=120
        )

        response.raise_for_status()
        explanation = response.json()["response"]

        return {
            "prediction": label,
            "fraud_probability": round(probability, 4),
            "ai_model": "gemma3:4b",
            "explanation": explanation
        }

    except requests.exceptions.RequestException as error:
        return {
            "prediction": label,
            "fraud_probability": round(probability, 4),
            "ai_model": "gemma3:4b",
            "explanation": None,
            "error": f"Ollama connection failed: {str(error)}"
        }
