import json
import time
import sqlite3
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
MODEL = joblib.load(BASE / "models/fraud_model_v3.joblib")

with open(BASE / "models/decision_threshold_v3.json") as f:
    CONFIG = json.load(f)

THRESHOLD = float(CONFIG["threshold"])

TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

FEATURES = [
    "step", "amount", "log_amount",
    "transaction_type", "sender_category", "recipient_category"
]

DB = BASE / "data" / "v3_history.db"
DB.parent.mkdir(exist_ok=True)

app = FastAPI(title="EdgeGuard PaySim V3", version="3.0.0")


class Transaction(BaseModel):
    step: int = Field(ge=1, le=743)
    amount: float = Field(ge=0)
    transaction_type: str
    sender_category: str
    recipient_category: str


def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER,
                amount REAL,
                transaction_type TEXT,
                prediction TEXT,
                probability REAL,
                latency_ms REAL,
                timestamp TEXT
            )
        """)


init_db()


def assess(tx):
    if tx.transaction_type not in TYPE_MAP:
        raise ValueError("Unsupported transaction type")

    if tx.sender_category not in ("CUSTOMER", "MERCHANT"):
        raise ValueError("Invalid sender category")

    if tx.recipient_category not in ("CUSTOMER", "MERCHANT"):
        raise ValueError("Invalid recipient category")

    row = pd.DataFrame([{
        "step": tx.step,
        "amount": tx.amount,
        "log_amount": __import__("numpy").log1p(tx.amount),
        "transaction_type": TYPE_MAP[tx.transaction_type],
        "sender_category": 1 if tx.sender_category == "CUSTOMER" else 0,
        "recipient_category": 1 if tx.recipient_category == "CUSTOMER" else 0
    }], columns=FEATURES)

    start = time.perf_counter()
    probability = float(MODEL.predict_proba(row)[0][1])
    latency = (time.perf_counter() - start) * 1000
    prediction = "FRAUD" if probability >= THRESHOLD else "LEGITIMATE"

    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 6),
        "decision_threshold": THRESHOLD,
        "latency_ms": round(latency, 3)
    }


@app.get("/")
def home():
    return {"service": "EdgeGuard PaySim V3", "status": "active"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "PaySim Random Forest V3"}


@app.post("/predict")
def predict(tx: Transaction):
    result = assess(tx)

    with sqlite3.connect(DB) as conn:
        conn.execute("""
            INSERT INTO history
            (step, amount, transaction_type, prediction,
             probability, latency_ms, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            tx.step, tx.amount, tx.transaction_type,
            result["prediction"], result["fraud_probability"],
            result["latency_ms"], datetime.now().isoformat()
        ))

    return result


@app.get("/history")
def history():
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/explain")
def explain(tx: Transaction):
    result = assess(tx)
    return {
        **result,
        "explanation": (
            f"The model classified this {tx.transaction_type} transaction "
            f"of amount {tx.amount:,.2f} as {result['prediction']}. "
            f"Its estimated fraud probability is "
            f"{result['fraud_probability']:.2%}, compared with the "
            f"decision threshold of {THRESHOLD:.2%}. "
            "This is a model risk assessment, not confirmation of fraud. "
            "Human review is recommended for suspicious transactions."
        )
    }
