import json
import time
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent

MODEL = joblib.load(BASE / "models/fraud_model_v4.joblib")

with open(BASE / "models/decision_threshold_v5.json") as f:
    CONFIG = json.load(f)

THRESHOLD = float(CONFIG["threshold"])

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"

TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

FEATURES = [
    "amount",
    "log_amount",
    "transaction_type",
    "transaction_hour",
    "recipient_category"
]

DB = BASE / "data" / "v3_history.db"
DB.parent.mkdir(exist_ok=True)

app = FastAPI(
    title="Financial Fraud Detection System",
    version="5.0.0"
)


class Transaction(BaseModel):
    customer_name: str = Field(min_length=1, max_length=100)
    amount: float = Field(ge=0, le=1_000_000_000)
    transaction_hour: int = Field(ge=0, le=23)
    transaction_type: str
    recipient_account_type: str


def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hybrid_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                customer_name TEXT,
                amount REAL,
                transaction_hour INTEGER,
                transaction_type TEXT,
                recipient_account_type TEXT,
                prediction TEXT,
                fraud_probability REAL,
                gemma_risk_level TEXT,
                recommended_action TEXT,
                latency_ms REAL,
                timestamp TEXT
            )
        """)


init_db()


def assess(tx):
    if tx.transaction_type not in TYPE_MAP:
        raise HTTPException(400, "Unsupported transaction type")

    if tx.recipient_account_type not in ("CUSTOMER", "MERCHANT"):
        raise HTTPException(400, "Invalid recipient account type")

    row = pd.DataFrame([{
        "amount": tx.amount,
        "log_amount": np.log1p(tx.amount),
        "transaction_type": TYPE_MAP[tx.transaction_type],
        "transaction_hour": tx.transaction_hour,
        "recipient_category": (
            1 if tx.recipient_account_type == "CUSTOMER" else 0
        )
    }], columns=FEATURES)

    start = time.perf_counter()

    probability = float(MODEL.predict_proba(row)[0][1])
    prediction = (
        "FRAUD" if probability >= THRESHOLD else "LEGITIMATE"
    )

    model_latency = (time.perf_counter() - start) * 1000

    return probability, prediction, model_latency


def explain_with_gemma(tx, probability, prediction):
    # Python is authoritative for classification, risk and action.
    if prediction == "FRAUD":
        risk_level = "FRAUD"
        recommended_action = "FLAG"
        policy_steps = (
            "Block or decline according to the configured financial policy, "
            "notify the fraud team and initiate an investigation."
        )
    elif probability >= 0.80:
        risk_level = "HIGH"
        recommended_action = "REVIEW"
        policy_steps = (
            "Temporarily hold the transaction for verification, "
            "alert the fraud team and verify the customer."
        )
    elif probability >= 0.50:
        risk_level = "MEDIUM"
        recommended_action = "REVIEW"
        policy_steps = (
            "Request additional authentication and verify the transaction "
            "before proceeding."
        )
    else:
        risk_level = "LOW"
        recommended_action = "ALLOW"
        policy_steps = (
            "Allow the transaction and continue routine monitoring."
        )

    comparison = "at or above" if probability >= THRESHOLD else "below"
    verified_fact = (
        f"Random Forest classified this transaction as {prediction} "
        f"because its fraud score ({probability:.6f}) is "
        f"{comparison} the decision threshold ({THRESHOLD:.6f})."
    )

    prompt = f"""
You are explaining a financial transaction assessment to an investigator.

The backend has already decided the classification, risk level and action.
These decisions are authoritative. Do not change them.

Transaction:
- Amount: {tx.amount}
- Type: {tx.transaction_type}
- Simulation hour: {tx.transaction_hour}
- Recipient account type: {tx.recipient_account_type}

Classification: {prediction}
Risk level: {risk_level}
Required action: {recommended_action}
Backend policy guidance: {policy_steps}

Write:
1. One short explanation using only the supplied transaction details.
2. A concise, practical next-step suggestion consistent with the backend policy.

Do not invent customer history, location, frequency or behaviour.
Do not describe simulation hour as a real-world clock time.
Do not override the classification, risk or required action.
Do not claim that fraud is confirmed solely from an uncalibrated score.

Return JSON only:
{{
  "reasoning": "Short transaction explanation",
  "suggested_next_steps": "Practical next steps"
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 160},
            "messages": [
                {"role": "system", "content": "Return concise valid JSON only."},
                {"role": "user", "content": prompt}
            ]
        },
        timeout=150
    )

    response.raise_for_status()
    content = response.json()["message"]["content"].strip()

    if content.startswith("```"):
        content = content.replace("```json", "", 1).replace("```", "").strip()

    generated = json.loads(content)

    return {
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "reasoning": verified_fact + " " + str(generated.get("reasoning", "")),
        "suggested_next_steps": str(
            generated.get("suggested_next_steps") or policy_steps
        ),
    }


@app.get("/")
def home():
    return {
        "service": "Financial Fraud Detection System",
        "version": "5.0.0",
        "status": "active"
    }


@app.get("/health")
def health():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=3
        )
        response.raise_for_status()
        names = [
            model["name"]
            for model in response.json().get("models", [])
        ]

        return {
            "status": "healthy",
            "model": "Random Forest V4 + V5 threshold policy",
            "gemma_model": OLLAMA_MODEL,
            "ollama_available": True,
            "gemma_available": OLLAMA_MODEL in names
        }

    except requests.RequestException:
        return {
            "status": "degraded",
            "model": "Random Forest V4 + V5 threshold policy",
            "gemma_available": False
        }


@app.post("/predict")
def predict(tx: Transaction):
    transaction_id = "TXN-" + uuid.uuid4().hex[:10].upper()

    probability, prediction, model_latency = assess(tx)

    try:
        gemma = explain_with_gemma(tx, probability, prediction)
        gemma_status = "success"

    except Exception as e:
        print(f"Gemma ERROR: {type(e).__name__}: {e}", flush=True)
        gemma = {
            "risk_level": (
                "HIGH" if prediction == "FRAUD" else "MEDIUM"
            ),
            "reasoning": (
                "Gemma explanation unavailable. "
                "Classification and risk policy are determined by the backend."
            ),
            "recommended_action": (
                "FLAG" if prediction == "FRAUD"
                else "REVIEW" if probability >= 0.50
                else "ALLOW"
            ),
            "suggested_next_steps": (
                "Block or decline according to policy and initiate investigation."
                if prediction == "FRAUD"
                else "Hold for verification and alert the fraud team."
                if probability >= 0.80
                else "Request additional authentication."
                if probability >= 0.50
                else "Allow and continue routine monitoring."
            )
        }
        gemma_status = "unavailable"

    result = {
        "transaction_id": transaction_id,
        "customer_name": tx.customer_name,
        "model": "Random Forest V4 + V5 threshold policy",
        "prediction": prediction,
        "fraud_probability": round(probability, 6),
        "decision_threshold": THRESHOLD,
        "probability_note": "Uncalibrated model probability estimate",
        "gemma_model": OLLAMA_MODEL,
        "gemma_status": gemma_status,
        "risk_level": gemma["risk_level"],
        "suggested_next_steps": gemma["suggested_next_steps"],
        "reasoning": gemma["reasoning"],
        "recommended_action": gemma["recommended_action"],
        "model_latency_ms": round(model_latency, 2)
    }

    with sqlite3.connect(DB) as conn:
        conn.execute("""
            INSERT INTO hybrid_history (
                transaction_id, customer_name, amount,
                transaction_hour, transaction_type,
                recipient_account_type, prediction,
                fraud_probability, gemma_risk_level,
                recommended_action, latency_ms, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction_id, tx.customer_name, tx.amount,
            tx.transaction_hour, tx.transaction_type,
            tx.recipient_account_type, prediction,
            probability, gemma["risk_level"],
            gemma["recommended_action"],
            model_latency, datetime.now().isoformat()
        ))

    return result


@app.get("/history")
def history():
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM hybrid_history
            ORDER BY id DESC LIMIT 100
        """).fetchall()

    return [dict(row) for row in rows]
