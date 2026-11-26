import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"

TYPE_OPTIONS = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
ACCOUNT_OPTIONS = {"CUSTOMER", "MERCHANT"}

DB = BASE / "data" / "v3_history.db"
DB.parent.mkdir(exist_ok=True)

app = FastAPI(title="Financial Fraud Detection System", version="3.1.0")


class Transaction(BaseModel):
    amount: float = Field(ge=0, le=1_000_000_000)
    transaction_hour: int = Field(ge=0, le=23)
    transaction_type: str
    recipient_account_type: str


def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gemma_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL,
                transaction_hour INTEGER,
                transaction_type TEXT,
                recipient_account_type TEXT,
                risk_level TEXT,
                risk_score INTEGER,
                recommended_action TEXT,
                latency_ms REAL,
                timestamp TEXT
            )
        """)


init_db()


def assess_with_gemma(tx):
    if tx.transaction_type not in TYPE_OPTIONS:
        raise HTTPException(400, "Unsupported transaction type")

    if tx.recipient_account_type not in ACCOUNT_OPTIONS:
        raise HTTPException(400, "Invalid recipient account type")

    system_prompt = """
You are a cautious transaction risk assessment assistant for a
college prototype using the synthetic PaySim mobile-money dataset.

Assess ONLY the transaction fields provided.
Do not invent customer history, location, account behaviour,
transaction frequency, typical amounts, or external facts.
The transaction hour is only a simulation hour-of-day input.
Do not treat the hour alone as evidence of fraud.
Do not claim that a transaction is definitely fraudulent.

Return ONLY valid JSON with these exact fields:
{
  "risk_level": "LOW, MEDIUM, or HIGH",
  "risk_score": 0,
  "reasoning": "One short sentence based only on supplied fields",
  "recommended_action": "ALLOW, REVIEW, or FLAG"
}

risk_score is a subjective model assessment from 0 to 100.
It is NOT a calibrated fraud probability.
Use REVIEW when evidence is insufficient.
Temperature and creativity must be low.
"""

    user_prompt = (
        f"Amount: {tx.amount}\n"
        f"Transaction type: {tx.transaction_type}\n"
        f"Transaction hour: {tx.transaction_hour} (0-23)\n"
        f"Recipient account type: {tx.recipient_account_type}\n"
        "Dataset: synthetic PaySim mobile-money simulation."
    )

    start = time.perf_counter()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "format": "json",
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "num_predict": 100
                },
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            timeout=180
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        result = json.loads(content)

    except requests.Timeout:
        raise HTTPException(504, "Gemma inference timed out")
    except requests.RequestException:
        raise HTTPException(503, "Could not connect to local Ollama")
    except (KeyError, ValueError, json.JSONDecodeError):
        raise HTTPException(502, "Gemma returned an invalid response")

    latency = (time.perf_counter() - start) * 1000

    level = str(result.get("risk_level", "")).upper()
    action = str(result.get("recommended_action", "")).upper()

    if level not in {"LOW", "MEDIUM", "HIGH"}:
        raise HTTPException(502, "Invalid risk level from Gemma")
    if action not in {"ALLOW", "REVIEW", "FLAG"}:
        raise HTTPException(502, "Invalid action from Gemma")

    try:
        score = int(result["risk_score"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(502, "Invalid risk score from Gemma")

    if not 0 <= score <= 100:
        raise HTTPException(502, "Risk score must be between 0 and 100")

    reasoning = str(result.get("reasoning", "")).strip()
    if not reasoning:
        raise HTTPException(502, "Gemma returned no explanation")

    return {
        "model": OLLAMA_MODEL,
        "prediction": level,
        "risk_level": level,
        "risk_score": score,
        "score_type": "Uncalibrated LLM assessment",
        "reasoning": reasoning,
        "recommended_action": action,
        "latency_ms": round(latency, 2)
    }


@app.get("/")
def home():
    return {"service": "Financial Fraud Detection System", "status": "active"}


@app.get("/health")
def health():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        available = OLLAMA_MODEL in models
        return {
            "status": "healthy" if available else "degraded",
            "model": OLLAMA_MODEL,
            "ollama_available": True,
            "model_available": available
        }
    except requests.RequestException:
        return {
            "status": "degraded",
            "model": OLLAMA_MODEL,
            "ollama_available": False
        }


@app.post("/predict")
def predict(tx: Transaction):
    result = assess_with_gemma(tx)

    with sqlite3.connect(DB) as conn:
        conn.execute("""
            INSERT INTO gemma_history (
                amount, transaction_hour, transaction_type,
                recipient_account_type, risk_level, risk_score,
                recommended_action, latency_ms, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tx.amount, tx.transaction_hour, tx.transaction_type,
            tx.recipient_account_type, result["risk_level"],
            result["risk_score"], result["recommended_action"],
            result["latency_ms"], datetime.now().isoformat()
        ))

    return result


@app.get("/history")
def history():
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM gemma_history ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return [dict(row) for row in rows]
