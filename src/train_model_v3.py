import json
import glob
import time
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
    classification_report
)

FEATURES = [
    "step", "amount", "log_amount",
    "transaction_type", "sender_category", "recipient_category"
]

def load_split(name):
    files = sorted(glob.glob(f"data/v3/{name}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found for {name}")
    data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"{name}: {len(data):,} rows | {int(data.isFraud.sum()):,} fraud")
    return data

print("Loading V3 training data...")
train = load_split("train")
valid = load_split("validation")
test = load_split("test")

# Keep every fraud case and sample legitimate transactions.
fraud = train[train.isFraud == 1]
legit = train[train.isFraud == 0].sample(
    n=min(250_000, int((train.isFraud == 0).sum())),
    random_state=42
)
fit_data = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

X_train = fit_data[FEATURES]
y_train = fit_data["isFraud"]

print(f"\nTraining on {len(fit_data):,} rows")
print(f"Fraud cases retained: {int(y_train.sum()):,}")

model = RandomForestClassifier(
    n_estimators=80,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=4,
    random_state=42
)

start = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time()-start:.1f} seconds")

# Select threshold using validation only.
print("\nEvaluating validation set...")
val_prob = model.predict_proba(valid[FEATURES])[:, 1]
precision, recall, thresholds = precision_recall_curve(
    valid["isFraud"], val_prob
)
f1 = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-12
)
best = int(np.argmax(f1))
threshold = float(thresholds[best])

print(f"Validation PR-AUC: {average_precision_score(valid['isFraud'], val_prob):.4f}")
print(f"Selected threshold: {threshold:.6f}")
print(f"Validation precision: {precision[best]:.4f}")
print(f"Validation recall: {recall[best]:.4f}")
print(f"Validation F1: {f1[best]:.4f}")

# Final evaluation on untouched chronological test set.
print("\nEvaluating untouched test set...")
test_prob = model.predict_proba(test[FEATURES])[:, 1]
test_pred = (test_prob >= threshold).astype(int)

print(f"Test ROC-AUC: {roc_auc_score(test['isFraud'], test_prob):.4f}")
print(f"Test PR-AUC: {average_precision_score(test['isFraud'], test_prob):.4f}")
print("Confusion matrix:")
print(confusion_matrix(test["isFraud"], test_pred))
print(classification_report(
    test["isFraud"], test_pred,
    target_names=["LEGITIMATE", "FRAUD"],
    zero_division=0
))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/fraud_model_v3.joblib")

with open("models/decision_threshold_v3.json", "w") as f:
    json.dump({
        "threshold": threshold,
        "features": FEATURES,
        "dataset": "PaySim",
        "model_version": "3.0"
    }, f, indent=2)

print("\nV3 MODEL SAVED SUCCESSFULLY")
print("models/fraud_model_v3.joblib")
print("models/decision_threshold_v3.json")
