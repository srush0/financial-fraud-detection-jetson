import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# Load dataset
df = pd.read_csv("data/transactions.csv")

# Cyclical time encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

FEATURES = [
    "amount",
    "hour_sin",
    "hour_cos",
    "frequency",
    "distance"
]

X = df[FEATURES]
y = df["fraud"]

# 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

candidates = {}

for weight in [None, "balanced"]:
    print(f"\nTraining model: class_weight={weight}")

    base_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=5,
        class_weight=weight,
        random_state=42,
        n_jobs=-1
    )

    start = time.perf_counter()
    base_model.fit(X_train, y_train)

    # Calibrate probabilities using training folds only
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3
    )
    calibrated_model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    val_prob = calibrated_model.predict_proba(X_val)[:, 1]

    brier = brier_score_loss(y_val, val_prob)
    pr_auc = average_precision_score(y_val, val_prob)

    print("Training time:", round(training_time, 3), "seconds")
    print("Validation Brier score:", round(brier, 4))
    print("Validation PR-AUC:", round(pr_auc, 4))

    candidates[str(weight)] = {
        "model": calibrated_model,
        "brier": brier,
        "pr_auc": pr_auc,
        "training_time": training_time
    }

# Select using validation Brier score
selected_name = min(
    candidates,
    key=lambda name: candidates[name]["brier"]
)

selected = candidates[selected_name]
model = selected["model"]

# Final evaluation on untouched test data
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)

print("\nSELECTED MODEL:", selected_name)
print("\nFINAL TEST RESULTS")
print("--------------------------")
print("Test Brier score:", round(brier_score_loss(y_test, test_prob), 4))
print("Test ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))
print("Test PR-AUC:", round(average_precision_score(y_test, test_prob), 4))

print("\nClassification Report:")
print(classification_report(y_test, test_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# Save as a separate model. Do not overwrite v1.
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_model_v2.joblib")

print("\nNew model saved to models/fraud_model_v2.joblib")
print("Original model remains unchanged.")
