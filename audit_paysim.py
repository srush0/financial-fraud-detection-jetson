import pandas as pd
from collections import Counter

PATH = "data/paysim.csv"
CHUNK_SIZE = 250_000

total_rows = 0
fraud_counts = Counter()
type_counts = Counter()
type_fraud = Counter()
flag_counts = Counter()
missing_counts = Counter()
step_min = float("inf")
step_max = float("-inf")
amount_min = float("inf")
amount_max = float("-inf")
fraud_amounts = []

print("Starting PaySim dataset audit...")

for chunk in pd.read_csv(PATH, chunksize=CHUNK_SIZE):
    total_rows += len(chunk)

    fraud_counts.update(chunk["isFraud"].value_counts().to_dict())
    type_counts.update(chunk["type"].value_counts().to_dict())

    grouped = chunk.groupby(["type", "isFraud"]).size()
    for (tx_type, label), count in grouped.items():
        type_fraud[(tx_type, int(label))] += int(count)

    flag_counts.update(
        chunk["isFlaggedFraud"].value_counts().to_dict()
    )

    missing_counts.update(
        chunk.isna().sum().to_dict()
    )

    step_min = min(step_min, chunk["step"].min())
    step_max = max(step_max, chunk["step"].max())

    amount_min = min(amount_min, chunk["amount"].min())
    amount_max = max(amount_max, chunk["amount"].max())

    fraud_amounts.extend(
        chunk.loc[chunk["isFraud"] == 1, "amount"].tolist()
    )

    print(f"Processed {total_rows:,} transactions")

fraud_total = fraud_counts.get(1, 0)
legit_total = fraud_counts.get(0, 0)

print("\n========== PAYSIM AUDIT RESULTS ==========")
print("Total transactions:", f"{total_rows:,}")
print("Fraud transactions:", f"{fraud_total:,}")
print("Legitimate transactions:", f"{legit_total:,}")
print("Fraud percentage:", round(fraud_total / total_rows * 100, 5), "%")

print("\nTransaction type distribution:")
for tx_type, count in type_counts.most_common():
    print(f"{tx_type}: {count:,}")

print("\nFraud distribution by transaction type:")
for tx_type in sorted(type_counts):
    fraud = type_fraud[(tx_type, 1)]
    legitimate = type_fraud[(tx_type, 0)]
    print(
        f"{tx_type}: fraud={fraud:,}, "
        f"legitimate={legitimate:,}"
    )

print("\nFlagged fraud distribution:", dict(flag_counts))
print("\nMissing values:", dict(missing_counts))
print("\nStep range:", step_min, "to", step_max)
print("Amount range:", amount_min, "to", amount_max)

if fraud_amounts:
    print("\nFraud amount statistics:")
    print(pd.Series(fraud_amounts).describe())

print("\n========== AUDIT COMPLETE ==========")
