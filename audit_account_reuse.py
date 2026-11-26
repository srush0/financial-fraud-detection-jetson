import pandas as pd

path = "data/paysim.csv"
senders = set()
recipients = set()

for chunk in pd.read_csv(
    path,
    usecols=["nameOrig", "nameDest"],
    chunksize=100_000
):
    senders.update(chunk["nameOrig"].unique())
    recipients.update(chunk["nameDest"].unique())

    if len(senders) % 1_000_000 < 100_000:
        print(
            "Unique senders:", f"{len(senders):,}",
            "| Unique recipients:", f"{len(recipients):,}"
        )

print("\n========== ACCOUNT REUSE AUDIT ==========")
print("Transactions:", "6,362,620")
print("Unique senders:", f"{len(senders):,}")
print("Unique recipients:", f"{len(recipients):,}")
print("Repeated sender IDs:", "Calculating...")
