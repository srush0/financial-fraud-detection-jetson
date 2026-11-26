import pandas as pd
import numpy as np
from pathlib import Path

SOURCE = Path("data/paysim.csv")
OUTPUT = Path("data/v3")
CHUNK_SIZE = 100_000

OUTPUT.mkdir(parents=True, exist_ok=True)

TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

def account_category(value):
    if isinstance(value, str):
        return 1 if value.startswith("C") else 0
    return 0

def transform(chunk):
    result = pd.DataFrame()

    result["step"] = chunk["step"].astype("int16")
    result["amount"] = chunk["amount"].astype("float32")
    result["log_amount"] = np.log1p(chunk["amount"]).astype("float32")

    result["transaction_type"] = (
        chunk["type"].map(TYPE_MAP).astype("int8")
    )

    result["sender_category"] = (
        chunk["nameOrig"].map(account_category).astype("int8")
    )

    result["recipient_category"] = (
        chunk["nameDest"].map(account_category).astype("int8")
    )

    result["isFraud"] = chunk["isFraud"].astype("int8")

    result["split"] = np.select(
        [
            result["step"] <= 323,
            result["step"] <= 377
        ],
        [
            "train",
            "validation"
        ],
        default="test"
    )

    return result

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    for split in ["train", "validation", "test"]:
        folder = OUTPUT / split
        folder.mkdir(exist_ok=True)

        for old_file in folder.glob("*.parquet"):
            old_file.unlink()

    total = 0
    fraud = 0

    print("Starting V3 feature extraction...")

    for index, chunk in enumerate(
        pd.read_csv(
            SOURCE,
            usecols=[
                "step",
                "type",
                "amount",
                "nameOrig",
                "nameDest",
                "isFraud"
            ],
            chunksize=CHUNK_SIZE
        )
    ):
        features = transform(chunk)

        for split in ["train", "validation", "test"]:
            part = features[features["split"] == split].drop(
                columns=["split"]
            )

            if not part.empty:
                output_file = OUTPUT / split / f"part_{index:04d}.parquet"
                part.to_parquet(output_file, index=False)

        total += len(features)
        fraud += int(features["isFraud"].sum())

        if (index + 1) % 5 == 0:
            print(
                f"Processed {total:,} rows | "
                f"Fraud cases: {fraud:,}"
            )

    print("\nFeature extraction complete.")
    print("Transactions:", f"{total:,}")
    print("Fraud cases:", f"{fraud:,}")
    print("Output:", OUTPUT)

if __name__ == "__main__":
    main()
