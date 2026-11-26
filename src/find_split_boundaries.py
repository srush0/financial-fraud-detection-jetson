import pandas as pd

PATH = "data/paysim.csv"
CHUNK = 250_000

counts = {}

for part in pd.read_csv(
    PATH,
    usecols=["step", "isFraud"],
    chunksize=CHUNK
):
    grouped = part.groupby("step")["isFraud"].agg(
        transactions="count",
        fraud="sum"
    )

    for step, row in grouped.iterrows():
        key = int(step)
        if key not in counts:
            counts[key] = [0, 0]

        counts[key][0] += int(row["transactions"])
        counts[key][1] += int(row["fraud"])

timeline = pd.DataFrame.from_dict(
    counts,
    orient="index",
    columns=["transactions", "fraud"]
).sort_index()

timeline["cumulative"] = timeline["transactions"].cumsum()

total = int(timeline["transactions"].sum())

print("Total transactions:", f"{total:,}")

for fraction in [0.70, 0.85]:
    target = total * fraction
    step = (timeline["cumulative"] - target).abs().idxmin()
    row = timeline.loc[step]

    print(f"\nTarget: {fraction * 100:.0f}%")
    print("Boundary step:", step)
    print("Cumulative transactions:", f"{int(row['cumulative']):,}")
    print("Cumulative percentage:",
          round(row["cumulative"] / total * 100, 3))
    print("Fraud cases through boundary:",
          f"{int(timeline.loc[:step, 'fraud'].sum()):,}")

print("\nSplit boundary audit complete.")
