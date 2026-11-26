import pandas as pd

path = "data/paysim.csv"
chunk_size = 250_000

timeline = {}
previous_step = -1
ordered = True
total = 0

for chunk in pd.read_csv(
    path,
    usecols=["step", "isFraud"],
    chunksize=chunk_size
):
    steps = chunk["step"].to_numpy()

    if len(steps):
        if steps[0] < previous_step:
            ordered = False
        if (steps[1:] < steps[:-1]).any():
            ordered = False
        previous_step = steps[-1]

    grouped = chunk.groupby("step")["isFraud"].agg(
        transactions="count",
        fraud="sum"
    )

    for step, row in grouped.iterrows():
        key = int(step)
        if key not in timeline:
            timeline[key] = [0, 0]
        timeline[key][0] += int(row["transactions"])
        timeline[key][1] += int(row["fraud"])

    total += len(chunk)

df = pd.DataFrame.from_dict(
    timeline,
    orient="index",
    columns=["transactions", "fraud"]
).sort_index()

print("Total rows:", f"{total:,}")
print("Steps:", len(df))
print("Rows sorted by step:", ordered)

print("\nTimeline summary:")
ranges = [
    ("First 70%", 1, 520),
    ("Next 15%", 521, 631),
    ("Final 15%", 632, 743)
]

for name, start, end in ranges:
    part = df.loc[start:end]
    tx = int(part["transactions"].sum())
    fraud = int(part["fraud"].sum())
    print(
        f"{name} (steps {start}-{end}): "
        f"{tx:,} transactions, {fraud:,} fraud"
    )

print("\nFraud counts by 100-step interval:")
for start in range(1, 744, 100):
    part = df.loc[start:min(start + 99, 743)]
    print(
        f"Steps {start}-{min(start + 99, 743)}: "
        f"{int(part['fraud'].sum()):,} fraud"
    )

print("\nTimeline audit complete.")
