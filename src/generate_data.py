import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000

amount = np.random.exponential(1500, N)
hour = np.random.randint(0, 24, N)
frequency = np.random.poisson(3, N)
distance = np.random.exponential(10, N)

fraud_probability = (
    0.02
    + 0.25 * (amount > 5000)
    + 0.20 * (hour < 5)
    + 0.20 * (frequency > 7)
    + 0.20 * (distance > 30)
)

fraud_probability = np.clip(fraud_probability, 0, 0.95)
fraud = np.random.binomial(1, fraud_probability)

df = pd.DataFrame({
    "amount": amount.round(2),
    "hour": hour,
    "frequency": frequency,
    "distance": distance.round(2),
    "fraud": fraud
})

df.to_csv("data/transactions.csv", index=False)

print("Dataset generated successfully!")
print("Total transactions:", len(df))
print("\nClass distribution:")
print(df["fraud"].value_counts())
print("\nDataset preview:")
print(df.head())
