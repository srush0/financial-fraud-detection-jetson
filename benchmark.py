import time
import requests
import statistics

URL = "http://localhost:8000/predict"

payload = {
    "amount": 8500,
    "hour": 2,
    "frequency": 10,
    "distance": 45
}

latencies = []

print("Running 100 inference requests...")

for i in range(100):
    start = time.perf_counter()

    response = requests.post(
        URL,
        json=payload,
        timeout=10
    )

    response.raise_for_status()

    elapsed = (time.perf_counter() - start) * 1000
    latencies.append(elapsed)

print("\nEDGE PERFORMANCE REPORT")
print("-------------------------")
print("Total requests:", len(latencies))
print("Average latency:", round(statistics.mean(latencies), 2), "ms")
print("Minimum latency:", round(min(latencies), 2), "ms")
print("Maximum latency:", round(max(latencies), 2), "ms")
print("P95 latency:", round(sorted(latencies)[94], 2), "ms")
print("Median latency:", round(statistics.median(latencies), 2), "ms")
print("\nAll requests completed successfully.")
