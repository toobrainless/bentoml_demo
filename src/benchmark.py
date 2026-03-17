import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


URL_PLAIN = "http://localhost:3000/predict"
URL_BATCHED = "http://localhost:3000/predict_batched"

TEXTS = [
    "thank you very much",
    "you are amazing",
    "i hate you",
    "go away",
    "have a nice day",
    "you are stupid",
] * 20

CONCURRENCY = 16


def call_plain(text: str) -> dict:
    response = requests.post(URL_PLAIN, json={"text": text}, timeout=30)
    response.raise_for_status()
    return response.json()


def call_batched(text: str) -> dict:
    response = requests.post(URL_BATCHED, json={"texts": [text]}, timeout=30)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return data[0]
    if isinstance(data, dict) and "texts" in data:
        return data["texts"][0]
    return data


def run_test(name: str, fn) -> None:
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(fn, text) for text in TEXTS]
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - start
    rps = len(TEXTS) / elapsed

    print(f"{name}:")
    print(f"  total requests = {len(TEXTS)}")
    print(f"  total time     = {elapsed:.3f} sec")
    print(f"  throughput     = {rps:.2f} req/sec")
    print(f"  sample result  = {results[0]}")
    print()


if __name__ == "__main__":
    print("Benchmarking plain endpoint...")
    run_test("plain", call_plain)

    print("Benchmarking batched endpoint...")
    run_test("batched", call_batched)