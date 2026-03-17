import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests import HTTPError


URL_PLAIN = "http://localhost:3000/predict"
URL_BATCHED = "http://localhost:3000/predict_batched"

BASE_TEXTS = [
    "thank you very much",
    "you are amazing",
    "i hate you",
    "go away",
    "have a nice day",
    "you are stupid",
]
TEXTS = [(text + " ") * 40 for text in BASE_TEXTS] * 40

CONCURRENCY = 32
TIMEOUT = 60
WARMUP_REQUESTS = 20
MAX_RETRIES = 3
RETRY_DELAY_SEC = 0.2


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = int((len(ordered) - 1) * p)
    return ordered[index]


def post_with_retries(url: str, payload: dict) -> tuple[float, dict]:
    last_error = None
    for attempt in range(MAX_RETRIES):
        start = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            latency = time.perf_counter() - start
            return latency, response.json()
        except HTTPError as error:
            last_error = error
            status_code = (
                error.response.status_code if error.response is not None else None
            )
            if status_code != 503 or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY_SEC)
    raise last_error  # pragma: no cover


def call_plain(text: str) -> tuple[float, dict]:
    return post_with_retries(URL_PLAIN, {"text": text})


def call_batched(text: str) -> tuple[float, dict]:
    latency, data = post_with_retries(URL_BATCHED, {"texts": [text]})
    if isinstance(data, list):
        return latency, data[0]
    if isinstance(data, dict) and "texts" in data:
        return latency, data["texts"][0]
    return latency, data


def warmup(name: str, fn) -> None:
    print(f"Warming up {name} endpoint...")
    for text in TEXTS[:WARMUP_REQUESTS]:
        fn(text)


def run_test(name: str, fn) -> None:
    latencies = []
    results = []
    failures = 0
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(fn, text) for text in TEXTS]
        for future in as_completed(futures):
            try:
                latency, result = future.result()
                latencies.append(latency)
                results.append(result)
            except requests.RequestException:
                failures += 1

    elapsed = time.perf_counter() - started
    successful_requests = len(latencies)
    rps = successful_requests / elapsed if elapsed > 0 else 0.0

    print(f"{name}:")
    print(f"  total requests = {len(TEXTS)}")
    print(f"  successful     = {successful_requests}")
    print(f"  failed         = {failures}")
    print(f"  concurrency    = {CONCURRENCY}")
    print(f"  total time     = {elapsed:.3f} sec")
    print(f"  throughput     = {rps:.2f} req/sec")
    if latencies:
        print(f"  avg latency    = {statistics.mean(latencies) * 1000:.1f} ms")
        print(f"  p50 latency    = {percentile(latencies, 0.50) * 1000:.1f} ms")
        print(f"  p95 latency    = {percentile(latencies, 0.95) * 1000:.1f} ms")
        print(f"  sample result  = {results[0]}")
    print()


if __name__ == "__main__":
    warmup("plain", call_plain)
    warmup("batched", call_batched)

    print("Benchmarking plain endpoint...")
    run_test("plain", call_plain)

    print("Benchmarking batched endpoint...")
    run_test("batched", call_batched)
