"""Measure the response times reported in the project document.

Usage: start the API (``uvicorn backend.main:app``) and, from the repository
root, run::

    python scripts/measure_response_times.py [photo_with_a_face.jpg]

For every model it prints the time of the first request (which loads the
model) and the median of the next 20 requests. With a photo it also times
``POST /emotion``. Restart the API before measuring to see the load times.
"""
import statistics
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"
REPEATS = 20


def timed(call) -> float:
    start = time.perf_counter()
    response = call()
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


def main() -> int:
    client = httpx.Client(base_url=BASE_URL, timeout=60)
    models = client.get("/health").json()["models"]
    print(f"{'request':<12}{'first (ms)':>12}{'median (ms)':>13}")
    for name in models:
        def call(name=name):
            return client.post(f"/models/{name}/predict", json={})

        first = timed(call)
        median = statistics.median(timed(call) for _ in range(REPEATS))
        print(f"{name:<12}{first:>12.0f}{median:>13.1f}")

    def command():
        return client.post("/command", json={"text": "JarvisTEC wine quality"})

    median = statistics.median(timed(command) for _ in range(REPEATS))
    print(f"{'command':<12}{'-':>12}{median:>13.1f}")

    if len(sys.argv) > 1:
        image = open(sys.argv[1], "rb").read()

        def emotion():
            return client.post(
                "/emotion", files={"image": ("photo.jpg", image, "image/jpeg")}
            )

        first = timed(emotion)
        median = statistics.median(timed(emotion) for _ in range(REPEATS))
        print(f"{'emotion':<12}{first:>12.0f}{median:>13.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
