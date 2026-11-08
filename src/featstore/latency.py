"""Latency measurement: a fixed-bucket histogram plus exact percentiles from a bounded sample."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

# Seconds. Fine resolution below 10 ms because feature fetch and scoring are sub-millisecond to a few ms.
BUCKETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0)


class LatencyTracker:
    def __init__(self, keep: int = 20_000):
        self.lock = threading.Lock()
        self.counts = [0] * len(BUCKETS)
        self.total = 0
        self.sum = 0.0
        self.recent: deque[float] = deque(maxlen=keep)

    def observe(self, seconds: float) -> None:
        with self.lock:
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    self.counts[i] += 1
                    break
            self.total += 1
            self.sum += seconds
            self.recent.append(seconds)

    def cumulative(self) -> list[int]:
        out, run = [], 0
        for c in self.counts:
            run += c
            out.append(run)
        return out

    def summary(self) -> dict:
        with self.lock:
            xs = np.array(self.recent) * 1000
        if len(xs) == 0:
            return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
        return {"count": int(len(xs)), "p50_ms": round(float(np.percentile(xs, 50)), 4),
                "p95_ms": round(float(np.percentile(xs, 95)), 4), "p99_ms": round(float(np.percentile(xs, 99)), 4),
                "max_ms": round(float(xs.max()), 4)}


class Timer:
    """with Timer(tracker): ... records the elapsed time on exit."""

    def __init__(self, tracker: LatencyTracker):
        self.tracker = tracker
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        self.tracker.observe(self.elapsed)
        return False
