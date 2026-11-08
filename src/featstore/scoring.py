"""Scoring: fetch online features, build the model input vector, score. Each stage is timed separately."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from .definitions import FEATURE_NAMES, MODEL_INPUTS
from .latency import LatencyTracker
from .online import OnlineStore


class Scorer:
    def __init__(self, store: OnlineStore, artifact: dict, fetch: LatencyTracker | None = None,
                 score: LatencyTracker | None = None):
        self.store = store
        self.model = artifact["model"]
        self.model_id = artifact["model_id"]
        self.threshold = artifact["threshold"]
        self.fetch = fetch or LatencyTracker()
        self.score_t = score or LatencyTracker()

    @classmethod
    def load(cls, data_dir: Path, **kw) -> Scorer:
        return cls(OnlineStore.load(data_dir / "online.sqlite"), joblib.load(data_dir / "model.joblib"), **kw)

    def score(self, req: dict) -> dict:
        """req: card_id, amount, mcc, device_id, lat, lon, cnp, and optional ts (defaults to the stream clock)."""
        ts = int(req.get("ts") or self.store.meta["max_event_ts"])
        request = {**req, "ts": ts}
        t0 = time.perf_counter()
        feats = self.store.get_features(int(req["card_id"]), ts, request)
        t1 = time.perf_counter()
        x = np.array([[feats[n] for n in FEATURE_NAMES] + [float(req["amount"]), float(req["cnp"])]])
        p = float(self.model.predict_proba(x)[0, 1])
        t2 = time.perf_counter()
        self.fetch.observe(t1 - t0)
        self.score_t.observe(t2 - t1)
        return {
            "card_id": int(req["card_id"]),
            "as_of": ts,
            "score": round(p, 6),
            "alert": p >= self.threshold,
            "threshold": round(self.threshold, 6),
            "model_id": self.model_id,
            "features": {k: round(v, 6) for k, v in feats.items()},
            "inputs": MODEL_INPUTS,
            "latency_ms": {"feature_fetch": round((t1 - t0) * 1000, 4), "model_score": round((t2 - t1) * 1000, 4)},
        }
