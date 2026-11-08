from __future__ import annotations

import time

from featstore.latency import BUCKETS, LatencyTracker, Timer
from featstore.metrics import Metrics
from featstore.scoring import Scorer


def test_tracker_buckets_and_cumulative():
    t = LatencyTracker()
    for s in (0.00005, 0.0003, 0.0003, 0.002, 5.0):
        t.observe(s)
    cum = t.cumulative()
    assert cum[0] == 1 and cum[BUCKETS.index(0.0005)] == 3 and cum[-1] == 4
    assert t.total == 5


def test_tracker_percentiles():
    t = LatencyTracker()
    for i in range(1, 101):
        t.observe(i / 1000)
    s = t.summary()
    assert abs(s["p50_ms"] - 50.5) < 0.6 and s["p99_ms"] > 98 and s["count"] == 100


def test_empty_summary():
    assert LatencyTracker().summary()["p50_ms"] is None


def test_timer_records_elapsed():
    t = LatencyTracker()
    with Timer(t) as tm:
        time.sleep(0.002)
    assert t.total == 1 and tm.elapsed >= 0.002


def test_metrics_render_histogram_per_stage():
    m = Metrics()
    m.latency["feature_fetch"].observe(0.0002)
    m.observe_request("/score", 200)
    text = m.render()
    assert 'featstore_stage_latency_seconds_bucket{stage="feature_fetch",le="0.00025"} 1' in text
    assert 'featstore_stage_latency_seconds_count{stage="model_score"} 0' in text
    assert 'featstore_http_requests_total{route="/score",status="200"} 1' in text


def test_scorer_times_fetch_and_score_separately(built):
    d, _ = built
    sc = Scorer.load(d)
    card = next(iter(sc.store.cards))
    out = sc.score({"card_id": card, "amount": 42.0, "mcc": 5411, "device_id": card * 4, "lat": 40, "lon": -100,
                    "cnp": 0})
    assert sc.fetch.total == 1 and sc.score_t.total == 1
    assert out["latency_ms"]["feature_fetch"] >= 0 and 0 <= out["score"] <= 1


def test_benchmark_reports_percentiles(built):
    _, report = built
    lat = report["latency"]
    assert lat["requests"] > 50
    for stage in ("feature_fetch", "model_score", "total"):
        assert lat[stage]["p50_ms"] <= lat[stage]["p99_ms"]
    assert lat["feature_fetch"]["p99_ms"] < 5
