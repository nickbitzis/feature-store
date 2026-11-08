"""Prometheus text exposition without extra dependencies."""

from __future__ import annotations

import threading
from collections import defaultdict

from .latency import BUCKETS, LatencyTracker


def _labels(d: dict) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(d.items())) + "}" if d else ""


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests: dict[tuple[str, str], int] = defaultdict(int)
        self.latency = {"feature_fetch": LatencyTracker(), "model_score": LatencyTracker(), "request": LatencyTracker()}
        self.gauges: dict[str, tuple[str, str, dict, float]] = {}

    def observe_request(self, route: str, status: int) -> None:
        with self.lock:
            self.requests[(route, str(status))] += 1

    def set_gauge(self, key: str, name: str, help_text: str, labels: dict, value: float) -> None:
        with self.lock:
            self.gauges[key] = (name, help_text, labels, value)

    def render(self) -> str:
        out = ["# HELP featstore_http_requests_total HTTP requests by route and status",
               "# TYPE featstore_http_requests_total counter"]
        with self.lock:
            out += [f"featstore_http_requests_total{_labels({'route': r, 'status': s})} {n}"
                    for (r, s), n in sorted(self.requests.items())]
            gauges = sorted(self.gauges.values(), key=lambda g: (g[0], str(g[2])))
        out += ["# HELP featstore_stage_latency_seconds Latency of feature fetch, model scoring, and the whole request",
                "# TYPE featstore_stage_latency_seconds histogram"]
        for stage, tr in self.latency.items():
            cum = tr.cumulative()
            for b, n in zip(BUCKETS, cum, strict=True):
                out.append(f'featstore_stage_latency_seconds_bucket{{stage="{stage}",le="{b}"}} {n}')
            out.append(f'featstore_stage_latency_seconds_bucket{{stage="{stage}",le="+Inf"}} {tr.total}')
            out.append(f'featstore_stage_latency_seconds_sum{{stage="{stage}"}} {tr.sum:.6f}')
            out.append(f'featstore_stage_latency_seconds_count{{stage="{stage}"}} {tr.total}')
        seen = set()
        for name, help_text, labels, value in gauges:
            if name not in seen:
                out += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
                seen.add(name)
            out.append(f"{name}{_labels(labels)} {value}")
        return "\n".join(out) + "\n"
