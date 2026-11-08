"""Command line: build the stores and model, replay the stream, run the skew gate, serve the API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .logs import get_logger, log_event


def _data_dir(a) -> Path:
    return Path(a.data_dir or os.environ.get("FEATSTORE_DATA_DIR", "data"))


def cmd_build(a) -> int:
    from .pipeline import build

    r = build(_data_dir(a), a.cards, a.days, a.seed)
    e = r["evaluation"]
    print(json.dumps({
        "events": r["dataset"]["events"], "fraud_events": r["dataset"]["fraud_events"],
        "auc_point_in_time": round(e["pit"]["auc"], 4), "auc_leaky_offline": round(e["leaky_offline"]["auc"], 4),
        "auc_leaky_in_production": round(e["leaky_in_production"]["auc"], 4),
        "skew_bug_off": r["skew"]["bug_off"]["worst_mismatch_rate"],
        "skew_bug_on": r["skew"]["bug_on"]["worst_mismatch_rate"],
        "fetch_p99_ms": r["latency"]["feature_fetch"]["p99_ms"], "score_p99_ms": r["latency"]["model_score"]["p99_ms"],
        "timings_s": r["timings_s"],
    }, indent=1))
    return 0


def cmd_stream(a) -> int:
    from .offline import OfflineStore
    from .pipeline import run_stream

    d = _data_dir(a)
    off = OfflineStore(d / "offline.duckdb", read_only=True)
    store = run_stream(off, d / "online.sqlite", snapshot_every=a.snapshot_every, limit=a.limit)
    off.close()
    log_event(get_logger(), "stream done", **store.meta)
    return 0


def cmd_skew(a) -> int:
    from .offline import OfflineStore
    from .online import OnlineStore
    from .skew import skew_report

    d = _data_dir(a)
    off = OfflineStore(d / "offline.duckdb", read_only=True)
    store = OnlineStore.load(d / "online.sqlite")
    r = skew_report(store, off, sample_cards=a.sample, window_bug=a.bug, tolerance=a.tolerance)
    off.close()
    for f in r["features"]:
        print(f"{f['feature']:<24} compared {f['compared']:>5}  mismatches {f['mismatches']:>5}  "
              f"rate {f['mismatch_rate']:.4f}  {'ok' if f['ok'] else 'SKEW'}")
    print(f"window_bug={a.bug} worst mismatch rate {r['worst_mismatch_rate']:.4f}, tolerance {a.tolerance}")
    if a.expect_clean and not r["passed"]:
        print("FAIL: online/offline skew above tolerance", file=sys.stderr)
        return 1
    if a.expect_detected and r["passed"]:
        print("FAIL: the planted window bug was not detected", file=sys.stderr)
        return 1
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    from .app import create_app

    d = _data_dir(a)
    if not (d / "report.json").exists():
        if not a.bootstrap:
            print(f"no build in {d}; run `featstore build` or pass --bootstrap", file=sys.stderr)
            return 1
        from .pipeline import build

        build(d, int(os.environ.get("FEATSTORE_CARDS", 50_000)), int(os.environ.get("FEATSTORE_DAYS", 60)))
    host = a.host or os.environ.get("HOST", "127.0.0.1")
    port = a.port or int(os.environ.get("PORT", 8858))
    uvicorn.run(create_app(d), host=host, port=port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="featstore")
    p.add_argument("--data-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate events, build both stores, train, check skew, benchmark latency")
    b.add_argument("--cards", type=int, default=int(os.environ.get("FEATSTORE_CARDS", 50_000)))
    b.add_argument("--days", type=int, default=int(os.environ.get("FEATSTORE_DAYS", 60)))
    b.add_argument("--seed", type=int, default=7)
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("stream", help="replay the event log through the stream processor into the online store")
    s.add_argument("--snapshot-every", type=int, default=500_000)
    s.add_argument("--limit", type=int, default=None, help="stop early, to see freshness go red")
    s.set_defaults(fn=cmd_stream)
    k = sub.add_parser("skew", help="compare served and recomputed feature values")
    k.add_argument("--bug", action="store_true", help="turn on the planted off-by-one window bug")
    k.add_argument("--sample", type=int, default=300)
    k.add_argument("--tolerance", type=float, default=0.0, help="allowed mismatch rate per feature")
    k.add_argument("--expect-clean", action="store_true")
    k.add_argument("--expect-detected", action="store_true")
    k.set_defaults(fn=cmd_skew)
    v = sub.add_parser("serve", help="run the scoring API and UI")
    v.add_argument("--host", default=None)
    v.add_argument("--port", type=int, default=None)
    v.add_argument("--bootstrap", action="store_true", help="build first if the data directory is empty")
    v.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
