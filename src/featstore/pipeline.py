"""End-to-end build: generate events, load the offline store, run the stream processor into the online
store, build point-in-time and leaky training sets, train and evaluate, check skew, and benchmark latency.
Everything lands in one data directory that the API serves from."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import model as fm
from .definitions import DAY, FEATURE_NAMES, MODEL_INPUTS
from .generate import generate
from .logs import get_logger, log_event
from .offline import OfflineStore
from .online import OnlineStore, StreamProcessor
from .registry import Registry, freshness
from .scoring import Scorer
from .skew import skew_report

MODEL_ID = "fraud-gbm-v1"


def run_stream(offline: OfflineStore, out: Path | None = None, snapshot_every: int = 0,
               limit: int | None = None) -> OnlineStore:
    """Replay the event log in arrival order through the stream processor."""
    store = OnlineStore()
    max_ts = offline.scalar("SELECT max(ts) FROM events")
    day_start = (int(max_ts) // DAY) * DAY
    store.load_tables(offline.con.execute("SELECT card_id, home_lat, home_lon FROM cards").fetchall(),
                      offline.con.execute("SELECT mcc, risk FROM mcc_risk").fetchall(), as_of=day_start)
    proc = StreamProcessor(store)
    q = "SELECT card_id, ts, amount, merchant_id, cnp, device_id, arrival_ts FROM events ORDER BY arrival_ts, event_id"
    if limit:
        q += f" LIMIT {int(limit)}"
    cur = offline.con.execute(q)
    n = 0
    while True:
        batch = cur.fetchmany(100_000)
        if not batch:
            break
        proc.run(batch)
        n += len(batch)
        if out and snapshot_every and n % snapshot_every < len(batch):
            store.save(out)
    if out:
        store.save(out)
    return store


def latency_benchmark(scorer: Scorer, offline: OfflineStore, n: int = 2000, seed: int = 0) -> dict:
    wm = scorer.store.meta["max_event_ts"]
    pool = offline.df("SELECT card_id, ts, amount, mcc, device_id, lat, lon, cnp FROM events WHERE ts >= ?",
                      [wm - DAY])
    reqs = pool.sample(min(len(pool), n + 100), random_state=seed).to_dict("records")
    for r in reqs[:100]:
        scorer.score(r)  # warm-up
    fetch, score = [], []
    for r in reqs[100:]:
        out = scorer.score(r)
        fetch.append(out["latency_ms"]["feature_fetch"])
        score.append(out["latency_ms"]["model_score"])
    f, s, t = np.array(fetch), np.array(score), np.array(fetch) + np.array(score)

    def pct(a):
        return {"p50_ms": round(float(np.percentile(a, 50)), 4), "p99_ms": round(float(np.percentile(a, 99)), 4),
                "mean_ms": round(float(a.mean()), 4)}

    hist_edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 1e9]
    return {"requests": len(f), "feature_fetch": pct(f), "model_score": pct(s), "total": pct(t),
            "histogram_ms": {"edges": hist_edges[:-1],
                             "feature_fetch": np.histogram(f, hist_edges)[0].tolist(),
                             "model_score": np.histogram(s, hist_edges)[0].tolist()}}


def build(data_dir: Path, n_cards: int = 50_000, days: int = 60, seed: int = 7, skew_cards: int = 400,
          latency_requests: int = 2000) -> dict:
    log = get_logger()
    data_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t = time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round(now - t, 2)
        log_event(log, "stage done", stage=name, seconds=timings[name])
        t = now

    ds = generate(n_cards, days, seed)
    lap("generate")
    db = data_dir / "offline.duckdb"
    for p in (db, Path(str(db) + ".wal")):
        p.unlink(missing_ok=True)
    off = OfflineStore(db)
    off.load(ds)
    lap("offline_load")
    store = run_stream(off, data_dir / "online.sqlite")
    lap("stream")

    bounds = fm.TimeSplit().bounds(ds.start_ts, days)
    off.build_training_set("pit")
    off.build_training_set("leaky")
    lap("training_sets")
    pit = off.df("SELECT * FROM train_pit")
    leaky = off.df("SELECT * FROM train_leaky")
    pit_tr, pit_te = fm.split(pit, bounds)
    lk_tr, lk_te = fm.split(leaky, bounds)
    model = fm.train(pit_tr, seed=seed)
    leaky_model = fm.train(lk_tr, seed=seed)
    lap("train")
    ev_pit = fm.evaluate(model, pit_te)
    ev_leak_offline = fm.evaluate(leaky_model, lk_te)
    ev_leak_online = fm.evaluate(leaky_model, pit_te)
    importance = fm.feature_importance(model, pit_te, seed=seed)
    by_type = {}
    p = model.predict_proba(pit_te[MODEL_INPUTS].to_numpy(dtype=float))[:, 1]
    k = max(1, int(round(len(p) * fm.ALERT_RATE)))
    threshold = float(np.sort(p)[::-1][k - 1])
    alerted = p >= threshold
    for ft in ("card_testing", "account_takeover", "merchant_compromise"):
        m = (pit_te.fraud_type == ft).to_numpy()
        by_type[ft] = {"rows": int(m.sum()), "recall_at_alert_rate": float(alerted[m].mean()) if m.any() else None}
    lap("evaluate")

    reg = Registry(data_dir / "registry.json")
    reg.register()
    refs = reg.feature_refs(FEATURE_NAMES)
    joblib.dump({"model": model, "model_id": MODEL_ID, "inputs": MODEL_INPUTS, "feature_refs": refs,
                 "threshold": threshold}, data_dir / "model.joblib")
    reg.register_model(MODEL_ID, refs, {
        "trained_at": time.time(), "algorithm": "HistGradientBoostingClassifier", "training_set": "train_pit",
        "train_window": bounds["train"], "test_window": bounds["test"], "auc": ev_pit["auc"],
    })
    reg.save()

    skew_off = skew_report(store, off, sample_cards=skew_cards, seed=seed, window_bug=False)
    skew_on = skew_report(store, off, sample_cards=skew_cards, seed=seed, window_bug=True)
    lap("skew")
    scorer = Scorer(store, joblib.load(data_dir / "model.joblib"))
    latency = latency_benchmark(scorer, off, n=latency_requests, seed=seed)
    lap("latency")

    ev = ds.events
    report = {
        "dataset": {
            "cards": n_cards, "days": days, "seed": seed, "events": int(len(ev)),
            "fraud_events": int(ev.is_fraud.sum()), "fraud_rate": float(ev.is_fraud.mean()),
            "fraud_by_type": ev[ev.is_fraud].fraud_type.value_counts().to_dict(),
            "start_ts": ds.start_ts, "end_ts": int(ev.ts.max()),
            "late_events": int((ev.arrival_ts - ev.ts > 60).sum()),
        },
        "stream": store.meta,
        "split": {k: list(v) for k, v in bounds.items()},
        "evaluation": {
            "pit": ev_pit,
            "leaky_offline": ev_leak_offline,
            "leaky_in_production": ev_leak_online,
            "by_fraud_type": by_type,
            "importance": importance,
            "threshold": threshold,
            "train_rows": int(len(pit_tr)), "test_rows": int(len(pit_te)),
        },
        "skew": {"bug_off": skew_off, "bug_on": skew_on},
        "latency": latency,
        "freshness": freshness(store.meta, int(off.scalar("SELECT max(ts) FROM events"))),
        "timings_s": timings,
    }
    (data_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    off.close()
    log_event(log, "build done", events=len(ev), auc_pit=ev_pit["auc"], auc_leaky_offline=ev_leak_offline["auc"],
              auc_leaky_prod=ev_leak_online["auc"], skew_off=skew_off["worst_mismatch_rate"],
              skew_on=skew_on["worst_mismatch_rate"], p99_ms=latency["total"]["p99_ms"])
    return report
