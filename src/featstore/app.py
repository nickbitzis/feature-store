"""FastAPI service: scoring, feature lookup, registry, lineage, skew, freshness, evaluation, latency."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .definitions import FEATURE_NAMES, FEATURES
from .logs import get_logger, log_event
from .metrics import Metrics
from .offline import OfflineStore
from .registry import Registry, freshness
from .scoring import Scorer
from .skew import skew_report

STATIC = Path(__file__).parent / "static"


class Transaction(BaseModel):
    card_id: int = Field(ge=0)
    amount: float = Field(gt=0)
    mcc: int
    device_id: int
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    cnp: int = Field(ge=0, le=1)
    ts: int | None = None


class State:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.metrics = Metrics()
        self.scorer = Scorer.load(data_dir, fetch=self.metrics.latency["feature_fetch"],
                                  score=self.metrics.latency["model_score"])
        self.registry = Registry(data_dir / "registry.json")
        self.report = json.loads((data_dir / "report.json").read_text())
        self.offline = OfflineStore(data_dir / "offline.duckdb", read_only=True)
        self.db_lock = threading.Lock()
        self.high_water = int(self.offline.scalar("SELECT max(ts) FROM events"))
        stale = self.registry.check_model(self.scorer.model_id)
        if stale:
            raise RuntimeError(f"model {self.scorer.model_id} was trained on old versions of {stale}")

    def df(self, sql: str, params: list | None = None):
        with self.db_lock:
            return self.offline.df(sql, params)


def create_app(data_dir: Path) -> FastAPI:
    app = FastAPI(title="Feature Store", version="1.0.0")
    st = State(data_dir)
    app.state.fs = st
    log = get_logger()
    meta = st.scorer.store.meta
    st.metrics.set_gauge("events", "featstore_stream_events_processed", "Events applied by the stream processor",
                         {}, meta["processed"])
    st.metrics.set_gauge("late", "featstore_stream_out_of_order_events", "Events applied out of event-time order",
                         {}, meta["out_of_order"])
    st.metrics.set_gauge("dropped", "featstore_stream_dropped_late_events", "Events dropped after the watermark",
                         {}, meta["dropped_late"])

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        st.metrics.observe_request(path, response.status_code)
        if path == "/score":
            st.metrics.latency["request"].observe(dt)
        response.headers["x-request-id"] = rid
        if path not in ("/healthz", "/metrics"):
            log_event(log, "request", request_id=rid, method=request.method, path=path,
                      status=response.status_code, ms=round(dt * 1000, 3))
        return response

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_id": st.scorer.model_id, "cards_in_online_store": len(st.scorer.store.cards),
                "events_processed": st.scorer.store.meta["processed"]}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        for f in freshness(st.scorer.store.meta, st.high_water):
            st.metrics.set_gauge(f"fresh:{f['feature']}", "featstore_feature_freshness_lag_seconds",
                                 "How far each feature trails its source, simulated seconds", {"feature": f["feature"]},
                                 f["lag_s"])
        return st.metrics.render()

    @app.post("/score")
    def score(txn: Transaction):
        out = st.scorer.score(txn.model_dump())
        log_event(log, "scored", card_id=txn.card_id, score=out["score"], alert=out["alert"])
        return out

    @app.get("/features/{card_id}")
    def features(card_id: int, as_of: int | None = None, device_id: int = -1, mcc: int = 0,
                 lat: float | None = None, lon: float | None = None):
        store = st.scorer.store
        if card_id not in store.home:
            raise HTTPException(404, "unknown card")
        ts = as_of or int(store.meta["max_event_ts"])
        home = store.home[card_id]
        req = {"ts": ts, "device_id": device_id, "mcc": mcc, "lat": home[0] if lat is None else lat,
               "lon": home[1] if lon is None else lon}
        t0 = time.perf_counter()
        vals = store.get_features(card_id, ts, req)
        ms = (time.perf_counter() - t0) * 1000
        return {"card_id": card_id, "as_of": ts, "features": vals, "fetch_ms": round(ms, 4),
                "versions": st.registry.feature_refs(FEATURE_NAMES)}

    @app.get("/registry")
    def registry():
        return {"features": st.registry.listing(), "models": st.registry.data["models"]}

    @app.get("/registry/{name}")
    def registry_feature(name: str):
        if name not in st.registry.data["features"]:
            raise HTTPException(404, "unknown feature")
        return {"feature": name, "versions": st.registry.history(name)}

    @app.get("/lineage")
    def lineage():
        return st.registry.lineage()

    @app.get("/skew")
    def skew(bug: bool = False, sample: int = 200, live: bool = False):
        if not live:
            return st.report["skew"]["bug_on" if bug else "bug_off"]
        with st.db_lock:
            return skew_report(st.scorer.store, st.offline, sample_cards=min(sample, 2000), window_bug=bug)

    @app.get("/freshness")
    def fresh():
        return {"source_high_water_ts": st.high_water, "online_max_event_ts": st.scorer.store.meta["max_event_ts"],
                "features": freshness(st.scorer.store.meta, st.high_water)}

    @app.get("/evaluation")
    def evaluation():
        return {**st.report["evaluation"], "split": st.report["split"], "dataset": st.report["dataset"]}

    @app.get("/latency")
    def latency():
        return {"benchmark": st.report["latency"],
                "live": {k: v.summary() for k, v in st.metrics.latency.items()}}

    @app.get("/report")
    def report():
        return st.report

    @app.get("/cards/examples")
    def examples():
        df = st.df("""SELECT fraud_type, card_id, count(*) AS n FROM events WHERE is_fraud
                      GROUP BY 1, 2 ORDER BY 1, 3 DESC, 2""")
        out = {ft: g.card_id.head(4).tolist() for ft, g in df.groupby("fraud_type")}
        out["normal"] = st.df("""SELECT card_id FROM events GROUP BY card_id HAVING NOT bool_or(is_fraud)
                                 ORDER BY count(*) DESC, card_id LIMIT 4""").card_id.tolist()
        return out

    @app.get("/cards/{card_id}/timeline")
    def timeline(card_id: int, limit: int = 400):
        cols = ", ".join(FEATURE_NAMES)
        df = st.df(f"""SELECT row_id AS event_id, ts, amount, cnp, is_fraud, fraud_type, {cols}
                       FROM train_pit WHERE card_id = ? ORDER BY ts DESC LIMIT ?""", [card_id, limit])
        if df.empty:
            raise HTTPException(404, "unknown card")
        extra = st.df("SELECT event_id, merchant_id, mcc, device_id, arrival_ts FROM events WHERE card_id = ?",
                      [card_id])
        df = df.merge(extra, on="event_id").sort_values("ts")
        return {"card_id": card_id, "features": [f.to_dict() for f in FEATURES],
                "transactions": json.loads(df.to_json(orient="records"))}

    return app
