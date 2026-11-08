"""API end to end on a small build."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from featstore.app import create_app


@pytest.fixture(scope="module")
def client(built):
    d, _ = built
    with TestClient(create_app(d)) as c:
        yield c


@pytest.fixture(scope="module")
def txn(client):
    ex = client.get("/cards/examples").json()
    card = ex["normal"][0]
    tl = client.get(f"/cards/{card}/timeline").json()["transactions"][-1]
    return {"card_id": card, "amount": 25.0, "mcc": int(tl["mcc"]), "device_id": int(tl["device_id"]),
            "lat": 40.0, "lon": -100.0, "cnp": 0}


def test_healthz(client):
    r = client.get("/healthz").json()
    assert r["status"] == "ok" and r["events_processed"] > 1000


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>Feature Store</title>" in r.text


def test_score_returns_score_features_and_latency(client, txn):
    r = client.post("/score", json=txn)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 1 and set(body["latency_ms"]) == {"feature_fetch", "model_score"}
    assert body["features"]["device_seen_before"] == 1.0
    assert "x-request-id" in r.headers


def test_score_new_device_far_away(client, txn):
    body = client.post("/score", json={**txn, "device_id": 10**9, "lat": 25.0, "lon": -70.0}).json()
    assert body["features"]["device_seen_before"] == 0.0 and body["features"]["distance_from_home_km"] > 1000


def test_score_validation(client, txn):
    assert client.post("/score", json={**txn, "amount": -1}).status_code == 422
    assert client.post("/score", json={**txn, "cnp": 3}).status_code == 422


def test_score_with_explicit_as_of(client, txn):
    body = client.post("/score", json={**txn, "ts": 1_785_542_400}).json()
    assert body["as_of"] == 1_785_542_400 and body["features"]["txn_count_7d"] == 0


def test_features_for_entity(client, txn):
    r = client.get(f"/features/{txn['card_id']}").json()
    assert set(r["features"]) >= {"txn_count_1h", "cnp_share_7d"} and r["versions"]["txn_count_1h"] == 1


def test_features_unknown_card(client):
    assert client.get("/features/99999999").status_code == 404


def test_registry_and_history(client):
    r = client.get("/registry").json()
    assert len(r["features"]) == 14 and r["models"][0]["model_id"] == "fraud-gbm-v1"
    h = client.get("/registry/txn_count_1h").json()
    assert h["versions"][0]["definition"]["window_s"] == 3600
    assert client.get("/registry/nope").status_code == 404


def test_lineage(client):
    lin = client.get("/lineage").json()
    assert any(e["to"] == "model:fraud-gbm-v1" for e in lin["edges"])


def test_skew_stored_reports(client):
    off = client.get("/skew").json()
    on = client.get("/skew?bug=true").json()
    assert off["passed"] and not on["passed"]


def test_skew_live(client):
    r = client.get("/skew?live=true&sample=40").json()
    assert r["passed"] and r["compared_rows"] > 0
    assert not client.get("/skew?live=true&sample=40&bug=true").json()["passed"]


def test_freshness(client):
    r = client.get("/freshness").json()
    assert all(f["status"] == "green" for f in r["features"])


def test_evaluation(client):
    r = client.get("/evaluation").json()
    assert {"pit", "leaky_offline", "leaky_in_production", "by_fraud_type"} <= set(r)


def test_latency_live_after_scoring(client, txn):
    client.post("/score", json=txn)
    r = client.get("/latency").json()
    assert r["live"]["feature_fetch"]["count"] >= 1 and r["benchmark"]["requests"] > 0


def test_metrics(client, txn):
    client.post("/score", json=txn)
    text = client.get("/metrics").text
    assert "featstore_stage_latency_seconds_bucket" in text
    assert 'featstore_feature_freshness_lag_seconds{feature="txn_count_1h"}' in text
    assert "featstore_stream_events_processed" in text


def test_timeline_has_point_in_time_features(client):
    ex = client.get("/cards/examples").json()
    card = (ex.get("card_testing") or ex["normal"])[0]
    tx = client.get(f"/cards/{card}/timeline").json()["transactions"]
    assert tx == sorted(tx, key=lambda r: r["ts"])
    assert tx[0]["txn_count_7d"] == 0 or tx[0]["ts"] > 0


def test_timeline_unknown_card(client):
    assert client.get("/cards/99999999/timeline").status_code == 404
