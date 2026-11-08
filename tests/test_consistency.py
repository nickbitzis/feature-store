"""Online vs offline equality on a generated stream, and the skew check catching the planted bug."""

from __future__ import annotations

import pytest

from featstore.definitions import FEATURE_NAMES
from featstore.generate import generate
from featstore.offline import OfflineStore
from featstore.pipeline import run_stream
from featstore.skew import skew_report, values_match


@pytest.fixture(scope="module")
def stores():
    ds = generate(n_cards=250, days=12, seed=11)
    off = OfflineStore()
    off.load(ds)
    store = run_stream(off)
    yield ds, off, store
    off.close()


def test_stream_applies_every_event_with_late_arrivals(stores):
    ds, _, store = stores
    assert store.meta["processed"] == len(ds.events)
    assert store.meta["out_of_order"] > 0
    assert store.meta["dropped_late"] == 0


def test_every_feature_equal_on_recent_transactions(stores):
    _, off, store = stores
    wm = store.meta["max_event_ts"]
    spine = off.df("""SELECT event_id AS row_id, card_id, ts AS as_of, ts AS req_ts, amount, mcc, device_id, lat, lon,
                      cnp FROM events WHERE ts >= ? ORDER BY event_id LIMIT 400""", [wm - 86400])
    offline = off.features_for_spine(spine).set_index("row_id")
    for r in spine.to_dict("records"):
        served = store.get_features(r["card_id"], r["as_of"], {**r, "ts": r["req_ts"]})
        for f in FEATURE_NAMES:
            assert values_match(served[f], float(offline.loc[r["row_id"], f])), (f, r["row_id"])


def test_skew_report_clean_with_bug_off(stores):
    _, off, store = stores
    r = skew_report(store, off, sample_cards=80)
    assert r["passed"] and r["worst_mismatch_rate"] == 0 and r["compared_rows"] > 50


def test_skew_report_catches_planted_bug(stores):
    _, off, store = stores
    r = skew_report(store, off, sample_cards=80, window_bug=True)
    assert not r["passed"]
    assert "txn_count_1h" in r["features_with_skew"]
    row = next(f for f in r["features"] if f["feature"] == "txn_count_1h")
    assert row["mismatch_rate"] == 1.0 and row["example"]["online"] == row["example"]["offline"] + 1


def test_skew_report_leaves_request_features_alone_under_bug(stores):
    _, off, store = stores
    r = skew_report(store, off, sample_cards=80, window_bug=True)
    assert "mcc_risk" not in r["features_with_skew"]
    assert "distance_from_home_km" not in r["features_with_skew"]


def test_tolerance_allows_small_mismatch_rates(stores):
    _, off, store = stores
    r = skew_report(store, off, sample_cards=80, window_bug=True, tolerance=1.0)
    assert r["passed"]


def test_snapshot_round_trip(stores, tmp_path):
    _, off, store = stores
    from featstore.online import OnlineStore

    store.save(tmp_path / "s.sqlite")
    back = OnlineStore.load(tmp_path / "s.sqlite")
    card = next(iter(store.cards))
    req = {"ts": store.meta["max_event_ts"], "device_id": 1, "mcc": 5411, "lat": 40.0, "lon": -100.0}
    assert back.get_features(card, req["ts"], req) == store.get_features(card, req["ts"], req)
    assert back.meta["processed"] == store.meta["processed"]


def test_values_match_tolerance():
    assert values_match(1.0, 1.0 + 1e-9)
    assert not values_match(1.0, 1.01)
    assert values_match(0.0, 5e-7)
