"""Each window aggregate on hand-built sequences, online and offline, including boundaries and late events."""

from __future__ import annotations

import pytest

from featstore.definitions import DAY, HOUR
from featstore.online import CardState, haversine_km

from .helpers import T0, both, ev, online_from, request

AS_OF = T0 + 5 * HOUR


def check(events, name, expected, **req):
    online, offline = both(events, request(AS_OF, **req))
    assert online[name] == pytest.approx(expected)
    assert offline[name] == pytest.approx(expected)
    return online, offline


def test_count_1h_left_boundary_is_inclusive():
    events = [ev(AS_OF - HOUR), ev(AS_OF - HOUR - 1), ev(AS_OF - 10)]
    check(events, "txn_count_1h", 2)


def test_count_excludes_events_at_or_after_as_of():
    events = [ev(AS_OF - 100), ev(AS_OF), ev(AS_OF + 5)]
    check(events, "txn_count_1h", 1)


def test_count_24h_and_7d():
    events = [ev(AS_OF - 2 * HOUR), ev(AS_OF - 20 * HOUR), ev(AS_OF - 2 * DAY), ev(AS_OF - 8 * DAY)]
    check(events, "txn_count_24h", 2)
    check(events, "txn_count_7d", 3)


def test_amount_sum_windows():
    events = [ev(AS_OF - 30 * 60, 12.5), ev(AS_OF - 3 * HOUR, 100.25), ev(AS_OF - 3 * DAY, 7.0)]
    check(events, "amount_sum_1h", 12.5)
    check(events, "amount_sum_24h", 112.75)
    check(events, "amount_sum_7d", 119.75)


def test_amount_max_windows_and_default():
    events = [ev(AS_OF - HOUR, 40), ev(AS_OF - 2 * DAY, 900), ev(AS_OF - 10 * DAY, 5000)]
    check(events, "amount_max_24h", 40)
    check(events, "amount_max_7d", 900)
    check([ev(AS_OF - 10 * DAY, 5000)], "amount_max_24h", 0.0)


def test_distinct_merchants_24h():
    events = [ev(AS_OF - 10, merchant=1), ev(AS_OF - 20, merchant=1), ev(AS_OF - 30, merchant=2),
              ev(AS_OF - 2 * DAY, merchant=3)]
    check(events, "distinct_merchants_24h", 2)


def test_secs_since_last_txn_and_cap():
    check([ev(AS_OF - 900), ev(AS_OF - 5000)], "secs_since_last_txn", 900)
    check([ev(AS_OF - 8 * DAY)], "secs_since_last_txn", 7 * DAY)


def test_cnp_share_7d():
    events = [ev(AS_OF - 100, cnp=1), ev(AS_OF - 200, cnp=0), ev(AS_OF - 300, cnp=1), ev(AS_OF - 400, cnp=1)]
    check(events, "cnp_share_7d", 0.75)


def test_device_seen_before_lifetime():
    events = [ev(AS_OF - 20 * DAY, device=99), ev(AS_OF - 100, device=7)]
    check(events, "device_seen_before", 1.0, device=99)
    check(events, "device_seen_before", 0.0, device=12345)


def test_device_first_seen_at_as_of_is_not_before():
    check([ev(AS_OF, device=5)], "device_seen_before", 0.0, device=5)


def test_distance_from_home_matches_haversine():
    events = [ev(AS_OF - 100)]
    expected = haversine_km(40.0, -100.0, 41.0, -101.0)
    online, offline = check(events, "distance_from_home_km", expected, lat=41.0, lon=-101.0)
    assert 130 < expected < 150


def test_mcc_risk_lookup_and_default():
    check([ev(AS_OF - 100)], "mcc_risk", 0.012, mcc=6051)
    check([ev(AS_OF - 100)], "mcc_risk", 0.0, mcc=1234)


def test_late_event_lands_in_its_event_time_window():
    # Arrives two hours late, after a newer event. It still belongs to the 1h window as of AS_OF.
    events = [ev(AS_OF - 600, 30, arrival=AS_OF + 2 * HOUR), ev(AS_OF - 60, 5, arrival=AS_OF - 60)]
    online, offline = check(events, "txn_count_1h", 2)
    assert online["amount_sum_1h"] == offline["amount_sum_1h"] == 35
    store, _ = online_from(events)
    assert store.meta["out_of_order"] == 1


def test_event_beyond_allowed_lateness_is_dropped_and_counted():
    events = [ev(AS_OF, arrival=AS_OF), ev(AS_OF - 4 * HOUR, arrival=AS_OF + 10)]
    store, proc = online_from(events)
    assert store.meta["dropped_late"] == 1
    assert store.meta["processed"] == 1
    assert proc.watermark == AS_OF - 3 * HOUR


def test_retention_prunes_old_events_but_keeps_device_history():
    st = CardState()
    for i in range(5):
        st.insert(T0 + i * DAY, 1.0, 1, 0, 100 + i)
    st.prune(T0 + 3 * DAY)
    assert st.ts == [T0 + 3 * DAY, T0 + 4 * DAY]
    assert set(st.devices) == {100, 101, 102, 103, 104}


def test_insert_keeps_event_time_order():
    st = CardState()
    for t in (5, 1, 3, 3, 9, 0):
        st.insert(t, float(t), t, 0, 1)
    assert st.ts == sorted(st.ts)
    assert st.amount == [float(t) for t in st.ts]


def test_window_bug_counts_the_scored_transaction():
    events = [ev(AS_OF - 100), ev(AS_OF)]
    online, offline = both(events, request(AS_OF), window_bug=True)
    assert online["txn_count_1h"] == 2
    assert offline["txn_count_1h"] == 1
