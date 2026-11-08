"""Point-in-time join correctness and the leak case."""

from __future__ import annotations

import pandas as pd

from featstore.definitions import DAY, HOUR

from .helpers import T0, ev, offline_from, request


def burst():
    # A legit purchase in the morning, then a fraud burst later the same day.
    base = T0 + 9 * HOUR
    return [ev(base - 2 * DAY, 20)] + [ev(base, 25)] + [ev(base + 3 * HOUR + i * 60, 300, fraud=True) for i in range(5)]


def test_pit_training_row_never_sees_its_own_or_later_events():
    off = offline_from(burst())
    off.build_training_set("pit")
    df = off.df("SELECT * FROM train_pit ORDER BY ts")
    first = df.iloc[1]  # the morning purchase
    assert first.txn_count_24h == 0 and first.amount_max_7d == 20
    last = df.iloc[-1]
    assert last.txn_count_1h == 4 and last.amount_sum_1h == 1200


def test_leaky_training_row_sees_the_rest_of_its_day():
    off = offline_from(burst())
    off.build_training_set("leaky")
    df = off.df("SELECT * FROM train_leaky ORDER BY ts")
    morning = df.iloc[1]
    # The morning purchase gets the evening's fraud burst and itself in its 24h window.
    assert morning.txn_count_24h == 6
    assert morning.amount_max_24h == 300


def test_leaky_and_pit_agree_on_request_features():
    off = offline_from(burst())
    off.build_training_set("pit")
    off.build_training_set("leaky")
    a = off.df("SELECT row_id, device_seen_before, distance_from_home_km, mcc_risk FROM train_pit ORDER BY row_id")
    b = off.df("SELECT row_id, device_seen_before, distance_from_home_km, mcc_risk FROM train_leaky ORDER BY row_id")
    pd.testing.assert_frame_equal(a, b)


def test_spine_at_arbitrary_times():
    off = offline_from([ev(T0 + 10), ev(T0 + 20), ev(T0 + 4000)])
    spine = pd.DataFrame([request(T0 + 15, row_id=0), request(T0 + 21, row_id=1), request(T0 + 3620, row_id=2)])
    out = off.features_for_spine(spine).set_index("row_id")
    assert list(out.txn_count_1h) == [1, 2, 1]


def test_spine_rows_with_the_same_timestamp_as_events_exclude_them():
    off = offline_from([ev(T0), ev(T0), ev(T0 - 5)])
    out = off.features_for_spine(pd.DataFrame([request(T0)]))
    assert out.txn_count_1h.iloc[0] == 1


def test_cards_are_partitioned():
    off = offline_from([ev(T0 - 10, card=1), ev(T0 - 10, card=2), ev(T0 - 20, card=2)])
    spine = pd.DataFrame([request(T0, card=1, row_id=0), request(T0, card=2, row_id=1)])
    out = off.features_for_spine(spine).set_index("row_id")
    assert list(out.txn_count_1h) == [1, 2]


def test_training_set_time_filter_and_labels():
    off = offline_from(burst())
    off.build_training_set("pit", ts_from=T0 + 8 * HOUR)
    df = off.df("SELECT * FROM train_pit")
    assert len(df) == 6 and int(df.is_fraud.sum()) == 5


def test_unknown_training_set_kind_rejected():
    off = offline_from(burst())
    try:
        off.build_training_set("future")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
