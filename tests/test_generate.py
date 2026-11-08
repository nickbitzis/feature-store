from __future__ import annotations

import numpy as np

from featstore.generate import COLUMNS, generate


def test_deterministic_for_a_seed():
    a, b = generate(200, 10, seed=5), generate(200, 10, seed=5)
    assert a.events.equals(b.events)


def test_columns_and_order():
    ds = generate(200, 10, seed=1)
    assert list(ds.events.columns) == ["event_id", *COLUMNS]
    assert ds.events.ts.is_monotonic_increasing
    assert (ds.events.event_id.to_numpy() == np.arange(len(ds.events))).all()


def test_all_fraud_patterns_planted():
    ds = generate(1500, 40, seed=2)
    kinds = set(ds.events[ds.events.is_fraud].fraud_type)
    assert kinds == {"card_testing", "account_takeover", "merchant_compromise"}
    assert 0.0005 < ds.events.is_fraud.mean() < 0.01


def test_arrival_never_before_event_and_some_late():
    ds = generate(500, 10, seed=3)
    lag = ds.events.arrival_ts - ds.events.ts
    assert (lag >= 0).all() and (lag <= 7200).all()
    assert 0.02 < (lag > 60).mean() < 0.07


def test_card_testing_is_many_small_cnp_charges():
    ds = generate(2000, 30, seed=4)
    ct = ds.events[ds.events.fraud_type == "card_testing"]
    small = ct[ct.amount <= 3.5]
    assert len(small) > 0.7 * len(ct) and small.cnp.all()
