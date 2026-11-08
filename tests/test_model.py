from __future__ import annotations

import numpy as np
import pandas as pd

from featstore.definitions import DAY, MODEL_INPUTS
from featstore.model import TimeSplit, evaluate, metrics_from_scores, split, train


def test_time_split_has_gap_and_no_overlap():
    b = TimeSplit().bounds(0, 60)
    assert b["train"] == (7 * DAY, 38 * DAY)
    assert b["test"] == (45 * DAY, 60 * DAY)
    assert b["train"][1] < b["test"][0]


def test_time_split_scales_with_shorter_runs():
    b = TimeSplit().bounds(100, 30)
    assert b["train"][1] < b["test"][0] <= 100 + 30 * DAY


def test_split_filters_by_event_time():
    df = pd.DataFrame({"ts": [6 * DAY, 10 * DAY, 40 * DAY, 50 * DAY]})
    tr, te = split(df, TimeSplit().bounds(0, 60))
    assert list(tr.ts) == [10 * DAY] and list(te.ts) == [50 * DAY]


def test_precision_recall_dollars_at_alert_rate():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0])
    amt = np.array([100.0, 1, 50, 1, 1, 1, 1, 1, 1, 1])
    m = metrics_from_scores(y, p, amt, alert_rate=0.2)
    assert m["alerts"] == 2
    assert m["precision_at_alert_rate"] == 0.5 and m["recall_at_alert_rate"] == 0.5
    assert m["fraud_dollars_caught"] == 100 and abs(m["dollar_recall"] - 2 / 3) < 1e-9


def test_auc_none_when_single_class():
    m = metrics_from_scores(np.zeros(5), np.arange(5) / 5, np.ones(5), 0.2)
    assert m["auc"] is None


def test_model_learns_planted_signal():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({c: rng.random(n) for c in MODEL_INPUTS})
    df["is_fraud"] = df.txn_count_1h > 0.97
    df["amount"] = rng.random(n) * 100
    m = train(df, seed=0)
    out = evaluate(m, df)
    assert out["auc"] > 0.95


def test_pipeline_evaluates_on_later_window(built):
    _, report = built
    tr, te = report["split"]["train"], report["split"]["test"]
    assert tr[1] < te[0]
    assert report["evaluation"]["test_rows"] > 0


def test_leak_inflates_offline_metrics(built):
    _, report = built
    e = report["evaluation"]
    assert e["leaky_offline"]["auc"] > e["leaky_in_production"]["auc"]
