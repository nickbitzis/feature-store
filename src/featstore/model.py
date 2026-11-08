"""Fraud model: gradient boosting on the point-in-time training set, evaluated on a later time window.

The split is by time, never random: train on days 7 to 38 (the first week is warm-up so 7-day windows are
full), skip a one-week gap, and test on day 45 onward. Metrics are the ones a fraud team runs on:
precision at a fixed alert rate (the team can review 0.5% of transactions), recall at that rate, AUC, and
the share of fraud dollars in the alerted transactions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .definitions import DAY, MODEL_INPUTS

ALERT_RATE = 0.0025


@dataclass(frozen=True)
class TimeSplit:
    train_from_day: int = 7
    train_to_day: int = 38
    test_from_day: int = 45

    def bounds(self, start_ts: int, days: int) -> dict[str, tuple[int, int]]:
        scale = days / 60
        tf, tt, sf = (round(d * scale) for d in (self.train_from_day, self.train_to_day, self.test_from_day))
        return {
            "train": (start_ts + tf * DAY, start_ts + tt * DAY),
            "test": (start_ts + sf * DAY, start_ts + days * DAY),
        }


def split(df: pd.DataFrame, bounds: dict[str, tuple[int, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[(df.ts >= bounds["train"][0]) & (df.ts < bounds["train"][1])]
    te = df[(df.ts >= bounds["test"][0]) & (df.ts < bounds["test"][1])]
    return tr, te


def train(df: pd.DataFrame, seed: int = 0, neg_rate: float = 0.3) -> HistGradientBoostingClassifier:
    rng = np.random.default_rng(seed)
    y = df.is_fraud.to_numpy().astype(int)
    keep = (y == 1) | (rng.random(len(y)) < neg_rate)
    w = np.where(y[keep] == 1, 1.0, 1.0 / neg_rate)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                       random_state=seed)
    m.fit(df.loc[keep, MODEL_INPUTS].to_numpy(dtype=float), y[keep], sample_weight=w)
    return m


def evaluate(model, df: pd.DataFrame, alert_rate: float = ALERT_RATE) -> dict:
    y = df.is_fraud.to_numpy().astype(int)
    p = model.predict_proba(df[MODEL_INPUTS].to_numpy(dtype=float))[:, 1]
    return metrics_from_scores(y, p, df.amount.to_numpy(), alert_rate)


def metrics_from_scores(y: np.ndarray, p: np.ndarray, amount: np.ndarray, alert_rate: float = ALERT_RATE) -> dict:
    n = len(y)
    k = max(1, int(round(n * alert_rate)))
    top = np.argsort(-p, kind="stable")[:k]
    tp = int(y[top].sum())
    fraud_dollars = float(amount[y == 1].sum())
    caught = float(amount[top][y[top] == 1].sum())
    return {
        "rows": int(n),
        "fraud_rows": int(y.sum()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < n else None,
        "alert_rate": alert_rate,
        "alerts": int(k),
        "precision_at_alert_rate": tp / k,
        "recall_at_alert_rate": tp / max(1, int(y.sum())),
        "fraud_dollars": round(fraud_dollars, 2),
        "fraud_dollars_caught": round(caught, 2),
        "dollar_recall": caught / fraud_dollars if fraud_dollars else None,
    }


def feature_importance(model, df: pd.DataFrame, seed: int = 0, n: int = 20_000) -> dict[str, float]:
    """Drop in AUC when one input is shuffled. Cheap permutation importance on a sample of the test set."""
    rng = np.random.default_rng(seed)
    s = df.sample(min(n, len(df)), random_state=seed) if len(df) > n else df
    if s.is_fraud.sum() == 0:
        return {}
    X = s[MODEL_INPUTS].to_numpy(dtype=float)
    y = s.is_fraud.to_numpy().astype(int)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for j, name in enumerate(MODEL_INPUTS):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        out[name] = round(float(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1])), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
