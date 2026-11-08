"""Online/offline skew check.

Sample cards with recent activity, take their transactions from the last 24 hours before the watermark,
and compute every feature twice: served by the online store, and recomputed by the offline SQL at the same
as-of time with the same request. Any difference beyond a small float tolerance is a mismatch. The gate
fails when any feature's mismatch rate is above the tolerance.
"""

from __future__ import annotations

import math

import numpy as np

from .definitions import DAY, FEATURES
from .offline import OfflineStore
from .online import OnlineStore

ABS_TOL = 1e-6
REL_TOL = 1e-9


def values_match(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def skew_report(online: OnlineStore, offline: OfflineStore, sample_cards: int = 300, per_card: int = 3,
                seed: int = 0, window_bug: bool = False, tolerance: float = 0.0) -> dict:
    wm = int(online.meta["max_event_ts"])
    lo = wm - DAY
    spine = offline.df(
        """SELECT event_id AS row_id, card_id, ts AS as_of, ts AS req_ts, amount, mcc, device_id, lat, lon, cnp
           FROM events WHERE ts >= ? AND ts <= ? ORDER BY event_id""",
        [lo, wm],
    )
    rng = np.random.default_rng(seed)
    cards = spine.card_id.unique()
    chosen = set(rng.choice(cards, min(sample_cards, len(cards)), replace=False).tolist()) if len(cards) else set()
    spine = spine[spine.card_id.isin(chosen)]
    spine = spine.sample(frac=1.0, random_state=seed).groupby("card_id").head(per_card)
    spine = spine.sort_values("row_id").reset_index(drop=True)
    off = offline.features_for_spine(spine).set_index("row_id")
    per_feature = {f.name: {"mismatches": 0, "max_abs_diff": 0.0, "example": None} for f in FEATURES}
    for r in spine.itertuples(index=False):
        req = {"ts": r.req_ts, "amount": r.amount, "mcc": r.mcc, "device_id": r.device_id, "lat": r.lat,
               "lon": r.lon, "cnp": r.cnp}
        served = online.get_features(int(r.card_id), int(r.as_of), req, window_bug=window_bug)
        row = off.loc[r.row_id]
        for f in FEATURES:
            a, b = served[f.name], float(row[f.name])
            if not values_match(a, b):
                pf = per_feature[f.name]
                pf["mismatches"] += 1
                pf["max_abs_diff"] = max(pf["max_abs_diff"], abs(a - b))
                if pf["example"] is None:
                    pf["example"] = {"card_id": int(r.card_id), "as_of": int(r.as_of), "online": a, "offline": b}
    n = len(spine)
    rows = []
    for f in FEATURES:
        pf = per_feature[f.name]
        rate = pf["mismatches"] / n if n else 0.0
        rows.append({"feature": f.name, "compared": n, "mismatches": pf["mismatches"], "mismatch_rate": rate,
                     "max_abs_diff": round(pf["max_abs_diff"], 6), "example": pf["example"],
                     "ok": rate <= tolerance})
    worst = max((r["mismatch_rate"] for r in rows), default=0.0)
    return {
        "window_bug": window_bug,
        "sampled_cards": len(chosen),
        "compared_rows": n,
        "as_of_range": [lo, wm],
        "tolerance": tolerance,
        "worst_mismatch_rate": worst,
        "features_with_skew": [r["feature"] for r in rows if r["mismatches"]],
        "passed": all(r["ok"] for r in rows),
        "features": rows,
    }
