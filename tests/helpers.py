"""Hand-built event sequences for window tests."""

from __future__ import annotations

import pandas as pd

from featstore.offline import OfflineStore
from featstore.online import OnlineStore, StreamProcessor

T0 = 1_785_542_400 + 10 * 86400  # a fixed point well inside the simulated period
HOME = (40.0, -100.0)


def ev(ts, amount=10.0, merchant=1, cnp=0, device=7, card=1, arrival=None, mcc=5411, lat=40.0, lon=-100.0,
       fraud=False):
    return {"card_id": card, "ts": int(ts), "arrival_ts": int(arrival if arrival is not None else ts),
            "amount": float(amount), "merchant_id": merchant, "mcc": mcc, "device_id": device, "lat": lat,
            "lon": lon, "cnp": cnp, "is_fraud": fraud, "fraud_type": "x" if fraud else ""}


def frame(events):
    df = pd.DataFrame(events)
    df = df.sort_values(["ts", "card_id"], kind="stable").reset_index(drop=True)
    df.insert(0, "event_id", range(len(df)))
    return df


def cards_df(cards=(1,)):
    return pd.DataFrame({"card_id": list(cards), "home_lat": HOME[0], "home_lon": HOME[1], "home_city": 0})


def mcc_df():
    return pd.DataFrame({"mcc": [5411, 6051], "category": ["grocery", "quasi-cash"], "risk": [0.0004, 0.012]})


def online_from(events, lateness=3 * 3600):
    store = OnlineStore()
    store.load_tables([(c, HOME[0], HOME[1]) for c in {e["card_id"] for e in events}],
                      [(5411, 0.0004), (6051, 0.012)], as_of=T0)
    proc = StreamProcessor(store, allowed_lateness_s=lateness)
    for e in sorted(events, key=lambda e: e["arrival_ts"]):
        proc.process(e["card_id"], e["ts"], e["amount"], e["merchant_id"], e["cnp"], e["device_id"], e["arrival_ts"])
    return store, proc


def offline_from(events):
    off = OfflineStore()
    off.load_events(frame(events), cards_df(sorted({e["card_id"] for e in events})), mcc_df())
    return off


def request(as_of, card=1, device=7, mcc=5411, lat=40.0, lon=-100.0, amount=5.0, cnp=0, row_id=0):
    return {"row_id": row_id, "card_id": card, "as_of": int(as_of), "req_ts": int(as_of), "amount": amount, "mcc": mcc,
            "device_id": device, "lat": lat, "lon": lon, "cnp": cnp, "ts": int(as_of)}


def both(events, req, window_bug=False):
    """Online and offline feature values for one request on the same events."""
    store, _ = online_from(events)
    online = store.get_features(req["card_id"], req["as_of"], req, window_bug=window_bug)
    off = offline_from(events)
    offline = off.features_for_spine(pd.DataFrame([req])).iloc[0].to_dict()
    off.close()
    return online, offline
