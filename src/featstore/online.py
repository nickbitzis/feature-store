"""Online path: a stream processor that keeps per-card window state, and the online store that serves it.

State per card is the raw events of the last 8 days, kept sorted by event time, plus a lifetime map of
device first-seen times. Features are aggregated at read time from that state, which keeps late events
exact: a late event is inserted at its event-time position, so every window that should contain it does.
Events older than the watermark (max event time seen minus the allowed lateness) are dropped and counted.
State is snapshotted to SQLite so the scoring API can start from it.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .definitions import DAY, FEATURES, Feature

RETENTION_S = 8 * DAY
ALLOWED_LATENESS_S = 3 * 3600


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


@dataclass
class CardState:
    ts: list[int] = field(default_factory=list)
    amount: list[float] = field(default_factory=list)
    merchant: list[int] = field(default_factory=list)
    cnp: list[int] = field(default_factory=list)
    devices: dict[int, int] = field(default_factory=dict)  # device_id -> first seen event time

    def insert(self, ts: int, amount: float, merchant: int, cnp: int, device: int) -> bool:
        """Insert in event-time order. Returns True if the event was out of order for this card."""
        late = bool(self.ts) and ts < self.ts[-1]
        i = bisect.bisect_right(self.ts, ts) if late else len(self.ts)
        self.ts.insert(i, ts)
        self.amount.insert(i, amount)
        self.merchant.insert(i, merchant)
        self.cnp.insert(i, cnp)
        first = self.devices.get(device)
        if first is None or ts < first:
            self.devices[device] = ts
        return late

    def prune(self, before: int) -> None:
        if self.ts and self.ts[0] < before:
            k = bisect.bisect_left(self.ts, before)
            del self.ts[:k], self.amount[:k], self.merchant[:k], self.cnp[:k]

    def to_json(self) -> str:
        return json.dumps([self.ts, self.amount, self.merchant, self.cnp, list(self.devices.items())])

    @classmethod
    def from_json(cls, s: str) -> CardState:
        ts, amount, merchant, cnp, dev = json.loads(s)
        return cls(ts, amount, merchant, cnp, {int(k): int(v) for k, v in dev})


def window_value(f: Feature, st: CardState, as_of: int, window_bug: bool = False) -> float:
    lo = bisect.bisect_left(st.ts, as_of - f.window_s)
    # Correct: events strictly before as_of. The planted bug uses <= as_of, an off-by-one on the right
    # boundary that pulls in the transaction being scored once the processor has written it.
    hi = bisect.bisect_right(st.ts, as_of) if window_bug else bisect.bisect_left(st.ts, as_of)
    n = hi - lo
    agg = f.aggregation
    if agg == "count":
        return float(n)
    if n == 0:
        return f.default
    if agg == "sum":
        return float(math.fsum(st.amount[lo:hi]))
    if agg == "max":
        return float(max(st.amount[lo:hi]))
    if agg == "mean":
        return sum(st.cnp[lo:hi]) / n
    if agg == "distinct":
        return float(len(set(st.merchant[lo:hi])))
    if agg == "last_gap":
        return float(as_of - st.ts[hi - 1])
    raise ValueError(agg)


class OnlineStore:
    def __init__(self, features: list[Feature] | None = None):
        self.features = features or FEATURES
        self.cards: dict[int, CardState] = {}
        self.home: dict[int, tuple[float, float]] = {}
        self.mcc_risk: dict[int, float] = {}
        self.meta: dict = {"max_event_ts": 0, "processed": 0, "out_of_order": 0, "dropped_late": 0,
                           "last_arrival_ts": 0, "batch_tables_as_of": 0}

    def load_tables(self, cards_rows: Iterable, mcc_rows: Iterable, as_of: int) -> None:
        self.home = {int(c): (float(la), float(lo)) for c, la, lo in cards_rows}
        self.mcc_risk = {int(m): float(r) for m, r in mcc_rows}
        self.meta["batch_tables_as_of"] = int(as_of)

    def get_features(self, card_id: int, as_of: int, request: dict, window_bug: bool = False) -> dict[str, float]:
        """Feature vector for one card as of a time. request carries the transaction being scored."""
        st = self.cards.get(card_id) or CardState()
        out: dict[str, float] = {}
        req_ts = int(request.get("ts", as_of))
        for f in self.features:
            if f.is_window:
                out[f.name] = window_value(f, st, as_of, window_bug)
            elif f.aggregation == "seen_before":
                first = st.devices.get(int(request["device_id"]))
                seen = first is not None and (first <= req_ts if window_bug else first < req_ts)
                out[f.name] = 1.0 if seen else 0.0
            elif f.aggregation == "distance":
                home = self.home.get(card_id)
                out[f.name] = haversine_km(home[0], home[1], request["lat"], request["lon"]) if home else f.default
            elif f.aggregation == "lookup":
                out[f.name] = self.mcc_risk.get(int(request["mcc"]), f.default)
        return out

    # Snapshots ------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        tmp = Path(str(path) + ".tmp")
        tmp.unlink(missing_ok=True)
        con = sqlite3.connect(tmp)
        con.execute("CREATE TABLE card_state (card_id INTEGER PRIMARY KEY, state TEXT)")
        con.execute("CREATE TABLE tables (name TEXT PRIMARY KEY, body TEXT)")
        con.executemany("INSERT INTO card_state VALUES (?, ?)", ((c, s.to_json()) for c, s in self.cards.items()))
        meta = {**self.meta, "snapshot_wall_ts": time.time()}
        con.executemany("INSERT INTO tables VALUES (?, ?)", [
            ("home", json.dumps(list(self.home.items()))),
            ("mcc_risk", json.dumps(list(self.mcc_risk.items()))),
            ("meta", json.dumps(meta)),
        ])
        con.commit()
        con.close()
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> OnlineStore:
        store = cls()
        con = sqlite3.connect(path)
        store.cards = {c: CardState.from_json(s) for c, s in con.execute("SELECT card_id, state FROM card_state")}
        tables = dict(con.execute("SELECT name, body FROM tables"))
        con.close()
        store.home = {int(c): (float(v[0]), float(v[1])) for c, v in json.loads(tables["home"])}
        store.mcc_risk = {int(m): float(r) for m, r in json.loads(tables["mcc_risk"])}
        store.meta = json.loads(tables["meta"])
        return store


class StreamProcessor:
    """Consumes events in arrival order. Event-time watermark with allowed lateness, per-card retention."""

    def __init__(self, store: OnlineStore, allowed_lateness_s: int = ALLOWED_LATENESS_S,
                 retention_s: int = RETENTION_S):
        self.store = store
        self.lateness = allowed_lateness_s
        self.retention = retention_s

    @property
    def watermark(self) -> int:
        return self.store.meta["max_event_ts"] - self.lateness

    def process(self, card_id: int, ts: int, amount: float, merchant: int, cnp: int, device: int,
                arrival_ts: int | None = None) -> bool:
        """Apply one event. Returns False if it was dropped for arriving after the watermark."""
        meta = self.store.meta
        if arrival_ts is not None and arrival_ts > meta["last_arrival_ts"]:
            meta["last_arrival_ts"] = arrival_ts
        if ts < self.watermark:
            meta["dropped_late"] += 1
            return False
        if ts > meta["max_event_ts"]:
            meta["max_event_ts"] = ts
        st = self.store.cards.get(card_id)
        if st is None:
            st = self.store.cards[card_id] = CardState()
        if st.insert(ts, amount, merchant, cnp, device):
            meta["out_of_order"] += 1
        meta["processed"] += 1
        st.prune(self.watermark - self.retention)
        return True

    def run(self, rows: Iterable[tuple]) -> int:
        """rows: (card_id, ts, amount, merchant_id, cnp, device_id, arrival_ts), already in arrival order."""
        n = 0
        for card, ts, amount, merch, cnp, dev, arr in rows:
            self.process(card, ts, amount, merch, cnp, dev, arr)
            n += 1
        return n
