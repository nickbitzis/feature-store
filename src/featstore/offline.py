"""Offline store: the same events in DuckDB, with features computed by SQL window functions.

Point-in-time join: label rows (the spine) are unioned into the event table as probe rows at their as-of
time. Every window frame is RANGE BETWEEN <window> PRECEDING AND 1 PRECEDING on integer seconds, which is
exactly as_of - window <= ts < as_of, the same rule the online path uses. Aggregates count only real
events (FILTER (WHERE NOT probe)), so a label row sees nothing at or after its own time.

The leaky variant reproduces a common mistake: aggregates come from a daily batch table stamped at the
end of each day, and labels are joined to it on card and date. A transaction at 09:20 then gets window
values as of midnight, which include itself and everything the card did for the rest of the day.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .definitions import FEATURES, Feature
from .generate import Dataset

SPINE_COLS = ["row_id", "card_id", "as_of", "req_ts", "amount", "mcc", "device_id", "lat", "lon", "cnp"]


def _agg_sql(f: Feature, frame: str) -> str:
    real = "FILTER (WHERE NOT probe)"
    a = f.aggregation
    if a == "count":
        return f"count(*) {real} OVER {frame}"
    if a == "sum":
        return f"coalesce(sum({f.column}) {real} OVER {frame}, {f.default})"
    if a == "max":
        return f"coalesce(max({f.column}) {real} OVER {frame}, {f.default})"
    if a == "mean":
        return f"coalesce(avg({f.column}::DOUBLE) {real} OVER {frame}, {f.default})"
    if a == "distinct":
        return f"count(DISTINCT {f.column}) {real} OVER {frame}"
    if a == "last_gap":
        return f"coalesce(ts - max(ts) {real} OVER {frame}, {f.default})"
    raise ValueError(a)


def feature_sql(spine: str, features: list[Feature] | None = None) -> str:
    """SQL that returns one row per spine row with every feature as of that row's as_of."""
    features = features or FEATURES
    windows = sorted({f.window_s for f in features if f.is_window})
    win_defs = ", ".join(
        f"w{w} AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN {w} PRECEDING AND 1 PRECEDING)" for w in windows
    )
    win_cols = ",\n      ".join(f"{_agg_sql(f, 'w' + str(f.window_s))}::DOUBLE AS {f.name}"
                               for f in features if f.is_window)
    req = {f.aggregation: f for f in features if not f.is_window}
    select = ["s.row_id", "s.card_id", "s.as_of", "s.req_ts"]
    select += [f"w.{f.name}" for f in features if f.is_window]
    joins = ""
    if "seen_before" in req:
        f = req["seen_before"]
        select.append(f"CASE WHEN d.prior > 0 THEN 1.0 ELSE 0.0 END AS {f.name}")
        joins += f"""
    LEFT JOIN (
      SELECT row_id, count(*) FILTER (WHERE NOT probe) OVER (
          PARTITION BY card_id, {f.column} ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior,
        probe
      FROM (
        SELECT card_id, {f.column}, ts, FALSE AS probe, NULL::BIGINT AS row_id FROM events
        UNION ALL
        SELECT card_id, {f.column}, req_ts AS ts, TRUE AS probe, row_id FROM {spine}
      )
    ) d ON d.row_id = s.row_id AND d.probe"""
    if "distance" in req:
        f = req["distance"]
        select.append(
            "coalesce(2 * 6371.0 * asin(sqrt(pow(sin(radians(s.lat - c.home_lat) / 2), 2) + cos(radians(c.home_lat))"
            f" * cos(radians(s.lat)) * pow(sin(radians(s.lon - c.home_lon) / 2), 2))), {f.default}) AS {f.name}"
        )
        joins += "\n    LEFT JOIN cards c ON c.card_id = s.card_id"
    if "lookup" in req:
        f = req["lookup"]
        select.append(f"coalesce(r.risk, {f.default})::DOUBLE AS {f.name}")
        joins += f"\n    LEFT JOIN mcc_risk r ON r.mcc = s.{f.column}"
    return f"""
    WITH u AS (
      SELECT card_id, ts, amount, merchant_id, cnp, FALSE AS probe, NULL::BIGINT AS row_id FROM events
      UNION ALL
      SELECT card_id, as_of AS ts, NULL, NULL, NULL, TRUE, row_id FROM {spine}
    ), w AS (
      SELECT row_id, probe,
      {win_cols}
      FROM u
      WINDOW {win_defs}
    )
    SELECT {", ".join(select)}
    FROM {spine} s
    JOIN w ON w.row_id = s.row_id AND w.probe{joins}
    ORDER BY s.row_id
    """


class OfflineStore:
    def __init__(self, path: Path | str = ":memory:", read_only: bool = False):
        self.path = str(path)
        self.con = duckdb.connect(self.path, read_only=read_only)

    def close(self) -> None:
        self.con.close()

    def load(self, ds: Dataset) -> None:
        for name, df in (("events", ds.events), ("cards", ds.cards), ("mcc_risk", ds.mcc_risk),
                         ("merchants", ds.merchants)):
            self.con.register("_df", df)
            self.con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
            self.con.unregister("_df")

    def load_events(self, events: pd.DataFrame, cards: pd.DataFrame, mcc_risk: pd.DataFrame) -> None:
        for name, df in (("events", events), ("cards", cards), ("mcc_risk", mcc_risk)):
            self.con.register("_df", df)
            self.con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
            self.con.unregister("_df")

    def features_for_spine(self, spine: pd.DataFrame, features: list[Feature] | None = None) -> pd.DataFrame:
        """Point-in-time features for arbitrary (card, as_of, request) rows."""
        self.con.register("_spine_df", spine[SPINE_COLS])
        self.con.execute("CREATE OR REPLACE TEMP TABLE _spine AS SELECT * FROM _spine_df")
        self.con.unregister("_spine_df")
        return self.con.execute(feature_sql("_spine", features)).df()

    def build_training_set(self, kind: str = "pit", table: str | None = None, ts_from: int | None = None,
                           ts_to: int | None = None) -> str:
        """Materialize a training set table. kind 'pit' is point-in-time correct; 'leaky' joins to daily
        snapshots stamped at the end of the day. Returns the table name."""
        if kind not in ("pit", "leaky"):
            raise ValueError(kind)
        table = table or f"train_{kind}"
        as_of = "ts" if kind == "pit" else "(ts // 86400 + 1) * 86400"
        where = []
        if ts_from is not None:
            where.append(f"ts >= {int(ts_from)}")
        if ts_to is not None:
            where.append(f"ts < {int(ts_to)}")
        cond = ("WHERE " + " AND ".join(where)) if where else ""
        self.con.execute(f"""CREATE OR REPLACE TABLE _spine_{kind} AS
            SELECT event_id AS row_id, card_id, {as_of} AS as_of, ts AS req_ts, amount, mcc, device_id, lat, lon, cnp
            FROM events {cond}""")
        self.con.execute(f"""CREATE OR REPLACE TABLE {table} AS
            SELECT f.*, e.amount, e.cnp, e.is_fraud, e.fraud_type, e.ts
            FROM ({feature_sql(f"_spine_{kind}")}) f JOIN events e ON e.event_id = f.row_id""")
        self.con.execute(f"DROP TABLE _spine_{kind}")
        return table

    def df(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def scalar(self, sql: str, params: list | None = None):
        return self.con.execute(sql, params or []).fetchone()[0]
