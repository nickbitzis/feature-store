"""Feature definitions as code.

Each feature is one declarative spec. The online path (Python over the stream state) and the offline path
(DuckDB SQL window functions) are both generated from the same spec, so there is one definition of what
"transactions in the last hour" means. Window semantics everywhere: events with as_of - window <= ts < as_of,
on integer epoch seconds. The transaction being scored is never part of its own features.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

HOUR = 3600
DAY = 86400

# Aggregations over a sliding event-time window of the card's transactions.
WINDOW_AGGS = ("count", "sum", "max", "mean", "distinct", "last_gap")
# Features that combine the request with entity state or a lookup table.
REQUEST_AGGS = ("seen_before", "distance", "lookup")


@dataclass(frozen=True)
class Feature:
    name: str
    aggregation: str
    source: str
    description: str
    owner: str
    entity: str = "card"
    column: str | None = None
    window_s: int | None = None
    default: float = 0.0
    freshness_sla_s: int = 60
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.aggregation not in WINDOW_AGGS + REQUEST_AGGS:
            raise ValueError(f"unknown aggregation {self.aggregation}")
        if self.aggregation in WINDOW_AGGS and not self.window_s:
            raise ValueError(f"{self.name}: window aggregations need window_s")

    @property
    def is_window(self) -> bool:
        return self.aggregation in WINDOW_AGGS

    def semantic(self) -> dict:
        """The fields that change the computed value. Owner and description edits do not bump the version."""
        return {
            "aggregation": self.aggregation,
            "column": self.column,
            "window_s": self.window_s,
            "default": self.default,
            "source": self.source,
            "entity": self.entity,
        }

    def definition_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.semantic(), sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["definition_hash"] = self.definition_hash()
        return d


FEATURE_REGISTRY: dict[str, Feature] = {}


def feature(**spec) -> Feature:
    """Register a feature. Used as a function call so definitions read like a small DSL."""
    f = Feature(**spec)
    if f.name in FEATURE_REGISTRY:
        raise ValueError(f"duplicate feature {f.name}")
    FEATURE_REGISTRY[f.name] = f
    return f


RISK = "fraud-risk-team"
PLATFORM = "feature-platform-team"
TXN = "card_transactions_stream"

for label, w in (("1h", HOUR), ("24h", DAY), ("7d", 7 * DAY)):
    feature(
        name=f"txn_count_{label}",
        aggregation="count",
        source=TXN,
        window_s=w,
        owner=RISK,
        description=f"Number of transactions on the card in the {label} before the as-of time.",
        tags=("velocity",),
    )
    feature(
        name=f"amount_sum_{label}",
        aggregation="sum",
        column="amount",
        source=TXN,
        window_s=w,
        owner=RISK,
        description=f"Total amount in USD spent on the card in the {label} before the as-of time.",
        tags=("spend",),
    )
for label, w in (("24h", DAY), ("7d", 7 * DAY)):
    feature(
        name=f"amount_max_{label}",
        aggregation="max",
        column="amount",
        source=TXN,
        window_s=w,
        owner=RISK,
        description=f"Largest single transaction in the {label} before the as-of time.",
        tags=("spend",),
    )
feature(
    name="distinct_merchants_24h",
    aggregation="distinct",
    column="merchant_id",
    source=TXN,
    window_s=DAY,
    owner=RISK,
    description="Distinct merchants the card used in the 24 hours before the as-of time.",
    tags=("velocity",),
)
feature(
    name="secs_since_last_txn",
    aggregation="last_gap",
    source=TXN,
    window_s=7 * DAY,
    default=float(7 * DAY),
    owner=RISK,
    description="Seconds since the card's previous transaction, capped at 7 days.",
    tags=("velocity",),
)
feature(
    name="cnp_share_7d",
    aggregation="mean",
    column="cnp",
    source=TXN,
    window_s=7 * DAY,
    owner=RISK,
    description="Share of the card's transactions in the last 7 days that were card-not-present.",
    tags=("channel",),
)
feature(
    name="device_seen_before",
    aggregation="seen_before",
    column="device_id",
    source=TXN,
    owner=PLATFORM,
    description="1 if this device was used with this card before the transaction, else 0. Lifetime state.",
    tags=("identity",),
)
feature(
    name="distance_from_home_km",
    aggregation="distance",
    source="cards_table",
    freshness_sla_s=DAY,
    owner=PLATFORM,
    description="Great-circle distance between the transaction location and the cardholder's home address.",
    tags=("location",),
)
feature(
    name="mcc_risk",
    aggregation="lookup",
    column="mcc",
    source="mcc_risk_table",
    freshness_sla_s=DAY,
    owner=RISK,
    description="Historical fraud rate of the merchant category, from the daily batch table.",
    tags=("merchant",),
)

FEATURES: list[Feature] = list(FEATURE_REGISTRY.values())
FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
# Raw request fields the model uses alongside the stored features.
REQUEST_INPUTS: list[str] = ["amount", "cnp"]
MODEL_INPUTS: list[str] = FEATURE_NAMES + REQUEST_INPUTS

SOURCES = {
    TXN: {"kind": "stream", "description": "Card authorization events, event time plus arrival time."},
    "cards_table": {"kind": "batch", "description": "Cardholder profile, home location. Daily snapshot."},
    "mcc_risk_table": {"kind": "batch", "description": "Fraud rate by merchant category code. Daily batch job."},
}
