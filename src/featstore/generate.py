"""Synthetic card transactions for Larkspur Card Services, a made-up card issuer.

Seeded and vectorized: 50,000 cards over 60 days is about 2 million events. Each card has a home city, a
spending rate, a typical amount, favorite merchants, and one to three devices. Three fraud patterns are
planted on top: card testing bursts, account takeover from a new device and location, and a merchant
compromise where cards used at breached merchants are later used for counterfeit spend far from home.
Every event has an event time and an arrival time; about 4% arrive out of order, up to 2 hours late.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

START_TS = int(datetime(2026, 8, 1, tzinfo=UTC).timestamp())

# (mcc, name, base fraud risk, amount multiplier, card-not-present probability)
MCCS = [
    (5411, "grocery", 0.0004, 1.0, 0.05),
    (5541, "fuel", 0.0006, 0.9, 0.02),
    (5812, "restaurants", 0.0005, 0.7, 0.05),
    (5311, "department stores", 0.0010, 1.6, 0.20),
    (5912, "pharmacy", 0.0004, 0.6, 0.05),
    (5999, "online retail", 0.0030, 1.4, 0.95),
    (5816, "digital goods", 0.0080, 0.4, 1.00),
    (5732, "electronics", 0.0060, 3.0, 0.40),
    (5944, "jewelry", 0.0070, 4.0, 0.30),
    (4511, "airlines", 0.0035, 6.0, 0.90),
    (6051, "quasi-cash", 0.0120, 2.5, 0.80),
    (5814, "fast food", 0.0003, 0.3, 0.02),
]
MCC_CODES = np.array([m[0] for m in MCCS])
MCC_WEIGHTS = np.array([0.20, 0.10, 0.14, 0.07, 0.07, 0.12, 0.05, 0.04, 0.02, 0.02, 0.01, 0.16])
DIGITAL = 6
RISKY = np.array([7, 8, 10])  # electronics, jewelry, quasi-cash
# Diurnal shape: share of transactions by hour of day.
HOURLY = np.array([1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 2.5, 3.5, 4, 4.5, 5,
                   5.5, 5, 4.5, 4.5, 5, 5.5, 6, 5.5, 4.5, 3.5, 2.5, 1.5])
HOURLY = HOURLY / HOURLY.sum()

COLUMNS = ["card_id", "ts", "arrival_ts", "amount", "merchant_id", "mcc", "device_id", "lat", "lon", "cnp",
           "is_fraud", "fraud_type"]


@dataclass
class Dataset:
    events: pd.DataFrame
    cards: pd.DataFrame
    merchants: pd.DataFrame
    mcc_risk: pd.DataFrame
    start_ts: int
    days: int


def _times(rng, n, days):
    day = rng.integers(0, days, n)
    hour = rng.choice(24, n, p=HOURLY)
    return START_TS + day * 86400 + hour * 3600 + rng.integers(0, 3600, n)


def generate(n_cards: int = 50_000, days: int = 60, seed: int = 7, late_share: float = 0.04,
             fraud_scale: float = 1.0) -> Dataset:
    rng = np.random.default_rng(seed)
    n_cities = 40
    city_lat = rng.uniform(30, 47, n_cities)
    city_lon = rng.uniform(-122, -72, n_cities)

    n_merch = max(400, n_cards * 2 // 5)
    m_city = rng.integers(0, n_cities, n_merch)
    m_mcc_idx = rng.choice(len(MCCS), n_merch, p=MCC_WEIGHTS)
    merchants = pd.DataFrame({
        "merchant_id": np.arange(n_merch),
        "city": m_city,
        "mcc": MCC_CODES[m_mcc_idx],
        "lat": city_lat[m_city] + rng.normal(0, 0.08, n_merch),
        "lon": city_lon[m_city] + rng.normal(0, 0.08, n_merch),
    })
    m_lat, m_lon = merchants.lat.to_numpy(), merchants.lon.to_numpy()
    mcc_mult = np.array([m[3] for m in MCCS])[m_mcc_idx]
    mcc_cnp = np.array([m[4] for m in MCCS])[m_mcc_idx]
    by_city = [np.flatnonzero(m_city == c) for c in range(n_cities)]
    for c in range(n_cities):
        if len(by_city[c]) == 0:
            by_city[c] = np.array([c % n_merch])

    # Cards: home, rate, amount scale, devices, favorite merchants.
    c_city = rng.integers(0, n_cities, n_cards)
    home_lat = city_lat[c_city] + rng.normal(0, 0.05, n_cards)
    home_lon = city_lon[c_city] + rng.normal(0, 0.05, n_cards)
    rate = rng.lognormal(np.log(0.55), 0.6, n_cards)
    scale = rng.lognormal(np.log(38), 0.5, n_cards)
    n_dev = rng.integers(1, 4, n_cards)
    n_fav = 10
    fav = np.empty((n_cards, n_fav), dtype=np.int64)
    for c in range(n_cities):
        idx = np.flatnonzero(c_city == c)
        if len(idx):
            fav[idx] = rng.choice(by_city[c], (len(idx), n_fav))
    travel = rng.random((n_cards, n_fav)) < 0.12
    fav[travel] = rng.integers(0, n_merch, travel.sum())
    cards = pd.DataFrame({"card_id": np.arange(n_cards), "home_lat": home_lat, "home_lon": home_lon,
                          "home_city": c_city})

    # Legitimate spend.
    counts = rng.poisson(rate * days)
    card = np.repeat(np.arange(n_cards), counts)
    n = len(card)
    merch = fav[card, rng.integers(0, n_fav, n)]
    ts = _times(rng, n, days)
    amount = np.round(rng.lognormal(np.log(scale[card] * mcc_mult[merch]), 0.7), 2) + 0.5
    cnp = rng.random(n) < mcc_cnp[merch]
    device = card * 4 + rng.integers(0, n_dev[card])
    # Trips: 6% of cards spend 2 to 6 days in another city, paying in person at merchants there.
    trip_card = rng.random(n_cards) < 0.06
    trip_start = START_TS + rng.integers(0, max(1, days - 6), n_cards) * 86400
    trip_end = trip_start + rng.integers(2, 7, n_cards) * 86400
    trip_city = (c_city + rng.integers(1, n_cities, n_cards)) % n_cities
    on_trip = trip_card[card] & (ts >= trip_start[card]) & (ts < trip_end[card])
    if on_trip.any():
        cities = trip_city[card[on_trip]]
        merch[on_trip] = np.array([rng.choice(by_city[c]) for c in cities])
        cnp[on_trip] = rng.random(on_trip.sum()) < 0.1
    # New phones: 8% of cards switch to a new device partway through.
    new_dev_at = np.where(rng.random(n_cards) < 0.08, START_TS + rng.integers(0, days * 86400, n_cards), 2**62)
    device = np.where(ts >= new_dev_at[card], card * 4 + 3, device)
    lat = np.where(cnp, home_lat[card] + rng.normal(0, 0.05, n), m_lat[merch])
    lon = np.where(cnp, home_lon[card] + rng.normal(0, 0.05, n), m_lon[merch])
    # IP geolocation is noisy: 15% of online purchases resolve to some other city.
    geo_miss = cnp & (rng.random(n) < 0.15)
    gc = rng.integers(0, n_cities, geo_miss.sum())
    lat[geo_miss], lon[geo_miss] = city_lat[gc], city_lon[gc]
    parts = [pd.DataFrame({"card_id": card, "ts": ts, "amount": amount, "merchant_id": merch, "device_id": device,
                           "lat": lat, "lon": lon, "cnp": cnp, "is_fraud": False, "fraud_type": ""})]

    digital_merch = np.flatnonzero(m_mcc_idx == DIGITAL)
    risky_merch = np.flatnonzero(np.isin(m_mcc_idx, RISKY))
    if len(digital_merch) == 0:
        digital_merch = np.arange(min(5, n_merch))
    if len(risky_merch) == 0:
        risky_merch = np.arange(min(5, n_merch))
    fraud_device = [n_cards * 4]

    def new_device():
        fraud_device[0] += 1
        return fraud_device[0]

    def rows(c, t, amt, m, dev, la, lo, is_cnp, kind):
        k = len(t)
        return pd.DataFrame({"card_id": np.full(k, c), "ts": np.asarray(t, dtype=np.int64), "amount": np.round(amt, 2),
                             "merchant_id": m, "device_id": np.full(k, dev), "lat": la, "lon": lo,
                             "cnp": is_cnp, "is_fraud": True, "fraud_type": kind})

    t_lo, t_hi = START_TS + 2 * 86400, START_TS + days * 86400 - 6 * 3600
    # Card testing: a burst of tiny card-not-present charges at digital merchants, then a big cash-out.
    n_test = max(3, int(n_cards * 0.0018 * fraud_scale))
    for c in rng.choice(n_cards, n_test, replace=False):
        t0 = rng.integers(t_lo, t_hi)
        k = rng.integers(8, 26)
        t = t0 + np.cumsum(rng.exponential(45, k)).astype(np.int64)
        city = c_city[c] if rng.random() < 0.4 else rng.integers(0, n_cities)
        dev = new_device()
        ms = rng.choice(digital_merch, rng.integers(1, 4))
        m = ms[rng.integers(0, len(ms), k)]
        la, lo = city_lat[city] + rng.normal(0, 0.05, k), city_lon[city] + rng.normal(0, 0.05, k)
        parts.append(rows(c, t, rng.uniform(0.5, 3.0, k), m, dev, la, lo, np.ones(k, bool), "card_testing"))
        j = rng.integers(1, 3)
        tc = t[-1] + rng.integers(600, 3600, j)
        parts.append(rows(c, tc, rng.uniform(250, 1500, j), rng.choice(risky_merch, j), dev,
                          np.full(j, city_lat[city]), np.full(j, city_lon[city]), np.ones(j, bool), "card_testing"))
    # Account takeover: new device, a different city, several large purchases within a few hours.
    n_ato = max(3, int(n_cards * 0.0025 * fraud_scale))
    for c in rng.choice(n_cards, n_ato, replace=False):
        t0 = rng.integers(t_lo, t_hi)
        k = rng.integers(3, 9)
        t = t0 + np.sort(rng.integers(0, rng.integers(1, 7) * 3600, k))
        city = c_city[c] if rng.random() < 0.45 else (c_city[c] + rng.integers(1, n_cities)) % n_cities
        dev = new_device() if rng.random() < 0.8 else c * 4
        m = rng.choice(np.concatenate([risky_merch, digital_merch, fav[c]]), k)
        amt = rng.lognormal(np.log(max(60.0, scale[c] * 4)), 0.7, k)
        parts.append(rows(c, t, amt, m, dev, city_lat[city] + rng.normal(0, 0.05, k),
                          city_lon[city] + rng.normal(0, 0.05, k), rng.random(k) < 0.8, "account_takeover"))
    # Merchant compromise: breached merchants leak card data; some cards then see counterfeit spend.
    legit = parts[0]
    breached = rng.choice(n_merch, max(2, int(n_merch * 0.002 * fraud_scale)), replace=False)
    b_lo, b_hi = START_TS + 8 * 86400, START_TS + max(10, days - 12) * 86400
    hit = legit[legit.merchant_id.isin(breached) & (legit.ts >= b_lo) & (legit.ts < b_hi)]
    victims = hit.drop_duplicates("card_id")
    victims = victims[rng.random(len(victims)) < 0.4]
    for c, tb in zip(victims.card_id.to_numpy(), victims.ts.to_numpy(), strict=True):
        t0 = tb + rng.integers(2, 10) * 86400 + rng.integers(0, 86400)
        if t0 >= START_TS + days * 86400 - 3600:
            continue
        k = rng.integers(1, 5)
        t = t0 + np.sort(rng.integers(0, 4 * 3600, k))
        city = c_city[c] if rng.random() < 0.4 else (c_city[c] + rng.integers(1, n_cities)) % n_cities
        m = rng.choice(by_city[city], k)
        # Counterfeit card swiped in person: no new device signal half the time.
        dev = c * 4 if rng.random() < 0.5 else new_device()
        parts.append(rows(c, t, rng.lognormal(np.log(180), 0.7, k), m, dev, m_lat[m], m_lon[m],
                          np.zeros(k, bool), "merchant_compromise"))

    ev = pd.concat(parts, ignore_index=True)
    ev["mcc"] = merchants.mcc.to_numpy()[ev.merchant_id.to_numpy()]
    # Arrival time: most events land within seconds, late_share arrive 1 minute to 2 hours late.
    delay = rng.exponential(2.0, len(ev)).astype(np.int64)
    late = rng.random(len(ev)) < late_share
    delay[late] = rng.integers(60, 7200, late.sum())
    ev["arrival_ts"] = ev.ts.to_numpy() + delay
    ev = ev.sort_values(["ts", "card_id"], kind="stable").reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(len(ev), dtype=np.int64))
    ev = ev[["event_id", *COLUMNS]]
    ev["cnp"] = ev.cnp.astype(int)

    # Merchant category risk: a published prior, not computed from this data, so it cannot leak labels.
    mcc_risk = pd.DataFrame({"mcc": MCC_CODES, "category": [m[1] for m in MCCS], "risk": [m[2] for m in MCCS]})
    return Dataset(ev, cards, merchants, mcc_risk, START_TS, days)
