from __future__ import annotations

import dataclasses

from featstore.definitions import FEATURE_NAMES, FEATURES, Feature
from featstore.registry import Registry, freshness


def test_register_once_then_idempotent(tmp_path):
    r = Registry(tmp_path / "r.json")
    assert set(r.register(now=1)) == set(FEATURE_NAMES)
    assert r.register(now=2) == []
    assert all(r.current_version(n) == 1 for n in FEATURE_NAMES)


def test_window_change_bumps_version():
    r = Registry()
    r.register(now=1)
    f = dataclasses.replace(FEATURES[0], window_s=FEATURES[0].window_s + 1)
    assert r.register([f], now=2) == [f.name]
    assert r.current_version(f.name) == 2
    assert [v["definition"]["window_s"] for v in r.history(f.name)] == [FEATURES[0].window_s, f.window_s]


def test_description_change_does_not_bump():
    r = Registry()
    r.register(now=1)
    f = dataclasses.replace(FEATURES[0], description="new words", owner="someone-else")
    assert r.register([f], now=2) == []
    assert r.listing()[0]["owner"] == "someone-else"


def test_hash_is_stable_and_semantic():
    f = FEATURES[0]
    assert f.definition_hash() == dataclasses.replace(f, description="x").definition_hash()
    assert f.definition_hash() != dataclasses.replace(f, aggregation="sum", column="amount").definition_hash()


def test_invalid_specs_rejected():
    for bad in ({"aggregation": "median"}, {"aggregation": "count", "window_s": None}):
        try:
            Feature(**{"name": "x", "source": "s", "description": "", "owner": "o", **bad})
        except ValueError:
            continue
        raise AssertionError(bad)


def test_model_pins_versions_and_detects_stale():
    r = Registry()
    r.register(now=1)
    r.register_model("m1", r.feature_refs(FEATURE_NAMES), {})
    assert r.check_model("m1") == []
    r.register([dataclasses.replace(FEATURES[0], window_s=999)], now=2)
    assert r.check_model("m1") == [FEATURES[0].name]


def test_lineage_runs_source_to_feature_to_model():
    r = Registry()
    r.register(now=1)
    r.register_model("m1", r.feature_refs(FEATURE_NAMES), {})
    lin = r.lineage()
    kinds = {n["kind"] for n in lin["nodes"]}
    assert kinds == {"source", "feature", "model"}
    assert {"from": "source:card_transactions_stream", "to": "feature:txn_count_1h@v1"} in lin["edges"]
    assert {"from": "feature:txn_count_1h@v1", "to": "model:m1"} in lin["edges"]


def test_save_and_reload(tmp_path):
    r = Registry(tmp_path / "r.json")
    r.register(now=1)
    r.save()
    assert Registry(tmp_path / "r.json").current_version("mcc_risk") == 1


def test_freshness_statuses():
    meta = {"max_event_ts": 1000, "batch_tables_as_of": 0}
    by = {f["feature"]: f for f in freshness(meta, source_high_water=1030, now_sim=1030)}
    assert by["txn_count_1h"]["status"] == "green" and by["txn_count_1h"]["lag_s"] == 30
    by = {f["feature"]: f for f in freshness(meta, source_high_water=1150)}
    assert by["txn_count_1h"]["status"] == "amber"
    by = {f["feature"]: f for f in freshness(meta, source_high_water=5000)}
    assert by["txn_count_1h"]["status"] == "red" and by["mcc_risk"]["status"] == "green"
