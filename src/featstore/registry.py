"""Registry: feature metadata, definition versions, lineage, and freshness.

A feature's version is bumped only when its semantic definition changes (aggregation, column, window,
default, source, entity). The hash of those fields is stored with every version, so a model trained on
txn_count_1h@v1 can be refused at serving time if the live definition is v2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .definitions import FEATURES, SOURCES, Feature


class Registry:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.data: dict = {"features": {}, "models": []}
        if path and path.exists():
            self.data = json.loads(path.read_text())

    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self.data, indent=1))

    def register(self, features: list[Feature] | None = None, now: float | None = None) -> list[str]:
        """Add or bump versions. Returns the names whose version changed."""
        now = now or time.time()
        changed = []
        for f in features or FEATURES:
            entry = self.data["features"].setdefault(f.name, {"versions": []})
            versions = entry["versions"]
            h = f.definition_hash()
            if versions and versions[-1]["definition_hash"] == h:
                versions[-1]["metadata"] = f.to_dict()  # owner or description edits update in place
                continue
            versions.append({"version": len(versions) + 1, "definition_hash": h, "registered_at": now,
                             "definition": f.semantic(), "metadata": f.to_dict()})
            changed.append(f.name)
        return changed

    def current_version(self, name: str) -> int:
        return self.data["features"][name]["versions"][-1]["version"]

    def feature_refs(self, names: list[str]) -> dict[str, int]:
        return {n: self.current_version(n) for n in names}

    def register_model(self, model_id: str, feature_refs: dict[str, int], info: dict) -> None:
        self.data["models"] = [m for m in self.data["models"] if m["model_id"] != model_id]
        self.data["models"].append({"model_id": model_id, "features": feature_refs, **info})

    def check_model(self, model_id: str) -> list[str]:
        """Features whose live version no longer matches what the model was trained on."""
        m = next(m for m in self.data["models"] if m["model_id"] == model_id)
        return [n for n, v in m["features"].items() if n in self.data["features"] and self.current_version(n) != v]

    def listing(self) -> list[dict]:
        out = []
        for name, entry in self.data["features"].items():
            cur = entry["versions"][-1]
            used_by = [m["model_id"] for m in self.data["models"] if name in m["features"]]
            out.append({**cur["metadata"], "version": cur["version"], "versions": len(entry["versions"]),
                        "registered_at": cur["registered_at"], "used_by": used_by})
        return out

    def history(self, name: str) -> list[dict]:
        return self.data["features"][name]["versions"]

    def lineage(self) -> dict:
        """Graph from source to feature version to model."""
        nodes, edges = [], []
        for s, meta in SOURCES.items():
            nodes.append({"id": f"source:{s}", "kind": "source", "label": s, "source_kind": meta["kind"],
                          "description": meta["description"]})
        for name, entry in self.data["features"].items():
            cur = entry["versions"][-1]
            fid = f"feature:{name}@v{cur['version']}"
            nodes.append({"id": fid, "kind": "feature", "label": f"{name} v{cur['version']}"})
            edges.append({"from": f"source:{cur['metadata']['source']}", "to": fid})
        for m in self.data["models"]:
            mid = f"model:{m['model_id']}"
            nodes.append({"id": mid, "kind": "model", "label": m["model_id"]})
            for n, v in m["features"].items():
                edges.append({"from": f"feature:{n}@v{v}", "to": mid})
        return {"nodes": nodes, "edges": edges}


def freshness(online_meta: dict, source_high_water: int, now_sim: int | None = None,
              features: list[Feature] | None = None) -> list[dict]:
    """Per-feature freshness. Stream features: how far the online store's latest processed event time trails
    the newest event in the source. Batch features: age of the last table load. Times are simulated seconds."""
    now_sim = now_sim if now_sim is not None else source_high_water
    out = []
    for f in features or FEATURES:
        if f.source == "card_transactions_stream":
            lag = max(0, int(source_high_water) - int(online_meta.get("max_event_ts", 0)))
        else:
            lag = max(0, int(now_sim) - int(online_meta.get("batch_tables_as_of", 0)))
        status = "green" if lag <= f.freshness_sla_s else ("amber" if lag <= 3 * f.freshness_sla_s else "red")
        out.append({"feature": f.name, "source": f.source, "sla_s": f.freshness_sla_s, "lag_s": lag,
                    "status": status})
    return out
