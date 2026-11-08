from __future__ import annotations

from featstore.cli import main


def test_skew_gate_clean_passes(built):
    d, _ = built
    assert main(["--data-dir", str(d), "skew", "--sample", "60", "--expect-clean"]) == 0


def test_skew_gate_detects_bug(built):
    d, _ = built
    assert main(["--data-dir", str(d), "skew", "--sample", "60", "--bug", "--expect-detected"]) == 0


def test_skew_gate_fails_when_bug_expected_clean(built):
    d, _ = built
    assert main(["--data-dir", str(d), "skew", "--sample", "60", "--bug", "--expect-clean"]) == 1


def test_stream_limit_makes_freshness_red(built, tmp_path):
    import shutil

    from featstore.offline import OfflineStore
    from featstore.online import OnlineStore
    from featstore.registry import freshness

    d, _ = built
    shutil.copy(d / "offline.duckdb", tmp_path / "offline.duckdb")
    assert main(["--data-dir", str(tmp_path), "stream", "--limit", "1000"]) == 0
    store = OnlineStore.load(tmp_path / "online.sqlite")
    off = OfflineStore(tmp_path / "offline.duckdb", read_only=True)
    hw = int(off.scalar("SELECT max(ts) FROM events"))
    off.close()
    fr = {f["feature"]: f["status"] for f in freshness(store.meta, hw)}
    assert fr["txn_count_1h"] == "red" and store.meta["processed"] == 1000


def test_serve_without_build_fails(tmp_path):
    assert main(["--data-dir", str(tmp_path), "serve"]) == 1
