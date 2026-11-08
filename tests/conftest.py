from __future__ import annotations

import pytest

from featstore.pipeline import build


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """One small end-to-end build shared by the API, skew, and pipeline tests (a few seconds)."""
    d = tmp_path_factory.mktemp("fs")
    report = build(d, n_cards=600, days=24, seed=3, skew_cards=120, latency_requests=200)
    return d, report
