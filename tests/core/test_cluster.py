"""Memory guard for complete-linkage components."""

from types import SimpleNamespace

import pytest

from deh.core import cluster


def test_large_component_fails_before_dense_allocation(monkeypatch):
    fake = SimpleNamespace(
        live=lambda: SimpleNamespace(ram=SimpleNamespace(available_mb=1024))
    )
    monkeypatch.setattr(cluster, "machine", lambda: fake)

    cluster.check_linkage_budget(1000)
    with pytest.raises(RuntimeError, match="stricter threshold"):
        cluster.check_linkage_budget(10000)
