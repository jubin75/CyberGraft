import pytest

from cybergraft.utils.memory_guard import MemoryGuard


def test_memory_guard_aborts_above_limit(monkeypatch):
    guard = MemoryGuard(warning_gb=1.0, abort_gb=2.0)
    monkeypatch.setattr(guard, "rss_gb", lambda: 2.1)
    with pytest.raises(MemoryError, match="safety limit"):
        guard.check("unit test")
