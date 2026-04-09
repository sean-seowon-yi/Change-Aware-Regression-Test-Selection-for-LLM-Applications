"""Unit tests for src/phase1/harness/cache.py."""

import pytest

from src.phase1.harness.cache import EvalCache
from src.phase1.models import EvalResult


@pytest.fixture
def tmp_cache(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path / "test_cache"))
    yield cache
    cache.close()


class TestEvalCache:
    def test_miss_returns_none(self, tmp_cache):
        assert tmp_cache.get("prompt", "input", "model", 0.0) is None

    def test_roundtrip(self, tmp_cache):
        result = EvalResult(
            version_id="v01",
            test_id="t_001",
            outcome="pass",
            raw_output='{"action": "billing"}',
            monitor_details={"parsed": True},
            latency_ms=123.4,
            cost_usd=0.001,
            cached=False,
        )
        tmp_cache.put("prompt", "input", "gpt-4o-mini", 0.0, result)
        retrieved = tmp_cache.get("prompt", "input", "gpt-4o-mini", 0.0)

        assert retrieved is not None
        assert retrieved.version_id == "v01"
        assert retrieved.test_id == "t_001"
        assert retrieved.outcome == "pass"
        assert retrieved.raw_output == '{"action": "billing"}'
        assert retrieved.latency_ms == 123.4
        assert retrieved.cost_usd == 0.001

    def test_different_key_misses(self, tmp_cache):
        result = EvalResult(
            version_id="v01",
            test_id="t_001",
            outcome="pass",
            raw_output="{}",
            monitor_details={},
            latency_ms=0,
            cost_usd=0,
            cached=False,
        )
        tmp_cache.put("prompt_A", "input", "model", 0.0, result)
        assert tmp_cache.get("prompt_B", "input", "model", 0.0) is None

    def test_temperature_differentiates(self, tmp_cache):
        result = EvalResult(
            version_id="v01",
            test_id="t_001",
            outcome="pass",
            raw_output="{}",
            monitor_details={},
            latency_ms=0,
            cost_usd=0,
            cached=False,
        )
        tmp_cache.put("p", "i", "m", 0.0, result)
        assert tmp_cache.get("p", "i", "m", 0.0) is not None
        assert tmp_cache.get("p", "i", "m", 0.7) is None
