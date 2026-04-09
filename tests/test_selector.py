"""Integration tests for src/phase2/selector.py and src/phase2/sentinel.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.phase1.models import TestCase
from src.phase1.prompts.loader import load_prompt_sections, assemble_prompt
from src.phase2.selector import select_tests, SelectionResult
from src.phase2.sentinel import sample_sentinels


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_test_cases(domain: str) -> list[TestCase]:
    path = DATA_DIR / domain / "eval_suite.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        TestCase(
            test_id=t["test_id"],
            input_text=t.get("input_text", ""),
            monitor_type=t["monitor_type"],
            monitor_config=t.get("monitor_config", {}),
            tags=t.get("tags", []),
            sensitive_to=t.get("sensitive_to", []),
        )
        for t in data["tests"]
    ]


def _load_base_prompt(domain: str) -> str:
    return assemble_prompt(load_prompt_sections(domain))


def _load_version_prompt(domain: str, version_id: str) -> str:
    path = DATA_DIR / domain / "versions" / f"{version_id}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["prompt_text"]


# ===========================================================================
# Sentinel sampler tests
# ===========================================================================

class TestSentinelSampler:
    def test_samples_from_non_predicted(self):
        all_ids = {f"t_{i:03d}" for i in range(1, 101)}
        predicted = {f"t_{i:03d}" for i in range(1, 51)}
        sentinels = sample_sentinels(all_ids, predicted, seed=42)
        assert sentinels.isdisjoint(predicted)
        assert sentinels.issubset(all_ids)

    def test_respects_min_sentinels(self):
        all_ids = {f"t_{i:03d}" for i in range(1, 21)}
        predicted = {f"t_{i:03d}" for i in range(1, 16)}
        sentinels = sample_sentinels(
            all_ids, predicted, fraction=0.01, min_sentinels=5, seed=42
        )
        assert len(sentinels) == 5

    def test_returns_empty_when_all_predicted(self):
        all_ids = {"t_001", "t_002"}
        predicted = {"t_001", "t_002"}
        sentinels = sample_sentinels(all_ids, predicted, seed=42)
        assert sentinels == set()

    def test_caps_at_pool_size(self):
        all_ids = {"t_001", "t_002", "t_003"}
        predicted = {"t_001"}
        sentinels = sample_sentinels(
            all_ids, predicted, fraction=1.0, min_sentinels=100, seed=42
        )
        assert sentinels == {"t_002", "t_003"}

    def test_deterministic_with_seed(self):
        all_ids = {f"t_{i:03d}" for i in range(1, 101)}
        predicted = {f"t_{i:03d}" for i in range(1, 51)}
        s1 = sample_sentinels(all_ids, predicted, seed=123)
        s2 = sample_sentinels(all_ids, predicted, seed=123)
        assert s1 == s2

    def test_different_seeds_differ(self):
        all_ids = {f"t_{i:03d}" for i in range(1, 101)}
        predicted = {f"t_{i:03d}" for i in range(1, 51)}
        s1 = sample_sentinels(all_ids, predicted, seed=1)
        s2 = sample_sentinels(all_ids, predicted, seed=2)
        assert s1 != s2


# ===========================================================================
# Selector integration tests (domain_a)
# ===========================================================================

class TestSelectorDomainA:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.base = _load_base_prompt("domain_a")
        self.cases = _load_test_cases("domain_a")
        self.all_ids = {tc.test_id for tc in self.cases}

    def test_v01_format_rename(self):
        v_prompt = _load_version_prompt("domain_a", "v01")
        result = select_tests(
            self.base, v_prompt, self.cases, sentinel_seed=42
        )
        assert result.selected_ids == result.predicted_ids | result.sentinel_ids
        assert result.total_tests == 100
        assert 0 < result.call_reduction < 1.0
        assert len(result.changes) >= 1

    def test_v30_policy_change(self):
        v_prompt = _load_version_prompt("domain_a", "v30")
        result = select_tests(
            self.base, v_prompt, self.cases, sentinel_seed=42
        )
        assert any(
            c.change.unit_type == "policy" for c in result.changes
        )
        assert len(result.predicted_ids) > 0

    def test_v37_workflow_reorder(self):
        v_prompt = _load_version_prompt("domain_a", "v37")
        result = select_tests(
            self.base, v_prompt, self.cases, sentinel_seed=42
        )
        wf = [c for c in result.changes if c.change.unit_type == "workflow"]
        assert len(wf) >= 1

    def test_sentinel_disjoint_from_predicted(self):
        v_prompt = _load_version_prompt("domain_a", "v01")
        result = select_tests(
            self.base, v_prompt, self.cases, sentinel_seed=42
        )
        assert result.sentinel_ids.isdisjoint(result.predicted_ids)

    def test_call_reduction_math(self):
        v_prompt = _load_version_prompt("domain_a", "v01")
        result = select_tests(
            self.base, v_prompt, self.cases, sentinel_seed=42
        )
        expected = 1.0 - len(result.selected_ids) / result.total_tests
        assert abs(result.call_reduction - expected) < 1e-9

    def test_identical_prompts(self):
        result = select_tests(
            self.base, self.base, self.cases, sentinel_seed=42
        )
        assert len(result.predicted_ids) == 0
        assert len(result.sentinel_ids) > 0
        assert result.selected_ids == result.sentinel_ids


# ===========================================================================
# Selector integration tests (domain_b)
# ===========================================================================

class TestSelectorDomainB:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.base = _load_base_prompt("domain_b")
        self.cases = _load_test_cases("domain_b")

    def test_v01_runs(self):
        v_prompt = _load_version_prompt("domain_b", "v01")
        result = select_tests(
            self.base, v_prompt, self.cases, sentinel_seed=42
        )
        assert result.total_tests == 80
        assert len(result.selected_ids) > 0

    def test_v16_demo_removal(self):
        v_prompt = _load_version_prompt("domain_b", "v16")
        result = select_tests(
            self.base, v_prompt, self.cases, sentinel_seed=42
        )
        demo_changes = [
            c for c in result.changes
            if c.change.unit_type in ("demonstration", "demo_item")
        ]
        assert len(demo_changes) >= 1

    def test_metadata_has_reasons(self):
        v_prompt = _load_version_prompt("domain_b", "v01")
        result = select_tests(
            self.base, v_prompt, self.cases, sentinel_seed=42
        )
        reasons = result.metadata.get("reasons", {})
        for tid in result.predicted_ids:
            assert tid in reasons
