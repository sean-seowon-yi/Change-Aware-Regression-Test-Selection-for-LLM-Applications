"""Unit tests for src/phase2/test_tagger.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.phase1.models import TestCase
from src.phase1.prompts.loader import load_prompt_sections
from src.phase2.test_tagger import (
    TaggedTest,
    tag_tests,
    _extract_demo_label_map,
    _extract_key_deps,
    _determine_sensitivity_category,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_test_cases(domain: str) -> list[TestCase]:
    """Load test cases from eval_suite.yaml."""
    path = DATA_DIR / domain / "eval_suite.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    cases = []
    for t in data["tests"]:
        cases.append(TestCase(
            test_id=t["test_id"],
            input_text=t.get("input_text", ""),
            monitor_type=t["monitor_type"],
            monitor_config=t.get("monitor_config", {}),
            tags=t.get("tags", []),
            sensitive_to=t.get("sensitive_to", []),
        ))
    return cases


def _load_demos_text(domain: str) -> str:
    """Load the demonstrations section text from the prompt."""
    sections = load_prompt_sections(domain)
    return sections.get("demonstrations", "")


# ============================================================
# Demo label map tests
# ============================================================

class TestExtractDemoLabelMap:
    def test_domain_a(self):
        demos = _load_demos_text("domain_a")
        label_map = _extract_demo_label_map(demos)
        assert 1 in label_map
        assert label_map[1] == "billing"
        assert label_map[2] == "refund"
        assert label_map[8] == "escalation"

    def test_domain_b(self):
        demos = _load_demos_text("domain_b")
        label_map = _extract_demo_label_map(demos)
        assert 1 in label_map
        assert len(label_map) >= 6

    def test_empty_text(self):
        assert _extract_demo_label_map("") == {}


# ============================================================
# Key dependency extraction tests
# ============================================================

class TestExtractKeyDeps:
    def test_schema_monitor(self):
        tc = TestCase(
            test_id="t_001",
            input_text="test",
            monitor_type="schema",
            monitor_config={
                "json_schema": {
                    "type": "object",
                    "required": ["action", "priority", "customer_id"],
                    "properties": {
                        "action": {"type": "string"},
                        "priority": {"type": "string"},
                        "customer_id": {"type": "string"},
                    },
                }
            },
        )
        keys = _extract_key_deps(tc)
        assert "action" in keys
        assert "priority" in keys
        assert "customer_id" in keys

    def test_required_keys_monitor(self):
        tc = TestCase(
            test_id="t_016",
            input_text="test",
            monitor_type="required_keys",
            monitor_config={"keys": ["action", "priority", "summary"]},
        )
        keys = _extract_key_deps(tc)
        assert keys == ["action", "priority", "summary"]

    def test_policy_monitor_no_keys(self):
        tc = TestCase(
            test_id="t_051",
            input_text="test",
            monitor_type="policy",
            monitor_config={"required_refusal": True},
        )
        keys = _extract_key_deps(tc)
        assert keys == []

    def test_code_execution_no_keys(self):
        tc = TestCase(
            test_id="t_001",
            input_text="test",
            monitor_type="code_execution",
            monitor_config={"language": "python", "test_code": "assert True"},
        )
        keys = _extract_key_deps(tc)
        assert keys == []


# ============================================================
# Sensitivity category tests
# ============================================================

class TestDetermineSensitivityCategory:
    def test_schema_sensitive(self):
        tc = TestCase(
            test_id="t_001",
            input_text="test",
            monitor_type="schema",
            tags=["schema_sensitive", "billing"],
            sensitive_to=["format"],
        )
        assert _determine_sensitivity_category(tc) == "format"

    def test_policy_safety(self):
        tc = TestCase(
            test_id="t_051",
            input_text="test",
            monitor_type="policy",
            tags=["safety_sensitive", "injection"],
            sensitive_to=["policy"],
        )
        assert _determine_sensitivity_category(tc) == "policy"

    def test_workflow_sensitive(self):
        tc = TestCase(
            test_id="t_091",
            input_text="test",
            monitor_type="keyword_presence",
            tags=["workflow_sensitive"],
            sensitive_to=["workflow"],
        )
        assert _determine_sensitivity_category(tc) == "workflow"

    def test_demo_sensitive(self):
        tc = TestCase(
            test_id="t_031",
            input_text="test",
            monitor_type="keyword_presence",
            tags=["domain_coverage", "billing"],
            sensitive_to=["demonstration", "demo:ex1"],
        )
        assert _determine_sensitivity_category(tc) == "demo"

    def test_code_execution_is_demo(self):
        tc = TestCase(
            test_id="t_001",
            input_text="test",
            monitor_type="code_execution",
            tags=["code_correctness"],
            sensitive_to=["demonstration", "format"],
        )
        cat = _determine_sensitivity_category(tc)
        assert cat in ("demo", "format")

    def test_format_monitor(self):
        tc = TestCase(
            test_id="t_081",
            input_text="test",
            monitor_type="format",
            tags=["format_sensitive"],
            sensitive_to=["format"],
        )
        assert _determine_sensitivity_category(tc) == "format"


# ============================================================
# Full tag_tests integration tests
# ============================================================

class TestTagTestsDomainA:
    def test_tags_all_100(self):
        cases = _load_test_cases("domain_a")
        demos = _load_demos_text("domain_a")
        tagged = tag_tests(cases, demos)
        assert len(tagged) == 100

    def test_schema_tests_have_key_deps(self):
        cases = _load_test_cases("domain_a")
        demos = _load_demos_text("domain_a")
        tagged = tag_tests(cases, demos)
        schema_tests = [t for t in tagged if t.monitor_type == "schema"]
        for t in schema_tests:
            assert len(t.inferred_key_deps) > 0
            assert "action" in t.inferred_key_deps or "customer_id" in t.inferred_key_deps

    def test_policy_tests_correct_category(self):
        cases = _load_test_cases("domain_a")
        demos = _load_demos_text("domain_a")
        tagged = tag_tests(cases, demos)
        policy_tests = [t for t in tagged if "safety_sensitive" in t.tags]
        for t in policy_tests:
            assert t.sensitivity_category == "policy"

    def test_domain_coverage_have_demo_deps(self):
        cases = _load_test_cases("domain_a")
        demos = _load_demos_text("domain_a")
        tagged = tag_tests(cases, demos)
        dc_tests = [
            t for t in tagged
            if "domain_coverage" in t.tags and any("demo:" in s for s in t.sensitive_to)
        ]
        for t in dc_tests:
            assert len(t.inferred_demo_deps) > 0
            assert all(isinstance(d, str) and len(d) > 0 for d in t.inferred_demo_deps)

    def test_workflow_tests_correct_category(self):
        cases = _load_test_cases("domain_a")
        demos = _load_demos_text("domain_a")
        tagged = tag_tests(cases, demos)
        wf_tests = [t for t in tagged if "workflow_sensitive" in t.tags]
        for t in wf_tests:
            assert t.sensitivity_category == "workflow"


class TestTagTestsDomainB:
    def test_tags_all_80(self):
        cases = _load_test_cases("domain_b")
        demos = _load_demos_text("domain_b")
        tagged = tag_tests(cases, demos)
        assert len(tagged) == 80

    def test_code_execution_tests(self):
        cases = _load_test_cases("domain_b")
        demos = _load_demos_text("domain_b")
        tagged = tag_tests(cases, demos)
        code_tests = [t for t in tagged if t.monitor_type == "code_execution"]
        assert len(code_tests) > 0
        for t in code_tests:
            assert t.sensitivity_category in ("demo", "format")

    def test_policy_tests_correct_category(self):
        cases = _load_test_cases("domain_b")
        demos = _load_demos_text("domain_b")
        tagged = tag_tests(cases, demos)
        policy_tests = [t for t in tagged if "safety_sensitive" in t.tags]
        for t in policy_tests:
            assert t.sensitivity_category == "policy"

    def test_domain_coverage_have_demo_deps(self):
        cases = _load_test_cases("domain_b")
        demos = _load_demos_text("domain_b")
        tagged = tag_tests(cases, demos)
        dc_tests = [
            t for t in tagged
            if "domain_coverage" in t.tags and any("demo:" in s for s in t.sensitive_to)
        ]
        for t in dc_tests:
            assert len(t.inferred_demo_deps) > 0

    def test_no_demos_text_still_works(self):
        """Tagger should work even without demo text (labels become generic)."""
        cases = _load_test_cases("domain_b")
        tagged = tag_tests(cases, "")
        assert len(tagged) == 80
        dc_tests = [
            t for t in tagged
            if any("demo:" in s for s in t.sensitive_to)
        ]
        for t in dc_tests:
            # Labels should be generic like 'example_1'
            for dep in t.inferred_demo_deps:
                assert dep.startswith("example_")
