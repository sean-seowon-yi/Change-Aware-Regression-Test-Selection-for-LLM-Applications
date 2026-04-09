"""Unit tests for src/phase2/change_classifier.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.phase1.prompts.loader import assemble_prompt, load_prompt_sections
from src.phase2.change_classifier import (
    ClassifiedChange,
    classify_changes,
    _extract_json_keys,
    _extract_demo_label,
    _diff_demos,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_base_prompt(domain: str) -> str:
    """Load the assembled base prompt for a domain."""
    sections = load_prompt_sections(domain)
    return assemble_prompt(sections)


def _load_version_prompt(domain: str, version_id: str) -> tuple[str, list[dict]]:
    """Load a version's prompt_text and changes metadata from its YAML."""
    path = DATA_DIR / domain / "versions" / f"{version_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["prompt_text"], data.get("changes", [])


# ============================================================
# Basic / edge-case tests
# ============================================================

class TestIdenticalPrompts:
    def test_returns_empty(self):
        prompt = "<role>\nHello\n</role>\n<format>\nJSON\n</format>"
        assert classify_changes(prompt, prompt) == []


class TestNoXmlTags:
    def test_fallback_to_whole_diff(self):
        old = "You are a helpful assistant. Respond in JSON."
        new = "You are a coding assistant. Respond in YAML."
        result = classify_changes(old, new)
        assert len(result) == 1
        assert result[0].change.unit_type == "unknown"
        assert result[0].change.change_type == "modified"
        assert result[0].change.magnitude > 0


class TestSectionInserted:
    def test_new_section_detected(self):
        old = "<role>\nBot\n</role>"
        new = "<role>\nBot\n</role>\n\n<policy>\nBe safe.\n</policy>"
        result = classify_changes(old, new)
        types = {c.change.unit_type for c in result}
        assert "policy" in types
        policy_change = [c for c in result if c.change.unit_type == "policy"][0]
        assert policy_change.change.change_type == "inserted"


class TestSectionDeleted:
    def test_removed_section_detected(self):
        old = "<role>\nBot\n</role>\n\n<policy>\nBe safe.\n</policy>"
        new = "<role>\nBot\n</role>"
        result = classify_changes(old, new)
        policy_change = [c for c in result if c.change.unit_type == "policy"][0]
        assert policy_change.change.change_type == "deleted"


# ============================================================
# Key extraction tests
# ============================================================

class TestExtractJsonKeys:
    def test_extracts_keys(self):
        text = '{\n  "action": "string",\n  "priority": "string"\n}'
        keys = _extract_json_keys(text)
        assert keys == {"action", "priority"}

    def test_empty_text(self):
        assert _extract_json_keys("") == set()


class TestExtractDemoLabel:
    def test_standard_format(self):
        assert _extract_demo_label("Example 1 (billing):") == "billing"
        assert _extract_demo_label("Example 5 (complaint):") == "complaint"

    def test_no_label(self):
        assert _extract_demo_label("Some random text") == ""


# ============================================================
# Real version tests (domain_a)
# ============================================================

class TestDomainAKeyRename:
    """v01: renames 'action' to 'task_type' across format/demos/policy."""

    def test_detects_format_modified(self):
        base = _load_base_prompt("domain_a")
        v_prompt, v_changes = _load_version_prompt("domain_a", "v01")
        result = classify_changes(base, v_prompt)
        # Should detect modifications (format key rename touches multiple sections)
        assert len(result) >= 1
        unit_types = {c.change.unit_type for c in result}
        assert "format" in unit_types or "demonstration" in unit_types or "policy" in unit_types

        # The format change should detect affected keys
        format_changes = [c for c in result if c.change.unit_type == "format"]
        if format_changes:
            keys = format_changes[0].affected_keys
            assert "task_type" in keys or "action" in keys

    def test_magnitude_reasonable(self):
        base = _load_base_prompt("domain_a")
        v_prompt, _ = _load_version_prompt("domain_a", "v01")
        result = classify_changes(base, v_prompt)
        for c in result:
            assert 0 < c.change.magnitude < 1.0


class TestDomainADemoRemoval:
    """v16: removes example 1 (billing) from demonstrations."""

    def test_detects_demo_deleted(self):
        base = _load_base_prompt("domain_a")
        v_prompt, v_changes = _load_version_prompt("domain_a", "v16")
        result = classify_changes(base, v_prompt)
        demo_changes = [
            c for c in result
            if c.change.unit_type in ("demonstration", "demo_item")
        ]
        assert len(demo_changes) >= 1
        change_types = {c.change.change_type for c in demo_changes}
        # Should detect a deletion
        assert "deleted" in change_types

    def test_affected_demo_labels_populated(self):
        base = _load_base_prompt("domain_a")
        v_prompt, _ = _load_version_prompt("domain_a", "v16")
        result = classify_changes(base, v_prompt)
        demo_changes = [
            c for c in result
            if c.change.unit_type in ("demonstration", "demo_item")
        ]
        all_labels = []
        for c in demo_changes:
            all_labels.extend(c.affected_demo_labels)
        assert len(all_labels) > 0


class TestDomainAWorkflowReorder:
    """v37: swaps workflow steps 1 and 2."""

    def test_detects_reorder(self):
        if not (DATA_DIR / "domain_a" / "versions" / "v37.yaml").exists():
            pytest.skip("v37 not present in domain_a versions")
        base = _load_base_prompt("domain_a")
        v_prompt, _ = _load_version_prompt("domain_a", "v37")
        result = classify_changes(base, v_prompt)
        wf_changes = [c for c in result if c.change.unit_type == "workflow"]
        assert len(wf_changes) >= 1
        assert wf_changes[0].change.change_type == "reordered"


class TestDomainACompound:
    """v48: multiple changes (format + demo + policy)."""

    def test_detects_multiple_changes(self):
        base = _load_base_prompt("domain_a")
        v_prompt, v_changes = _load_version_prompt("domain_a", "v48")
        result = classify_changes(base, v_prompt)
        unit_types = {c.change.unit_type for c in result}
        # v48 has format, demonstration, and policy changes
        assert len(unit_types) >= 2


class TestDomainAPolicyChange:
    """v30: strengthens a policy phrase."""

    def test_detects_policy_modified(self):
        if not (DATA_DIR / "domain_a" / "versions" / "v30.yaml").exists():
            pytest.skip("v30 not present in domain_a versions")
        base = _load_base_prompt("domain_a")
        v_prompt, _ = _load_version_prompt("domain_a", "v30")
        result = classify_changes(base, v_prompt)
        policy_changes = [c for c in result if c.change.unit_type == "policy"]
        assert len(policy_changes) >= 1
        assert policy_changes[0].change.change_type == "modified"


# ============================================================
# Real version tests (domain_b)
# ============================================================

class TestDomainBFormatRename:
    """v01: renames '## Explanation' to '## Analysis' across sections."""

    def test_detects_changes(self):
        base = _load_base_prompt("domain_b")
        v_prompt, _ = _load_version_prompt("domain_b", "v01")
        result = classify_changes(base, v_prompt)
        assert len(result) >= 1
        unit_types = {c.change.unit_type for c in result}
        assert len(unit_types) >= 1


class TestDomainBDemoRemoval:
    """v16: removes example 1 from domain_b demonstrations."""

    def test_detects_deletion(self):
        base = _load_base_prompt("domain_b")
        v_prompt, _ = _load_version_prompt("domain_b", "v16")
        result = classify_changes(base, v_prompt)
        demo_changes = [
            c for c in result
            if c.change.unit_type in ("demonstration", "demo_item")
        ]
        assert len(demo_changes) >= 1
        assert any(c.change.change_type == "deleted" for c in demo_changes)


class TestDomainBWorkflowReorder:
    """v37: swaps workflow steps in domain_b."""

    def test_detects_reorder(self):
        base = _load_base_prompt("domain_b")
        v_prompt, _ = _load_version_prompt("domain_b", "v37")
        result = classify_changes(base, v_prompt)
        wf_changes = [c for c in result if c.change.unit_type == "workflow"]
        assert len(wf_changes) >= 1
        assert wf_changes[0].change.change_type == "reordered"
