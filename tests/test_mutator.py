"""Unit tests for src/phase1/prompts/mutator.py."""

from __future__ import annotations

import pytest

from src.phase1.prompts.loader import assemble_prompt, load_prompt_sections
from src.phase1.prompts.mutator import (
    _mutate_edit_demo_output,
    _mutate_policy_add_clause,
    _mutate_policy_remove_clause,
    _mutate_policy_replace,
    _mutate_remove_demo,
    _mutate_rename_key_a,
    _mutate_role_rephrase,
    _mutate_role_tone,
    _mutate_swap_demo,
    _mutate_workflow_add_step,
    _mutate_workflow_reorder,
    _mutate_workflow_remove_step,
    _split_demos,
    _token_edit_distance,
    generate_domain_a_versions,
    generate_domain_b_versions,
)


@pytest.fixture(scope="module")
def sections_a():
    return load_prompt_sections("domain_a")


@pytest.fixture(scope="module")
def sections_b():
    return load_prompt_sections("domain_b")


class TestTokenEditDistance:
    def test_identical(self):
        assert _token_edit_distance("hello world", "hello world") == 0.0

    def test_completely_different(self):
        assert _token_edit_distance("a b c", "x y z") == 1.0

    def test_empty_both(self):
        assert _token_edit_distance("", "") == 0.0

    def test_one_empty(self):
        assert _token_edit_distance("hello", "") == 1.0

    def test_partial_overlap(self):
        dist = _token_edit_distance("a b c d", "a b x d")
        assert 0 < dist < 1


class TestSplitDemos:
    def test_split_domain_a(self, sections_a):
        examples = _split_demos(sections_a["demonstrations"])
        assert len(examples) == 8
        assert "Example 1 (billing):" in examples[0]
        assert "Example 8 (escalation):" in examples[7]

    def test_split_domain_b(self, sections_b):
        examples = _split_demos(sections_b["demonstrations"])
        assert len(examples) == 6
        assert "Example 1" in examples[0]


class TestRenameKeyA:
    def test_renames_in_format(self, sections_a):
        s, changes = _mutate_rename_key_a(sections_a, "action", "task_type")
        assert '"task_type"' in s["format"]
        assert '"action"' not in s["format"]

    def test_renames_in_demos(self, sections_a):
        s, changes = _mutate_rename_key_a(sections_a, "action", "task_type")
        assert '"task_type"' in s["demonstrations"]

    def test_change_metadata(self, sections_a):
        _, changes = _mutate_rename_key_a(sections_a, "action", "task_type")
        assert len(changes) == 1
        assert changes[0].unit_type == "format"
        assert changes[0].change_type == "modified"
        assert 0 < changes[0].magnitude <= 1

    def test_renames_in_policy(self, sections_a):
        s, _ = _mutate_rename_key_a(sections_a, "action", "task_type")
        if '"action"' in sections_a.get("policy", ""):
            assert '"action"' not in s["policy"]
            assert '"task_type"' in s["policy"]

    def test_other_sections_unchanged(self, sections_a):
        s, _ = _mutate_rename_key_a(sections_a, "action", "task_type")
        assert s["role"] == sections_a["role"]
        assert s["workflow"] == sections_a["workflow"]


class TestRemoveDemo:
    def test_removes_one_example(self, sections_a):
        s, changes = _mutate_remove_demo(sections_a, 0, "domain_a")
        examples = _split_demos(s["demonstrations"])
        assert len(examples) == 7

    def test_renumbers_after_remove(self, sections_a):
        s, _ = _mutate_remove_demo(sections_a, 2, "domain_a")
        examples = _split_demos(s["demonstrations"])
        assert "Example 1" in examples[0]
        assert "Example 3" in examples[2]

    def test_change_type_is_deleted(self, sections_a):
        _, changes = _mutate_remove_demo(sections_a, 0, "domain_a")
        assert changes[0].change_type == "deleted"
        assert changes[0].unit_type == "demonstration"


class TestSwapDemo:
    def test_swaps_content(self, sections_a):
        s, changes = _mutate_swap_demo(sections_a, 0, "warranty", "Example 1 (warranty):\nTest swap", "domain_a")
        assert "warranty" in s["demonstrations"]

    def test_change_type_is_modified(self, sections_a):
        _, changes = _mutate_swap_demo(sections_a, 0, "warranty", "Example 1 (warranty):\nTest", "domain_a")
        assert changes[0].change_type == "modified"


class TestEditDemoOutput:
    def test_edits_content(self, sections_a):
        s, changes = _mutate_edit_demo_output(sections_a, 0, '"priority": "high"', '"priority": "critical"', "domain_a")
        assert '"priority": "critical"' in s["demonstrations"]

    def test_change_type(self, sections_a):
        _, changes = _mutate_edit_demo_output(sections_a, 0, '"priority": "high"', '"priority": "critical"', "domain_a")
        assert changes[0].unit_type == "demo_item"


class TestPolicyMutations:
    def test_strengthen(self, sections_a):
        s, changes = _mutate_policy_replace(
            sections_a,
            "Never include full credit card numbers",
            "You must NEVER include full credit card numbers",
            "strengthen",
        )
        assert "You must NEVER" in s["policy"]
        assert changes[0].unit_type == "policy"

    def test_add_clause(self, sections_a):
        s, changes = _mutate_policy_add_clause(sections_a, "\nNew rule: always be polite.")
        assert "always be polite" in s["policy"]
        assert changes[0].magnitude > 0

    def test_remove_clause(self, sections_a):
        s, changes = _mutate_policy_remove_clause(sections_a, "PII Handling:", "Safety and Refusal:")
        assert "PII Handling:" not in s["policy"]
        assert "Safety and Refusal:" in s["policy"]


class TestWorkflowMutations:
    def test_reorder(self, sections_a):
        s, changes = _mutate_workflow_reorder(sections_a, 0, 1)
        assert "Step 1" in s["workflow"]
        assert changes[0].change_type == "reordered"

    def test_add_step(self, sections_a):
        s, changes = _mutate_workflow_add_step(sections_a, "Step 5 — Log: Record the result.")
        assert "Step 5" in s["workflow"]

    def test_remove_step(self, sections_a):
        s, changes = _mutate_workflow_remove_step(sections_a, 3)
        assert changes[0].change_type == "deleted"


class TestRoleMutations:
    def test_rephrase(self, sections_a):
        s, changes = _mutate_role_rephrase(
            sections_a,
            "You are a customer service ticket extraction assistant.",
            "You are an AI system.",
        )
        assert "AI system" in s["role"]

    def test_tone_change(self, sections_a):
        s, changes = _mutate_role_tone(sections_a, "domain_a", "formal")
        assert s["role"] != sections_a["role"]
        assert changes[0].unit_type == "role"


class TestFullGeneration:
    def test_domain_a_produces_50_versions(self, sections_a):
        versions = generate_domain_a_versions(sections_a)
        assert len(versions) == 50

    def test_domain_b_produces_50_versions(self, sections_b):
        versions = generate_domain_b_versions(sections_b)
        assert len(versions) == 50

    def test_all_versions_have_valid_ids(self, sections_a):
        versions = generate_domain_a_versions(sections_a)
        ids = [v[0] for v in versions]
        assert ids == [f"v{i:02d}" for i in range(1, 51)]

    def test_all_versions_have_changes(self, sections_a):
        versions = generate_domain_a_versions(sections_a)
        for vid, _, changes in versions:
            assert len(changes) >= 1, f"{vid} has no changes"

    def test_magnitudes_in_range(self, sections_a):
        versions = generate_domain_a_versions(sections_a)
        for vid, _, changes in versions:
            for c in changes:
                assert 0 <= c.magnitude <= 1, f"{vid} change magnitude {c.magnitude} out of range"

    def test_mutated_prompts_assemble(self, sections_a):
        versions = generate_domain_a_versions(sections_a)
        for vid, mutated_sections, _ in versions:
            prompt = assemble_prompt(mutated_sections)
            assert len(prompt) > 100, f"{vid} assembled prompt too short"

    def test_mutated_prompts_differ_from_base(self, sections_a):
        base_prompt = assemble_prompt(sections_a)
        versions = generate_domain_a_versions(sections_a)
        for vid, mutated_sections, _ in versions:
            mutated_prompt = assemble_prompt(mutated_sections)
            assert mutated_prompt != base_prompt, f"{vid} prompt identical to base"
