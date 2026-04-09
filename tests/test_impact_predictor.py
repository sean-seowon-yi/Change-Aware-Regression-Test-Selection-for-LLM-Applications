"""Unit tests for src/phase2/impact_predictor.py."""

from __future__ import annotations

import pytest

from src.phase1.models import Change
from src.phase2.change_classifier import ClassifiedChange
from src.phase2.test_tagger import TaggedTest
from src.phase2.impact_predictor import predict_impacted, PredictionResult


# ---------------------------------------------------------------------------
# Helpers: build synthetic tagged tests
# ---------------------------------------------------------------------------

def _make_tagged(
    test_id: str,
    category: str,
    *,
    tags: list[str] | None = None,
    sensitive_to: list[str] | None = None,
    key_deps: list[str] | None = None,
    demo_deps: list[str] | None = None,
) -> TaggedTest:
    return TaggedTest(
        test_id=test_id,
        input_text="test input",
        monitor_type="schema",
        tags=tags or [],
        sensitive_to=sensitive_to or [],
        inferred_key_deps=key_deps or [],
        inferred_demo_deps=demo_deps or [],
        sensitivity_category=category,
    )


def _make_change(
    unit_type: str,
    change_type: str = "modified",
    magnitude: float = 0.10,
    affected_keys: list[str] | None = None,
    affected_demo_labels: list[str] | None = None,
) -> ClassifiedChange:
    return ClassifiedChange(
        change=Change(
            unit_id=f"{unit_type}_section",
            unit_type=unit_type,
            change_type=change_type,
            magnitude=magnitude,
            content_diff=f"Test {unit_type} change",
        ),
        affected_keys=affected_keys or [],
        affected_demo_labels=affected_demo_labels or [],
    )


# Shared fixture: a small, representative tagged test suite
@pytest.fixture
def suite() -> list[TaggedTest]:
    return [
        _make_tagged("t_fmt_1", "format", tags=["schema_sensitive"],
                      key_deps=["action", "priority"]),
        _make_tagged("t_fmt_2", "format", tags=["schema_sensitive"],
                      key_deps=["customer_id", "summary"]),
        _make_tagged("t_fmt_3", "format", tags=["format_sensitive"],
                      key_deps=[]),
        _make_tagged("t_demo_1", "demo", tags=["domain_coverage"],
                      demo_deps=["billing"]),
        _make_tagged("t_demo_2", "demo", tags=["domain_coverage"],
                      demo_deps=["refund"]),
        _make_tagged("t_policy_1", "policy", tags=["safety_sensitive"]),
        _make_tagged("t_policy_2", "policy", tags=["safety_sensitive"]),
        _make_tagged("t_wf_1", "workflow", tags=["workflow_sensitive"],
                      sensitive_to=["workflow"]),
        _make_tagged("t_wf_2", "workflow", tags=["workflow_sensitive"],
                      sensitive_to=["workflow"]),
        _make_tagged("t_general", "general", tags=["edge_case"]),
    ]


# ===========================================================================
# Format change tests
# ===========================================================================

class TestFormatChange:
    def test_selects_all_format_sensitive_when_no_specific_keys(self, suite):
        change = _make_change("format", affected_keys=[])
        result = predict_impacted([change], suite)
        assert "t_fmt_1" in result.predicted_ids
        assert "t_fmt_2" in result.predicted_ids
        assert "t_fmt_3" in result.predicted_ids
        assert "t_demo_1" not in result.predicted_ids
        assert "t_policy_1" not in result.predicted_ids

    def test_fine_grained_key_matching(self, suite):
        change = _make_change("format", affected_keys=["action"])
        result = predict_impacted([change], suite)
        assert "t_fmt_1" in result.predicted_ids
        assert "t_fmt_3" in result.predicted_ids  # structural test, no key_deps
        assert "t_fmt_2" not in result.predicted_ids  # has key_deps but no overlap

    def test_key_overlap_from_non_format_category(self, suite):
        extra = _make_tagged("t_extra", "demo", key_deps=["action"])
        tests = suite + [extra]
        change = _make_change("format", affected_keys=["action"])
        result = predict_impacted([change], tests)
        assert "t_extra" in result.predicted_ids

    def test_selects_via_sensitive_to_format(self):
        """A test not categorised as format but with 'format' in sensitive_to."""
        tests = [
            _make_tagged("t_x", "demo", sensitive_to=["demonstration", "format"]),
            _make_tagged("t_y", "demo", sensitive_to=["demonstration"]),
        ]
        change = _make_change("format", affected_keys=[])
        result = predict_impacted([change], tests)
        assert "t_x" in result.predicted_ids
        assert "t_y" not in result.predicted_ids


# ===========================================================================
# Demonstration change tests
# ===========================================================================

class TestDemonstrationChange:
    def test_selects_all_demo_sensitive(self, suite):
        change = _make_change("demonstration", change_type="deleted",
                               affected_demo_labels=["billing"])
        result = predict_impacted([change], suite)
        assert "t_demo_1" in result.predicted_ids
        assert "t_demo_2" in result.predicted_ids

    def test_selects_via_sensitive_to(self):
        tests = [
            _make_tagged("t_x", "general",
                          sensitive_to=["demonstration", "demo:ex1"]),
        ]
        change = _make_change("demonstration", change_type="inserted")
        result = predict_impacted([change], tests)
        assert "t_x" in result.predicted_ids


class TestDemoItemChange:
    def test_fine_grained_label_matching(self, suite):
        change = _make_change("demo_item", affected_demo_labels=["billing"])
        result = predict_impacted([change], suite)
        assert "t_demo_1" in result.predicted_ids
        assert "t_demo_2" not in result.predicted_ids

    def test_structural_demo_test_always_selected(self):
        """A demo test with no specific deps should be selected even when
        fine-grained matching succeeds for other tests."""
        tests = [
            _make_tagged("t_specific", "demo", demo_deps=["billing"]),
            _make_tagged("t_structural", "demo",
                          sensitive_to=["demonstration"], demo_deps=[]),
        ]
        change = _make_change("demo_item", affected_demo_labels=["billing"])
        result = predict_impacted([change], tests)
        assert "t_specific" in result.predicted_ids
        assert "t_structural" in result.predicted_ids

    def test_skips_demo_test_with_non_overlapping_deps(self):
        """A demo test checking a different label should be skipped."""
        tests = [
            _make_tagged("t_billing", "demo", demo_deps=["billing"]),
            _make_tagged("t_refund", "demo", demo_deps=["refund"]),
        ]
        change = _make_change("demo_item", affected_demo_labels=["billing"])
        result = predict_impacted([change], tests)
        assert "t_billing" in result.predicted_ids
        assert "t_refund" not in result.predicted_ids

    def test_no_labels_selects_all_demo_sensitive(self):
        """A demo_item change with empty labels selects all demo tests."""
        tests = [
            _make_tagged("t_demo_a", "demo", demo_deps=["billing"]),
            _make_tagged("t_demo_b", "demo", demo_deps=["refund"]),
            _make_tagged("t_other", "format"),
        ]
        change = _make_change("demo_item", affected_demo_labels=[])
        result = predict_impacted([change], tests)
        assert "t_demo_a" in result.predicted_ids
        assert "t_demo_b" in result.predicted_ids
        assert "t_other" not in result.predicted_ids


# ===========================================================================
# Policy change tests
# ===========================================================================

class TestPolicyChange:
    def test_selects_policy_sensitive(self, suite):
        change = _make_change("policy")
        result = predict_impacted([change], suite)
        assert "t_policy_1" in result.predicted_ids
        assert "t_policy_2" in result.predicted_ids
        assert "t_fmt_1" not in result.predicted_ids

    def test_selects_safety_tagged(self):
        tests = [
            _make_tagged("t_safe", "general", tags=["safety_sensitive"]),
            _make_tagged("t_other", "general"),
        ]
        change = _make_change("policy")
        result = predict_impacted([change], tests)
        assert "t_safe" in result.predicted_ids
        assert "t_other" not in result.predicted_ids

    def test_selects_via_sensitive_to_policy(self):
        """A test not categorised as policy but with 'policy' in sensitive_to."""
        tests = [
            _make_tagged("t_x", "demo", sensitive_to=["demonstration", "policy"]),
            _make_tagged("t_y", "demo", sensitive_to=["demonstration"]),
        ]
        change = _make_change("policy")
        result = predict_impacted([change], tests)
        assert "t_x" in result.predicted_ids
        assert "t_y" not in result.predicted_ids


# ===========================================================================
# Workflow change tests
# ===========================================================================

class TestWorkflowChange:
    def test_selects_workflow_sensitive(self, suite):
        change = _make_change("workflow", change_type="reordered")
        result = predict_impacted([change], suite)
        assert "t_wf_1" in result.predicted_ids
        assert "t_wf_2" in result.predicted_ids
        assert "t_fmt_1" not in result.predicted_ids

    def test_selects_via_sensitive_to_workflow(self):
        tests = [
            _make_tagged("t_x", "format", sensitive_to=["format", "workflow"]),
            _make_tagged("t_y", "format", sensitive_to=["format"]),
        ]
        change = _make_change("workflow")
        result = predict_impacted([change], tests)
        assert "t_x" in result.predicted_ids
        assert "t_y" not in result.predicted_ids


# ===========================================================================
# Role change tests
# ===========================================================================

class TestRoleChange:
    def test_selects_all_tests(self, suite):
        change = _make_change("role")
        result = predict_impacted([change], suite)
        assert result.predicted_ids == {t.test_id for t in suite}


# ===========================================================================
# Compound change tests
# ===========================================================================

class TestCompoundChange:
    def test_union_of_format_and_policy(self, suite):
        fmt = _make_change("format", affected_keys=[])
        pol = _make_change("policy")
        result = predict_impacted([fmt, pol], suite)
        assert "t_fmt_1" in result.predicted_ids
        assert "t_policy_1" in result.predicted_ids
        assert "t_demo_1" not in result.predicted_ids

    def test_compound_accumulates_reasons(self, suite):
        fmt = _make_change("format", affected_keys=[])
        pol = _make_change("policy")
        result = predict_impacted([fmt, pol], suite)
        assert len(result.reasons.get("t_fmt_1", [])) >= 1


# ===========================================================================
# Magnitude gating tests
# ===========================================================================

class TestMagnitudeGating:
    def test_below_threshold_skipped(self, suite):
        change = _make_change("format", magnitude=0.001)
        result = predict_impacted([change], suite, magnitude_threshold=0.005)
        assert len(result.predicted_ids) == 0

    def test_above_threshold_not_skipped(self, suite):
        change = _make_change("format", magnitude=0.01)
        result = predict_impacted([change], suite, magnitude_threshold=0.005)
        assert len(result.predicted_ids) > 0

    def test_zero_threshold_allows_all(self, suite):
        change = _make_change("format", magnitude=0.0001)
        result = predict_impacted([change], suite, magnitude_threshold=0.0)
        assert len(result.predicted_ids) > 0


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    def test_empty_changes(self, suite):
        result = predict_impacted([], suite)
        assert len(result.predicted_ids) == 0

    def test_empty_tests(self):
        change = _make_change("format")
        result = predict_impacted([change], [])
        assert len(result.predicted_ids) == 0

    def test_unknown_type_selects_all(self, suite):
        change = _make_change("unknown")
        result = predict_impacted([change], suite)
        assert result.predicted_ids == {t.test_id for t in suite}

    def test_reasons_populated(self, suite):
        change = _make_change("policy")
        result = predict_impacted([change], suite)
        for tid in result.predicted_ids:
            assert tid in result.reasons
            assert len(result.reasons[tid]) > 0
