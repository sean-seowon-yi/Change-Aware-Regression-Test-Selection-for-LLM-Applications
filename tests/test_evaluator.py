"""Tests for src/phase2/evaluator.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.phase1.models import TestCase
from src.phase2.evaluator import (
    GroundTruth,
    VersionMetrics,
    _magnitude_bucket,
    _primary_unit_type,
    _version_to_category,
    _evaluate_baseline,
    _aggregate_group,
    aggregate_results,
    baseline_full_rerun,
    baseline_monitor_heuristic,
    baseline_random_50,
    load_ground_truth,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cases(n: int = 10) -> list[TestCase]:
    """Create n dummy test cases with varied monitor types."""
    monitor_types = ["schema", "required_keys", "keyword_presence", "policy", "format"]
    return [
        TestCase(
            test_id=f"t_{i:03d}",
            input_text=f"input {i}",
            monitor_type=monitor_types[i % len(monitor_types)],
            tags=[],
            sensitive_to=[],
        )
        for i in range(1, n + 1)
    ]


def _make_version_metrics(
    version_id: str = "v01",
    domain: str = "domain_a",
    recall_pred: float = 0.8,
    recall_sent: float = 0.9,
    call_red: float = 0.4,
    for_rate: float = 0.1,
    sentinel_hit: bool = False,
    num_impacted: int = 10,
    changes: list[dict] | None = None,
    change_types_gt: list[str] | None = None,
    mag_max: float = 0.15,
) -> VersionMetrics:
    if changes is None:
        changes = [{"unit_type": "format", "change_type": "modified", "magnitude": 0.15}]
    if change_types_gt is None:
        change_types_gt = ["format:modified"]
    return VersionMetrics(
        version_id=version_id,
        domain=domain,
        num_impacted=num_impacted,
        num_predicted=6,
        num_sentinel=3,
        num_selected=9,
        total_tests=20,
        recall_predictor=recall_pred,
        recall_with_sentinel=recall_sent,
        call_reduction=call_red,
        false_omission_rate=for_rate,
        false_omissions=["t_011"],
        sentinel_hit=sentinel_hit,
        changes_detected=changes,
        change_types_gt=change_types_gt,
        magnitude_max=mag_max,
        mutation_category=_version_to_category(version_id),
    )


# ---------------------------------------------------------------------------
# version_to_category
# ---------------------------------------------------------------------------

class TestVersionToCategory:
    def test_mechanical_range(self):
        for vid in ["v01", "v25", "v50"]:
            assert _version_to_category(vid) == "mechanical"

    def test_llm_generated_range(self):
        for vid in ["v51", "v60", "v70"]:
            assert _version_to_category(vid) == "llm_generated"


# ---------------------------------------------------------------------------
# magnitude_bucket
# ---------------------------------------------------------------------------

class TestMagnitudeBucket:
    def test_low(self):
        assert _magnitude_bucket(0.0) == "low"
        assert _magnitude_bucket(0.049) == "low"

    def test_medium(self):
        assert _magnitude_bucket(0.05) == "medium"
        assert _magnitude_bucket(0.20) == "medium"

    def test_high(self):
        assert _magnitude_bucket(0.21) == "high"
        assert _magnitude_bucket(1.0) == "high"


# ---------------------------------------------------------------------------
# primary_unit_type
# ---------------------------------------------------------------------------

class TestPrimaryUnitType:
    def test_empty(self):
        assert _primary_unit_type([]) == "none"

    def test_single(self):
        assert _primary_unit_type([{"unit_type": "format"}]) == "format"

    def test_compound(self):
        result = _primary_unit_type([
            {"unit_type": "format"},
            {"unit_type": "policy"},
        ])
        assert result == "compound"

    def test_duplicate_same_type(self):
        result = _primary_unit_type([
            {"unit_type": "format"},
            {"unit_type": "format"},
        ])
        assert result == "format"


# ---------------------------------------------------------------------------
# _evaluate_baseline
# ---------------------------------------------------------------------------

class TestEvaluateBaseline:
    def test_perfect_recall(self):
        selected = {"a", "b", "c"}
        impacted = {"a", "b"}
        result = _evaluate_baseline(selected, impacted, 5)
        assert result["recall"] == 1.0
        assert result["call_reduction"] == pytest.approx(0.4)

    def test_partial_recall(self):
        selected = {"a"}
        impacted = {"a", "b"}
        result = _evaluate_baseline(selected, impacted, 4)
        assert result["recall"] == 0.5
        # FOR = missed impacted among not-selected: 1 / (4 - 1)
        assert result["false_omission_rate"] == pytest.approx(1.0 / 3.0)

    def test_no_impacted(self):
        result = _evaluate_baseline({"a"}, set(), 5)
        assert result["recall"] == 1.0
        assert result["false_omission_rate"] == 0.0

    def test_empty_selected(self):
        result = _evaluate_baseline(set(), {"a", "b"}, 5)
        assert result["recall"] == 0.0
        assert result["call_reduction"] == 1.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselineFullRerun:
    def test_selects_all(self):
        cases = _make_cases(10)
        result = baseline_full_rerun(cases)
        assert result == {tc.test_id for tc in cases}


class TestBaselineRandom50:
    def test_selects_half(self):
        cases = _make_cases(20)
        result = baseline_random_50(cases, seed=42)
        assert len(result) == 10

    def test_deterministic(self):
        cases = _make_cases(20)
        r1 = baseline_random_50(cases, seed=42)
        r2 = baseline_random_50(cases, seed=42)
        assert r1 == r2

    def test_different_seed_differs(self):
        cases = _make_cases(20)
        r1 = baseline_random_50(cases, seed=42)
        r2 = baseline_random_50(cases, seed=99)
        assert r1 != r2


class TestBaselineMonitorHeuristic:
    def test_format_selects_schema_and_format(self):
        cases = _make_cases(10)
        result = baseline_monitor_heuristic(cases, ["format"])
        for tid in result:
            tc = next(c for c in cases if c.test_id == tid)
            assert tc.monitor_type in {"schema", "required_keys", "format"}

    def test_policy_selects_policy(self):
        cases = _make_cases(10)
        result = baseline_monitor_heuristic(cases, ["policy"])
        for tid in result:
            tc = next(c for c in cases if c.test_id == tid)
            assert tc.monitor_type == "policy"

    def test_role_selects_all(self):
        cases = _make_cases(10)
        result = baseline_monitor_heuristic(cases, ["role"])
        assert len(result) == 10

    def test_compound_unions(self):
        cases = _make_cases(10)
        result = baseline_monitor_heuristic(cases, ["format", "policy"])
        for tid in result:
            tc = next(c for c in cases if c.test_id == tid)
            assert tc.monitor_type in {"schema", "required_keys", "format", "policy"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregateGroup:
    def test_empty(self):
        assert _aggregate_group([]) == {}

    def test_single_metric(self):
        m = _make_version_metrics(recall_pred=0.8, recall_sent=0.9)
        result = _aggregate_group([m])
        assert result["count"] == 1
        assert result["mean_recall_predictor"] == 0.8
        assert result["mean_recall_with_sentinel"] == 0.9

    def test_mean_of_two(self):
        m1 = _make_version_metrics(recall_pred=0.8, call_red=0.4)
        m2 = _make_version_metrics(version_id="v02", recall_pred=1.0, call_red=0.6)
        result = _aggregate_group([m1, m2])
        assert result["count"] == 2
        assert result["mean_recall_predictor"] == pytest.approx(0.9, abs=0.01)
        assert result["mean_call_reduction"] == pytest.approx(0.5, abs=0.01)

    def test_sentinel_catch_rate(self):
        m1 = _make_version_metrics(
            recall_pred=0.5, sentinel_hit=True, num_impacted=5
        )
        m2 = _make_version_metrics(
            version_id="v02", recall_pred=0.7, sentinel_hit=False, num_impacted=3
        )
        result = _aggregate_group([m1, m2])
        assert result["sentinel_catch_rate"] == pytest.approx(0.5)

    def test_perfect_recall_count(self):
        m1 = _make_version_metrics(recall_sent=1.0)
        m2 = _make_version_metrics(version_id="v02", recall_sent=0.5)
        result = _aggregate_group([m1, m2])
        assert result["versions_with_perfect_recall"] == 1


# ---------------------------------------------------------------------------
# Zero impacted edge case
# ---------------------------------------------------------------------------

class TestZeroImpacted:
    def test_recall_is_1_when_no_impacted(self):
        m = _make_version_metrics(
            num_impacted=0,
            recall_pred=1.0,
            recall_sent=1.0,
            for_rate=0.0,
        )
        assert m.recall_predictor == 1.0
        assert m.recall_with_sentinel == 1.0
        assert m.false_omission_rate == 0.0


# ---------------------------------------------------------------------------
# Ground truth loader (with temp file)
# ---------------------------------------------------------------------------

class TestLoadGroundTruth:
    def test_loads_correctly(self, tmp_path: Path):
        gt_file = tmp_path / "ground_truth_test.jsonl"
        rows = [
            {"domain": "test", "version_id": "v01", "test_id": "t_001",
             "outcome_base": "pass", "outcome_version": "fail",
             "impacted": True, "change_types": ["format:modified"]},
            {"domain": "test", "version_id": "v01", "test_id": "t_002",
             "outcome_base": "pass", "outcome_version": "pass",
             "impacted": False, "change_types": ["format:modified"]},
            {"domain": "test", "version_id": "v02", "test_id": "t_001",
             "outcome_base": "pass", "outcome_version": "pass",
             "impacted": False, "change_types": ["policy:modified"]},
        ]
        gt_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with patch("src.phase2.evaluator._RESULTS_DIR", tmp_path):
            (tmp_path / "baseline").mkdir()
            gt_real = tmp_path / "baseline" / "ground_truth_test.jsonl"
            gt_real.write_text(gt_file.read_text())
            gt = load_ground_truth("test")

        assert gt.impacted_by_version["v01"] == {"t_001"}
        assert gt.impacted_by_version["v02"] == set()
        assert gt.change_types_by_version["v01"] == ["format:modified"]
        assert gt.change_types_by_version["v02"] == ["policy:modified"]


# ---------------------------------------------------------------------------
# Full aggregation with baselines
# ---------------------------------------------------------------------------

class TestAggregateResults:
    def test_produces_all_sections(self):
        cases = _make_cases(20)
        m1 = _make_version_metrics(version_id="v01", mag_max=0.03)
        m2 = _make_version_metrics(
            version_id="v55", mag_max=0.25,
            changes=[{"unit_type": "policy", "change_type": "modified", "magnitude": 0.25}],
            change_types_gt=["policy:modified"],
        )
        gt = GroundTruth(
            impacted_by_version={"v01": {"t_001"}, "v55": {"t_002", "t_003"}},
            change_types_by_version={"v01": ["format:modified"], "v55": ["policy:modified"]},
        )
        result = aggregate_results([m1, m2], cases, gt)

        assert "aggregate" in result
        assert "by_change_type" in result
        assert "by_magnitude_bucket" in result
        assert "by_category" in result
        assert "baselines" in result

        assert result["aggregate"]["total_versions"] == 2
        assert "format" in result["by_change_type"]
        assert "policy" in result["by_change_type"]
        assert "low" in result["by_magnitude_bucket"]
        assert "high" in result["by_magnitude_bucket"]
        assert "mechanical" in result["by_category"]
        assert "llm_generated" in result["by_category"]

        assert "full_rerun" in result["baselines"]
        assert "random_50" in result["baselines"]
        assert "monitor_heuristic" in result["baselines"]
