"""Tests for src/phase3/learned_predictor.py and src/phase3/learned_selector.py"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.phase1.models import TestCase
from src.phase2.change_classifier import ClassifiedChange, classify_changes
from src.phase2.impact_predictor import PredictionResult
from src.phase2.selector import SelectionResult
from src.phase2.test_tagger import TaggedTest, tag_tests


# ---------------------------------------------------------------------------
# Skip guard: require a trained model to exist
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _PROJECT_ROOT / "models" / "lightgbm_domain_b_temporal.pkl"
_HAS_MODEL = _MODEL_PATH.exists()

pytestmark = pytest.mark.skipif(not _HAS_MODEL, reason="No trained model found")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def domain_b_data():
    from src.phase2.evaluator import load_base_prompt, load_test_cases, load_version_prompt
    base = load_base_prompt("domain_b")
    test_cases = load_test_cases("domain_b")
    v05 = load_version_prompt("domain_b", "v05")
    return base, v05, test_cases


@pytest.fixture
def changes_and_tagged(domain_b_data):
    base, version, test_cases = domain_b_data
    from src.phase1.prompts.parser import parse_prompt
    changes = classify_changes(base, version)
    demos = parse_prompt(base).get("demonstrations", "")
    tagged = tag_tests(test_cases, demos)
    return changes, tagged, test_cases


# ---------------------------------------------------------------------------
# learned_predictor.predict_impacted
# ---------------------------------------------------------------------------

class TestLearnedPredictor:
    def test_returns_prediction_result(self, changes_and_tagged):
        from src.phase3.learned_predictor import predict_impacted
        changes, tagged, _ = changes_and_tagged
        result = predict_impacted(
            changes, tagged, model_path=_MODEL_PATH, threshold=0.20,
        )
        assert isinstance(result, PredictionResult)
        assert isinstance(result.predicted_ids, set)
        assert isinstance(result.reasons, dict)

    def test_predicted_ids_are_valid_test_ids(self, changes_and_tagged):
        from src.phase3.learned_predictor import predict_impacted
        changes, tagged, _ = changes_and_tagged
        all_ids = {t.test_id for t in tagged}
        result = predict_impacted(
            changes, tagged, model_path=_MODEL_PATH, threshold=0.20,
        )
        assert result.predicted_ids <= all_ids

    def test_reasons_contain_probability(self, changes_and_tagged):
        from src.phase3.learned_predictor import predict_impacted
        changes, tagged, _ = changes_and_tagged
        result = predict_impacted(
            changes, tagged, model_path=_MODEL_PATH, threshold=0.01,
        )
        for tid, reason_list in result.reasons.items():
            assert len(reason_list) > 0
            assert "P(impacted)" in reason_list[0]

    def test_threshold_affects_selection_size(self, changes_and_tagged):
        from src.phase3.learned_predictor import predict_impacted
        changes, tagged, _ = changes_and_tagged
        low = predict_impacted(changes, tagged, model_path=_MODEL_PATH, threshold=0.01)
        high = predict_impacted(changes, tagged, model_path=_MODEL_PATH, threshold=0.90)
        assert len(low.predicted_ids) >= len(high.predicted_ids)

    def test_deterministic(self, changes_and_tagged):
        from src.phase3.learned_predictor import predict_impacted
        changes, tagged, _ = changes_and_tagged
        r1 = predict_impacted(changes, tagged, model_path=_MODEL_PATH, threshold=0.20)
        r2 = predict_impacted(changes, tagged, model_path=_MODEL_PATH, threshold=0.20)
        assert r1.predicted_ids == r2.predicted_ids

    def test_empty_changes(self, changes_and_tagged):
        from src.phase3.learned_predictor import predict_impacted
        _, tagged, _ = changes_and_tagged
        result = predict_impacted([], tagged, model_path=_MODEL_PATH)
        assert len(result.predicted_ids) == 0

    def test_magnitude_threshold_skips(self, changes_and_tagged):
        from src.phase3.learned_predictor import predict_impacted
        changes, tagged, _ = changes_and_tagged
        result = predict_impacted(
            changes, tagged,
            model_path=_MODEL_PATH,
            threshold=0.01,
            magnitude_threshold=999.0,
        )
        assert len(result.predicted_ids) == 0


# ---------------------------------------------------------------------------
# learned_selector.select_tests_learned
# ---------------------------------------------------------------------------

class TestLearnedSelector:
    def test_returns_selection_result(self, domain_b_data):
        from src.phase3.learned_selector import select_tests_learned
        base, version, test_cases = domain_b_data
        result = select_tests_learned(
            base, version, test_cases,
            model_path=_MODEL_PATH,
            threshold=0.20,
            sentinel_seed=42,
        )
        assert isinstance(result, SelectionResult)
        assert result.total_tests == len(test_cases)

    def test_selected_superset_of_predicted(self, domain_b_data):
        from src.phase3.learned_selector import select_tests_learned
        base, version, test_cases = domain_b_data
        result = select_tests_learned(
            base, version, test_cases,
            model_path=_MODEL_PATH,
            sentinel_seed=42,
        )
        assert result.predicted_ids <= result.selected_ids

    def test_sentinels_from_non_predicted(self, domain_b_data):
        from src.phase3.learned_selector import select_tests_learned
        base, version, test_cases = domain_b_data
        result = select_tests_learned(
            base, version, test_cases,
            model_path=_MODEL_PATH,
            sentinel_seed=42,
        )
        assert not (result.sentinel_ids & result.predicted_ids)

    def test_call_reduction_consistent(self, domain_b_data):
        from src.phase3.learned_selector import select_tests_learned
        base, version, test_cases = domain_b_data
        result = select_tests_learned(
            base, version, test_cases,
            model_path=_MODEL_PATH,
            sentinel_seed=42,
        )
        expected_cr = 1.0 - len(result.selected_ids) / result.total_tests
        assert abs(result.call_reduction - expected_cr) < 1e-9
