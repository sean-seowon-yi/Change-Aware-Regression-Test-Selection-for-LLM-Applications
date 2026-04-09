"""Selector: compose classifier + tagger + predictor + sentinel.

Top-level entry point that takes two prompt texts and a test suite,
and returns the final set of tests to run along with full metadata.

All inputs are available in a real CI/CD deployment:
- The old (production) prompt text
- The new (PR) prompt text
- The eval suite (test cases with metadata)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.phase1.models import TestCase
from src.phase1.prompts.parser import parse_prompt
from src.phase2.change_classifier import ClassifiedChange, classify_changes
from src.phase2.test_tagger import TaggedTest, tag_tests
from src.phase2.impact_predictor import PredictionResult, predict_impacted
from src.phase2.sentinel import sample_sentinels


@dataclass
class SelectionResult:
    """Complete output of the test-selection pipeline."""

    selected_ids: set[str] = field(default_factory=set)
    predicted_ids: set[str] = field(default_factory=set)
    sentinel_ids: set[str] = field(default_factory=set)
    changes: list[ClassifiedChange] = field(default_factory=list)
    total_tests: int = 0
    call_reduction: float = 0.0
    metadata: dict = field(default_factory=dict)


def select_tests(
    old_prompt: str,
    new_prompt: str,
    test_cases: list[TestCase],
    *,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int | None = None,
) -> SelectionResult:
    """Full selection pipeline: classify -> tag -> predict -> sentinel -> combine.

    Args:
        old_prompt: The current production prompt (P).
        new_prompt: The proposed new prompt (P').
        test_cases: Raw test cases from eval_suite.yaml.
        sentinel_fraction: Fraction of non-predicted tests to sample as sentinels.
        magnitude_threshold: Skip changes below this magnitude.
        sentinel_seed: RNG seed for sentinel sampling reproducibility.

    Returns:
        SelectionResult with selected IDs, breakdown, and metadata.
    """
    old_sections = parse_prompt(old_prompt)
    demos_text = old_sections.get("demonstrations", "")

    changes = classify_changes(old_prompt, new_prompt)

    tagged = tag_tests(test_cases, demos_text)

    prediction = predict_impacted(
        changes, tagged, magnitude_threshold=magnitude_threshold
    )

    all_ids = {tc.test_id for tc in test_cases}
    sentinels = sample_sentinels(
        all_ids,
        prediction.predicted_ids,
        fraction=sentinel_fraction,
        seed=sentinel_seed,
    )

    selected = prediction.predicted_ids | sentinels
    total = len(test_cases)
    reduction = 1.0 - len(selected) / total if total > 0 else 0.0

    return SelectionResult(
        selected_ids=selected,
        predicted_ids=prediction.predicted_ids,
        sentinel_ids=sentinels,
        changes=changes,
        total_tests=total,
        call_reduction=reduction,
        metadata={"reasons": prediction.reasons},
    )
