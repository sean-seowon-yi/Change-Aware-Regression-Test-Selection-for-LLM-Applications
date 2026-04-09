"""Learned selector: compose classifier + tagger + learned predictor + sentinel.

Thin wrapper that replaces the rule-based predictor in the Phase 2 selector
with the Phase 3 learned model while reusing every other pipeline component.
"""

from __future__ import annotations

from pathlib import Path

from src.phase1.models import TestCase
from src.phase1.prompts.parser import parse_prompt
from src.phase2.change_classifier import classify_changes
from src.phase2.selector import SelectionResult
from src.phase2.sentinel import sample_sentinels
from src.phase2.test_tagger import tag_tests
from src.phase3.learned_predictor import predict_impacted


def select_tests_learned(
    old_prompt: str,
    new_prompt: str,
    test_cases: list[TestCase],
    *,
    model_path: str | Path,
    threshold: float | None = None,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int | None = None,
    sensitivity_profiles: dict | None = None,
) -> SelectionResult:
    """Full learned-model selection pipeline.

    Pipeline:
        1. classify_changes(old, new)
        2. tag_tests(test_cases, demos_text)
        3. learned predict_impacted(changes, tagged, model)
        4. sample_sentinels from non-predicted pool
        5. combine → SelectionResult

    All inputs are available in a real deployment.
    """
    old_sections = parse_prompt(old_prompt)
    demos_text = old_sections.get("demonstrations", "")

    changes = classify_changes(old_prompt, new_prompt)
    tagged = tag_tests(test_cases, demos_text)

    prediction = predict_impacted(
        changes,
        tagged,
        model_path=model_path,
        threshold=threshold,
        magnitude_threshold=magnitude_threshold,
        sensitivity_profiles=sensitivity_profiles,
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
