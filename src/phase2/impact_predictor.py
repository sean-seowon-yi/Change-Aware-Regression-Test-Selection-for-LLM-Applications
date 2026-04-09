"""Rule-based impact predictor: map classified changes to predicted test sets.

Given a list of ClassifiedChange objects (from the change classifier) and a
list of TaggedTest objects (from the test tagger), predict which tests are
likely impacted by the changes.

All inputs are derived from observable prompt text and eval-suite metadata.
No ground truth, version YAML metadata, or impacted labels are used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.phase2.change_classifier import ClassifiedChange
from src.phase2.test_tagger import TaggedTest


@dataclass
class PredictionResult:
    """Output of the impact predictor."""

    predicted_ids: set[str] = field(default_factory=set)
    reasons: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Affinity rules
# ---------------------------------------------------------------------------

def _select_for_format(
    change: ClassifiedChange,
    tagged_tests: list[TaggedTest],
) -> dict[str, str]:
    """Return {test_id: reason} for a format-type change."""
    hits: dict[str, str] = {}
    affected = set(change.affected_keys)

    for t in tagged_tests:
        if t.sensitivity_category == "format" or "format" in t.sensitive_to:
            if not affected:
                hits[t.test_id] = f"format change (no specific keys) -> format-sensitive"
            elif not t.inferred_key_deps:
                hits[t.test_id] = f"format change [{', '.join(sorted(affected))}] -> structural format test"
            elif affected & set(t.inferred_key_deps):
                overlap = sorted(affected & set(t.inferred_key_deps))
                hits[t.test_id] = f"format change affects keys {overlap}"
            else:
                continue
        elif affected and affected & set(t.inferred_key_deps):
            overlap = sorted(affected & set(t.inferred_key_deps))
            hits[t.test_id] = f"format key overlap {overlap} (non-format-category test)"

    return hits


def _select_for_demonstration(
    change: ClassifiedChange,
    tagged_tests: list[TaggedTest],
) -> dict[str, str]:
    """Return {test_id: reason} for a whole-demonstration change (insert/delete/reorder)."""
    hits: dict[str, str] = {}
    labels = set(change.affected_demo_labels)

    for t in tagged_tests:
        if t.sensitivity_category == "demo":
            hits[t.test_id] = f"demonstration change -> demo-sensitive"
        elif "demonstration" in t.sensitive_to:
            hits[t.test_id] = f"demonstration change -> sensitive_to includes 'demonstration'"
        elif labels and labels & set(t.inferred_demo_deps):
            overlap = sorted(labels & set(t.inferred_demo_deps))
            hits[t.test_id] = f"demonstration change affects demos {overlap}"

    return hits


def _select_for_demo_item(
    change: ClassifiedChange,
    tagged_tests: list[TaggedTest],
) -> dict[str, str]:
    """Return {test_id: reason} for a single demo_item change.

    Analogous to _select_for_format: structural demo tests (empty
    inferred_demo_deps) are always selected, while tests with specific
    deps are only selected on label overlap.
    """
    hits: dict[str, str] = {}
    labels = set(change.affected_demo_labels)
    label_str = ", ".join(sorted(labels)) if labels else "?"

    for t in tagged_tests:
        is_demo = t.sensitivity_category == "demo" or "demonstration" in t.sensitive_to
        if is_demo:
            if not labels:
                hits[t.test_id] = "demo_item change (no labels) -> demo-sensitive"
            elif not t.inferred_demo_deps:
                hits[t.test_id] = f"demo_item change [{label_str}] -> structural demo test"
            elif labels & set(t.inferred_demo_deps):
                overlap = sorted(labels & set(t.inferred_demo_deps))
                hits[t.test_id] = f"demo_item change [{label_str}] -> demo dep {overlap}"
            else:
                continue
        elif labels and labels & set(t.inferred_demo_deps):
            overlap = sorted(labels & set(t.inferred_demo_deps))
            hits[t.test_id] = f"demo_item [{label_str}] -> demo dep overlap (non-demo test)"

    return hits


def _select_for_policy(
    change: ClassifiedChange,
    tagged_tests: list[TaggedTest],
) -> dict[str, str]:
    """Return {test_id: reason} for a policy change."""
    hits: dict[str, str] = {}
    for t in tagged_tests:
        if t.sensitivity_category == "policy":
            hits[t.test_id] = "policy change -> policy-sensitive"
        elif "safety_sensitive" in t.tags:
            hits[t.test_id] = "policy change -> safety_sensitive tag"
        elif "policy" in t.sensitive_to:
            hits[t.test_id] = "policy change -> sensitive_to includes 'policy'"
    return hits


def _select_for_workflow(
    change: ClassifiedChange,
    tagged_tests: list[TaggedTest],
) -> dict[str, str]:
    """Return {test_id: reason} for a workflow change."""
    hits: dict[str, str] = {}
    for t in tagged_tests:
        if t.sensitivity_category == "workflow":
            hits[t.test_id] = "workflow change -> workflow-sensitive"
        elif "workflow_sensitive" in t.tags:
            hits[t.test_id] = "workflow change -> workflow_sensitive tag"
        elif "workflow" in t.sensitive_to:
            hits[t.test_id] = "workflow change -> sensitive_to includes 'workflow'"
    return hits


def _select_all(
    reason: str,
    tagged_tests: list[TaggedTest],
) -> dict[str, str]:
    """Select every test (used for role, unknown, model_settings changes)."""
    return {t.test_id: reason for t in tagged_tests}


_AFFINITY_DISPATCH: dict[
    str, Callable[[ClassifiedChange, list[TaggedTest]], dict[str, str]]
] = {
    "format": _select_for_format,
    "demonstration": _select_for_demonstration,
    "demonstrations": _select_for_demonstration,
    "demo_item": _select_for_demo_item,
    "policy": _select_for_policy,
    "workflow": _select_for_workflow,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_impacted(
    changes: list[ClassifiedChange],
    tagged_tests: list[TaggedTest],
    *,
    magnitude_threshold: float = 0.005,
) -> PredictionResult:
    """Predict which tests are impacted by the given changes.

    Args:
        changes: Classified changes from the change classifier.
        tagged_tests: Enriched test cases from the test tagger.
        magnitude_threshold: Skip changes with magnitude below this value.

    Returns:
        PredictionResult with predicted test IDs and per-test reasons.
    """
    result = PredictionResult()

    for change in changes:
        if change.change.magnitude < magnitude_threshold:
            continue

        unit_type = change.change.unit_type
        handler = _AFFINITY_DISPATCH.get(unit_type)

        if handler is not None:
            hits = handler(change, tagged_tests)
        elif unit_type in ("role", "unknown", "model_settings"):
            hits = _select_all(
                f"{unit_type} change -> select all tests",
                tagged_tests,
            )
        else:
            hits = _select_all(
                f"unrecognised change type '{unit_type}' -> select all tests",
                tagged_tests,
            )

        for test_id, reason in hits.items():
            result.predicted_ids.add(test_id)
            result.reasons.setdefault(test_id, []).append(reason)

    return result
