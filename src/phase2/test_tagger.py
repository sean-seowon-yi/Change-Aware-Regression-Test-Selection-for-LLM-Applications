"""Test tagger: enrich test cases with inferred sensitivity information.

Derives sensitivity categories, key dependencies, and demo dependencies
from the test's monitor_type, monitor_config, tags, and sensitive_to fields,
plus the observable prompt text (for demo label extraction).

All information used is available in a real deployment:
- Test suite metadata (eval_suite.yaml)
- The prompt text (for parsing demonstration example labels)
No ground truth or version metadata is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.phase1.models import TestCase
from src.phase1.prompts.mutator import _split_demos


@dataclass
class TaggedTest:
    """A test case enriched with inferred sensitivity information."""

    test_id: str
    input_text: str
    monitor_type: str
    tags: list[str]
    sensitive_to: list[str]
    inferred_key_deps: list[str] = field(default_factory=list)
    inferred_demo_deps: list[str] = field(default_factory=list)
    sensitivity_category: str = "general"


def _extract_demo_label_map(demos_text: str) -> dict[int, str]:
    """Parse demonstration text and return {1-based index -> label}.

    Derives labels dynamically from the prompt text itself, e.g.:
        'Example 1 (billing):' -> {1: 'billing'}
        'Example 2 (refund):' -> {2: 'refund'}
    """
    examples = _split_demos(demos_text)
    label_map: dict[int, str] = {}
    for i, ex in enumerate(examples, 1):
        m = re.search(r"Example\s+\d+\s*\(([^)]+)\)", ex)
        if m:
            label_map[i] = m.group(1).strip()
    return label_map


def _extract_key_deps(test_case: TestCase) -> list[str]:
    """Extract JSON key dependencies from monitor_config."""
    config = test_case.monitor_config
    keys: list[str] = []

    if test_case.monitor_type == "schema":
        schema = config.get("json_schema", {})
        required = schema.get("required", [])
        keys.extend(required)
        props = schema.get("properties", {})
        keys.extend(k for k in props if k not in keys)

    elif test_case.monitor_type == "required_keys":
        keys.extend(config.get("keys", []))

    elif test_case.monitor_type == "keyword_presence":
        must_contain = config.get("must_contain", [])
        must_not_contain = config.get("must_not_contain", [])
        all_keywords = must_contain + must_not_contain
        for kw in all_keywords:
            m = re.match(r'^"?(\w+)"?$', kw.strip())
            if m and _looks_like_json_key(m.group(1)):
                keys.append(m.group(1))

    return sorted(set(keys))


def _looks_like_json_key(word: str) -> bool:
    """Heuristic: does this word look like a JSON key name?

    Uses structural cues only (snake_case or short single lowercase word)
    rather than a hardcoded allowlist, to avoid encoding domain knowledge.
    """
    if not word or not word[0].isalpha():
        return False
    if word != word.lower():
        return False
    if len(word) > 30:
        return False
    if "_" in word:
        return True
    if word.isalpha() and len(word) <= 20:
        return True
    return False


def _extract_demo_deps(
    test_case: TestCase,
    demo_label_map: dict[int, str],
) -> list[str]:
    """Extract demo dependencies from sensitive_to and tags.

    Uses the demo_label_map (derived from prompt text) to resolve
    'demo:exN' references to human-readable labels like 'billing'.
    """
    deps: list[str] = []
    for s in test_case.sensitive_to:
        m = re.match(r"demo:ex(\d+)", s)
        if m:
            idx = int(m.group(1))
            label = demo_label_map.get(idx, f"example_{idx}")
            deps.append(label)
    return deps


def _determine_sensitivity_category(test_case: TestCase) -> str:
    """Determine the primary sensitivity category from test metadata.

    Priority order reflects the most specific signal available:
    1. Explicit sensitive_to field (authored by test writer)
    2. Tags (authored by test writer)
    3. Monitor type (structural)
    """
    sensitive_to = test_case.sensitive_to
    tags = test_case.tags

    if "safety_sensitive" in tags:
        return "policy"

    # Check explicit sensitive_to
    sensitive_sections = set()
    for s in sensitive_to:
        if s.startswith("demo:") or s == "demonstration":
            sensitive_sections.add("demo")
        elif s in ("format", "policy", "workflow", "role"):
            sensitive_sections.add(s)

    # If test is sensitive to a single section, use that
    if len(sensitive_sections) == 1:
        return sensitive_sections.pop()

    # Multiple sensitivities — use most specific signal from tags/monitor
    if "workflow_sensitive" in tags:
        return "workflow"
    if "schema_sensitive" in tags or "format_sensitive" in tags:
        return "format"
    if "key_correctness" in tags:
        return "format"
    if "domain_coverage" in tags:
        return "demo"
    if "explanation_quality" in tags:
        return "format"

    # Fall back to monitor type
    monitor_type = test_case.monitor_type
    if monitor_type in ("schema", "required_keys", "format"):
        return "format"
    if monitor_type == "policy":
        return "policy"
    if monitor_type == "code_execution":
        return "demo"

    return "general"


def tag_tests(
    test_cases: list[TestCase],
    demos_text: str = "",
) -> list[TaggedTest]:
    """Enrich test cases with inferred sensitivity information.

    Args:
        test_cases: Raw test cases from eval_suite.yaml.
        demos_text: The demonstrations section text from the prompt.
            Used to derive demo example labels dynamically. If empty,
            demo:exN references resolve to generic 'example_N' labels.

    Returns:
        List of TaggedTest with inferred key deps, demo deps, and
        sensitivity category.
    """
    demo_label_map = _extract_demo_label_map(demos_text) if demos_text else {}

    tagged: list[TaggedTest] = []
    for tc in test_cases:
        key_deps = _extract_key_deps(tc)
        demo_deps = _extract_demo_deps(tc, demo_label_map)
        category = _determine_sensitivity_category(tc)

        tagged.append(TaggedTest(
            test_id=tc.test_id,
            input_text=tc.input_text,
            monitor_type=tc.monitor_type,
            tags=list(tc.tags),
            sensitive_to=list(tc.sensitive_to),
            inferred_key_deps=key_deps,
            inferred_demo_deps=demo_deps,
            sensitivity_category=category,
        ))

    return tagged
