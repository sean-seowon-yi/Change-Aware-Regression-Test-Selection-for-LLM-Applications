"""Change classifier: diff two prompts into typed ClassifiedChange objects.

Given an old and new prompt text, this module:
1. Parses both into labeled sections (role, format, demonstrations, policy, workflow)
2. Diffs at the section level (inserted, deleted, modified, unchanged)
3. For demonstration sections, diffs at the individual example level
4. For format sections, detects affected JSON key names
5. Computes magnitude via normalised token-level edit distance
6. Returns ClassifiedChange objects wrapping the core Change dataclass

All inputs are raw prompt strings — no oracle metadata is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.phase1.models import Change
from src.phase1.prompts.parser import parse_prompt, diff_sections
from src.phase1.prompts.mutator import (
    _token_edit_distance,
    _split_demos,
    _STEP_RE,
)


@dataclass
class ClassifiedChange:
    """A Change enriched with classifier metadata for downstream matching."""

    change: Change
    affected_keys: list[str] = field(default_factory=list)
    affected_demo_labels: list[str] = field(default_factory=list)


_JSON_KEY_RE = re.compile(r'"(\w+)"\s*:')


def _extract_json_keys(text: str) -> set[str]:
    """Extract JSON-style key names from a text block."""
    return set(_JSON_KEY_RE.findall(text))


def _extract_demo_label(example_text: str) -> str:
    """Extract the parenthesised label from an example header."""
    m = re.search(r"Example\s+\d+\s*\(([^)]+)\)", example_text)
    return m.group(1).strip() if m else ""


def _diff_demos(
    old_demos: str, new_demos: str
) -> list[ClassifiedChange]:
    """Item-level diffing of the demonstrations section."""
    old_examples = _split_demos(old_demos)
    new_examples = _split_demos(new_demos)

    def _build_label_map(
        examples: list[str],
    ) -> dict[str, tuple[int, str]]:
        """Map labels to (index, text), using positional fallback for
        empty or duplicate labels."""
        by_label: dict[str, tuple[int, str]] = {}
        seen: set[str] = set()
        for i, ex in enumerate(examples):
            label = _extract_demo_label(ex)
            if not label or label in seen:
                label = f"__pos_{i}"
            seen.add(label)
            by_label[label] = (i, ex)
        return by_label

    old_by_label = _build_label_map(old_examples)
    new_by_label = _build_label_map(new_examples)

    changes: list[ClassifiedChange] = []
    all_labels = list(dict.fromkeys(
        list(old_by_label.keys()) + list(new_by_label.keys())
    ))

    for label in all_labels:
        in_old = label in old_by_label
        in_new = label in new_by_label

        if in_old and in_new:
            old_idx, old_text = old_by_label[label]
            new_idx, new_text = new_by_label[label]
            # Strip example numbers before comparing content since
            # renumbering after insertion/deletion is cosmetic.
            old_content = re.sub(r"Example\s+\d+", "Example N", old_text)
            new_content = re.sub(r"Example\s+\d+", "Example N", new_text)
            if old_content != new_content:
                mag = _token_edit_distance(old_text, new_text)
                changes.append(ClassifiedChange(
                    change=Change(
                        unit_id=f"demo:ex{old_idx + 1}",
                        unit_type="demo_item",
                        change_type="modified",
                        magnitude=round(mag, 4),
                        content_diff=f"Modified example ({label}): {new_text[:100]}...",
                    ),
                    affected_demo_labels=[label],
                ))
        elif in_old:
            old_idx, old_text = old_by_label[label]
            changes.append(ClassifiedChange(
                change=Change(
                    unit_id=f"demo:ex{old_idx + 1}",
                    unit_type="demonstration",
                    change_type="deleted",
                    magnitude=round(
                        _token_edit_distance(old_text, ""), 4
                    ),
                    content_diff=f"Removed example ({label})",
                ),
                affected_demo_labels=[label],
            ))
        else:
            new_idx, new_text = new_by_label[label]
            changes.append(ClassifiedChange(
                change=Change(
                    unit_id=f"demo:ex{new_idx + 1}",
                    unit_type="demonstration",
                    change_type="inserted",
                    magnitude=round(
                        _token_edit_distance("", new_text), 4
                    ),
                    content_diff=f"Added example ({label})",
                ),
                affected_demo_labels=[label],
            ))

    # Detect reordering: if no content changes but positions shifted.
    # Only reliable when every example has a unique, non-empty label.
    if not changes and old_examples and new_examples:
        old_labels = [_extract_demo_label(ex) for ex in old_examples]
        new_labels = [_extract_demo_label(ex) for ex in new_examples]
        all_valid = all(old_labels) and all(new_labels)
        if (
            all_valid
            and set(old_labels) == set(new_labels)
            and old_labels != new_labels
        ):
            mag = _token_edit_distance(old_demos, new_demos)
            changes.append(ClassifiedChange(
                change=Change(
                    unit_id="demonstrations_section",
                    unit_type="demonstration",
                    change_type="reordered",
                    magnitude=round(mag, 4),
                    content_diff="Demonstration examples reordered",
                ),
                affected_demo_labels=new_labels,
            ))

    return changes


def _detect_workflow_reorder(old_text: str, new_text: str) -> bool:
    """Check if a workflow change is purely a step reorder."""
    old_steps = _STEP_RE.findall(old_text)
    new_steps = _STEP_RE.findall(new_text)
    if not old_steps or not new_steps:
        return False
    # Normalise step numbers for comparison
    def _strip_step_num(s: str) -> str:
        return re.sub(r"Step\s+\d+", "Step N", s).strip()

    old_normalised = {_strip_step_num(s) for s in old_steps}
    new_normalised = {_strip_step_num(s) for s in new_steps}
    if old_normalised != new_normalised:
        return False
    old_order = [_strip_step_num(s) for s in old_steps]
    new_order = [_strip_step_num(s) for s in new_steps]
    return old_order != new_order


def classify_changes(
    old_prompt: str,
    new_prompt: str,
) -> list[ClassifiedChange]:
    """Diff two prompts and return typed, classified changes.

    Only uses the two prompt texts as input — no oracle metadata.
    """
    if old_prompt == new_prompt:
        return []

    old_sections = parse_prompt(old_prompt)
    new_sections = parse_prompt(new_prompt)

    # Fallback: if parsing yields nothing (no XML tags), treat as whole-string diff
    if not old_sections and not new_sections:
        mag = _token_edit_distance(old_prompt, new_prompt)
        return [ClassifiedChange(
            change=Change(
                unit_id="prompt",
                unit_type="unknown",
                change_type="modified",
                magnitude=round(mag, 4),
                content_diff=f"Whole-prompt change: {new_prompt[:120]}...",
            ),
        )]

    section_diffs = diff_sections(old_sections, new_sections)
    results: list[ClassifiedChange] = []

    for section_name, change_type in section_diffs:
        if change_type == "unchanged":
            continue

        old_text = old_sections.get(section_name, "")
        new_text = new_sections.get(section_name, "")
        magnitude = _token_edit_distance(old_text, new_text)

        if section_name == "demonstrations" and change_type == "modified":
            demo_changes = _diff_demos(old_text, new_text)
            if demo_changes:
                results.extend(demo_changes)
            else:
                # Whole-section change without identifiable item-level diffs
                # (e.g., preamble edit, label format change)
                results.append(ClassifiedChange(
                    change=Change(
                        unit_id="demonstrations_section",
                        unit_type="demonstration",
                        change_type="modified",
                        magnitude=round(magnitude, 4),
                        content_diff=f"Demonstration section modified: {new_text[:120]}...",
                    ),
                ))
            continue

        # Workflow reorder detection
        if section_name == "workflow" and change_type == "modified":
            if _detect_workflow_reorder(old_text, new_text):
                change_type = "reordered"

        # Format key detection
        affected_keys: list[str] = []
        if section_name == "format" and change_type == "modified":
            old_keys = _extract_json_keys(old_text)
            new_keys = _extract_json_keys(new_text)
            added_keys = new_keys - old_keys
            removed_keys = old_keys - new_keys
            affected_keys = sorted(added_keys | removed_keys)

        content_diff = _build_content_diff(
            section_name, change_type, old_text, new_text
        )

        results.append(ClassifiedChange(
            change=Change(
                unit_id=f"{section_name}_section",
                unit_type=section_name,
                change_type=change_type,
                magnitude=round(magnitude, 4),
                content_diff=content_diff,
            ),
            affected_keys=affected_keys,
        ))

    return results


def _build_content_diff(
    section_name: str,
    change_type: str,
    old_text: str,
    new_text: str,
) -> str:
    """Build a human-readable content_diff summary."""
    if change_type == "inserted":
        return f"Added {section_name} section: {new_text[:120]}..."
    if change_type == "deleted":
        return f"Removed {section_name} section"
    if change_type == "reordered":
        return f"Reordered items in {section_name} section"
    # modified — show a brief diff summary
    preview = new_text[:120].replace("\n", " ")
    return f"Modified {section_name}: {preview}..."
