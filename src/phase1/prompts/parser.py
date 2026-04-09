from __future__ import annotations

import re

from src.phase1.prompts.loader import SECTION_ORDER


def parse_prompt(prompt_text: str) -> dict[str, str]:
    """Parse an assembled prompt (XML-style tags) back into labeled sections.

    Expects format produced by loader.assemble_prompt():
        <role>...content...</role>
        <format>...content...</format>
        ...
    """
    sections: dict[str, str] = {}
    for name in SECTION_ORDER:
        pattern = rf"<{name}>\s*(.*?)\s*</{name}>"
        match = re.search(pattern, prompt_text, re.DOTALL)
        if match:
            sections[name] = match.group(1).strip()
    return sections


def diff_sections(
    old: dict[str, str], new: dict[str, str]
) -> list[tuple[str, str]]:
    """Compare two parsed prompt section dicts and classify each section's change.

    Returns a list of (section_name, change_type) for every section that appears
    in either dict.  change_type is one of:
        "inserted"  — present in new but not old
        "deleted"   — present in old but not new
        "modified"  — present in both but content differs
        "unchanged" — present in both with identical content
    """
    all_keys = list(dict.fromkeys(list(old.keys()) + list(new.keys())))

    results: list[tuple[str, str]] = []
    for key in all_keys:
        in_old = key in old
        in_new = key in new
        if in_old and in_new:
            change_type = "unchanged" if old[key] == new[key] else "modified"
        elif in_new:
            change_type = "inserted"
        else:
            change_type = "deleted"
        results.append((key, change_type))
    return results
