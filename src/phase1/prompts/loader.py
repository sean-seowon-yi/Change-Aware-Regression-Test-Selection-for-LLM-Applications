from __future__ import annotations

from pathlib import Path

SECTION_ORDER = ["role", "format", "demonstrations", "policy", "workflow"]

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt_sections(domain: str) -> dict[str, str]:
    """Read each .txt section file from src/phase1/prompts/{domain}/."""
    domain_dir = _PROMPTS_DIR / domain
    if not domain_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {domain_dir}")

    sections: dict[str, str] = {}
    for name in SECTION_ORDER:
        path = domain_dir / f"{name}.txt"
        if path.exists():
            sections[name] = path.read_text().strip()
    return sections


def assemble_prompt(sections: dict[str, str]) -> str:
    """Join sections into a single system prompt using XML-style tags."""
    parts: list[str] = []
    for name in SECTION_ORDER:
        if name in sections:
            parts.append(f"<{name}>\n{sections[name]}\n</{name}>")
    return "\n\n".join(parts)


def load_prompt(domain: str) -> str:
    """Convenience: load sections and assemble into a full prompt string."""
    return assemble_prompt(load_prompt_sections(domain))
