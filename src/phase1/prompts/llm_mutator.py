"""LLM-powered prompt mutations for realistic, human-like edits.

Uses gpt-4o-mini to generate natural rewrites and semantic changes that
mimic how a real prompt engineer would iterate on prompts — messy,
multi-word edits rather than surgical string replacements.
"""

from __future__ import annotations

import copy

from dotenv import load_dotenv
from litellm import acompletion

from src.phase1.models import Change
from src.phase1.prompts.mutator import _token_edit_distance

load_dotenv()

MutationResult = tuple[dict[str, str], list[Change]]

_MODEL = "gpt-4o-mini"

_PREAMBLE_MARKERS = [
    "Section name:",
    "Current text:",
    "Semantic change:",
    "Instruction:",
    "Here is the",
    "Apply this semantic",
]


def _strip_preamble(text: str) -> str:
    """Remove any accidentally echoed prompt structure from LLM output."""
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if any(line.strip().startswith(m) for m in _PREAMBLE_MARKERS):
            start = i + 1
        elif line.strip() == "" and start == i:
            start = i + 1
        else:
            break
    return "\n".join(lines[start:]).strip()


async def _llm_rewrite(
    section_name: str,
    section_text: str,
    instruction: str,
    *,
    temperature: float = 0.7,
) -> str:
    """Ask the LLM to rewrite a prompt section following a natural instruction."""
    response = await acompletion(
        model=_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior prompt engineer revising a system prompt section. "
                    "Rewrite the section as instructed. Return ONLY the rewritten section "
                    "text — no explanations, no markdown fencing, no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Section name: {section_name}\n\n"
                    f"Current text:\n{section_text}\n\n"
                    f"Instruction: {instruction}"
                ),
            },
        ],
        temperature=temperature,
        max_tokens=2048,
        num_retries=5,
        timeout=60,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError(f"LLM returned null content for section '{section_name}'")
    return _strip_preamble(content.strip())


async def _llm_semantic_rewrite(
    section_name: str,
    section_text: str,
    semantic_instruction: str,
) -> str:
    """Rewrite a section to change meaning while preserving structure."""
    response = await acompletion(
        model=_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior prompt engineer. The user will give you a prompt "
                    "section and an instruction. Apply the semantic change and return "
                    "ONLY the rewritten section text. Do NOT include any preamble, "
                    "labels, explanations, or the instruction itself. Do NOT echo back "
                    "'Section name' or 'Current text'. Just output the final rewritten "
                    "section content directly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the {section_name} section of a system prompt:\n\n"
                    f"{section_text}\n\n"
                    f"Apply this semantic change while preserving the exact same "
                    f"formatting, bullet style, headings, and structure:\n"
                    f"{semantic_instruction}"
                ),
            },
        ],
        temperature=0.4,
        max_tokens=2048,
        num_retries=5,
        timeout=60,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError(f"LLM returned null content for semantic rewrite of '{section_name}'")
    return _strip_preamble(content.strip())


# ---------------------------------------------------------------------------
# Domain A: LLM-generated mutations
# ---------------------------------------------------------------------------

_DOMAIN_A_REWRITES: list[tuple[str, str, str]] = [
    (
        "role",
        "Rewrite this role description to emphasize speed and throughput over "
        "meticulous accuracy. The assistant should prioritize fast extraction even if "
        "it means occasionally approximating fields. Keep it about the same length.",
        "Rewrote role to emphasize speed over accuracy",
    ),
    (
        "role",
        "Rewrite this role description so the assistant is a quality-focused auditor "
        "that double-checks every field before responding. Make it sound cautious and "
        "detail-oriented. Keep about the same length.",
        "Rewrote role to emphasize quality auditing and caution",
    ),
    (
        "policy",
        "Rewrite this policy to be more lenient about edge cases. The assistant "
        "should try to extract information even from borderline or ambiguous messages "
        "rather than flagging them. Soften the refusal language. Keep the same "
        "heading structure (PII Handling / Safety and Refusal).",
        "Rewrote policy to be more permissive about edge cases",
    ),
    (
        "policy",
        "Rewrite this policy to be extremely strict. Add requirements about "
        "double-checking all fields, flagging anything remotely suspicious, and "
        "defaulting to escalation when uncertain. Keep the same heading structure.",
        "Rewrote policy to be extremely strict with aggressive escalation defaults",
    ),
    (
        "workflow",
        "Rewrite this workflow as a more detailed checklist with sub-steps. "
        "Keep the same four main phases (classify, assess priority, extract, validate) "
        "but add 1-2 sub-bullets under each. Use a different numbering style "
        "(e.g., 1a, 1b, 2a, 2b).",
        "Rewrote workflow as a detailed checklist with sub-steps",
    ),
]

_DOMAIN_A_SEMANTIC: list[tuple[str, str, str]] = [
    (
        "role",
        "Change the scope: instead of just extracting information, the assistant "
        "should also suggest a recommended next action for the support agent "
        "(e.g., 'call customer back', 'issue refund', 'escalate to tier 2'). "
        "Keep the same format and structure.",
        "Expanded role scope to include next-action recommendations",
    ),
    (
        "policy",
        "Change the PII handling rules so that customer IDs are ALSO treated as "
        "sensitive information and should be partially redacted (e.g., 'C-****42'). "
        "Keep the exact same bullet-point structure.",
        "Changed PII policy to also redact customer IDs",
    ),
    (
        "workflow",
        "Change Step 2 so that priority assessment uses a different rubric: "
        "'critical' is for legal threats or data breaches only, 'high' includes "
        "complaints and escalations, 'medium' for billing and refunds, 'low' for "
        "everything else. Keep the same Step format.",
        "Changed priority rubric in workflow (different criteria for each level)",
    ),
]


# ---------------------------------------------------------------------------
# Domain B: LLM-generated mutations
# ---------------------------------------------------------------------------

_DOMAIN_B_REWRITES: list[tuple[str, str, str]] = [
    (
        "role",
        "Rewrite this role to make the assistant a competitive programming coach "
        "that prioritizes optimal algorithmic solutions and Big-O efficiency over "
        "readability. Keep about the same length.",
        "Rewrote role as competitive programming coach prioritizing efficiency",
    ),
    (
        "role",
        "Rewrite this role to make the assistant a beginner-friendly Python tutor "
        "that uses analogies and avoids advanced features. Prefer simple loops over "
        "comprehensions, avoid type hints, explain like talking to a 12-year-old. "
        "Keep about the same length.",
        "Rewrote role as beginner-friendly tutor avoiding advanced features",
    ),
    (
        "policy",
        "Rewrite this policy to also forbid code that accesses the filesystem, "
        "makes network requests, or uses subprocess/os.system. Add these as explicit "
        "bullet points. Keep the same heading structure.",
        "Rewrote policy to also forbid filesystem/network/subprocess access",
    ),
    (
        "policy",
        "Rewrite this policy to be much more relaxed. The assistant should attempt "
        "to help with any coding request, only refusing truly dangerous code like "
        "viruses or ransomware. Remove the ambiguity-handling rules. Keep headings.",
        "Rewrote policy to be very permissive, only refusing truly dangerous code",
    ),
    (
        "workflow",
        "Rewrite this workflow to add a mandatory planning phase between understanding "
        "and coding. The assistant should first outline its approach in pseudocode "
        "before writing real code. Keep the Step format.",
        "Rewrote workflow to include mandatory pseudocode planning phase",
    ),
]

_DOMAIN_B_SEMANTIC: list[tuple[str, str, str]] = [
    (
        "role",
        "Change the philosophy from 'prioritize correctness over cleverness' to "
        "'prioritize concise, elegant one-liner solutions when possible'. Keep "
        "everything else the same — same structure, same libraries listed.",
        "Changed coding philosophy from correctness-first to elegance-first",
    ),
    (
        "policy",
        "In the 'Scope' section, change 'Only write Python code' to allow both "
        "Python and JavaScript. Add 'JavaScript (Node.js)' as an accepted language. "
        "Keep the exact same bullet structure.",
        "Expanded language scope to include JavaScript alongside Python",
    ),
    (
        "workflow",
        "In Step 3, change the explanation requirement: instead of summarizing the "
        "approach in 2-4 sentences, require a detailed line-by-line walkthrough of "
        "the code. Keep the same Step format.",
        "Changed explanation style from high-level summary to line-by-line walkthrough",
    ),
]


# ---------------------------------------------------------------------------
# Domain C: LLM-generated mutations
# ---------------------------------------------------------------------------

_DOMAIN_C_REWRITES: list[tuple[str, str, str]] = [
    (
        "role",
        "Rewrite this role to emphasize empathy and patient comfort. The assistant "
        "should use warm, compassionate language while still maintaining clinical "
        "accuracy. It should acknowledge patient fear and anxiety. Keep about the same length.",
        "Rewrote role to emphasize empathy and patient comfort",
    ),
    (
        "role",
        "Rewrite this role to make it strictly clinical and impersonal — like a "
        "hospital intake form processor. Remove any language about being careful or "
        "safety-conscious. Just categorize and route. Keep about the same length.",
        "Rewrote role to be strictly clinical and impersonal",
    ),
    (
        "policy",
        "Rewrite this policy to be more lenient. Allow the assistant to mention general "
        "categories of treatment (e.g., 'physical therapy may help') without specifying "
        "medications or dosages. Soften the refusal language. Keep the same heading structure.",
        "Rewrote policy to be more lenient, allowing general treatment category mentions",
    ),
    (
        "policy",
        "Rewrite this policy to be extremely strict. Add requirements about always "
        "recommending emergency evaluation for any symptom that could possibly be serious, "
        "even if unlikely. Default to higher urgency when uncertain. Keep the same headings.",
        "Rewrote policy to be extremely strict with aggressive emergency defaults",
    ),
    (
        "workflow",
        "Rewrite this workflow as a more detailed clinical protocol with sub-steps. "
        "Keep the same five main phases but add 1-2 clinical considerations under each. "
        "Use a numbered sub-step format (e.g., 1a, 1b, 2a, 2b).",
        "Rewrote workflow as a detailed clinical protocol with sub-steps",
    ),
]

_DOMAIN_C_SEMANTIC: list[tuple[str, str, str]] = [
    (
        "role",
        "Change the scope: instead of routing to departments, the assistant should "
        "now recommend a specific type of diagnostic test (e.g., 'blood panel', 'MRI', "
        "'ECG', 'X-ray'). Replace the department routing concept with diagnostic "
        "recommendations. Keep the same format and structure.",
        "Expanded role scope to recommend diagnostic tests instead of departments",
    ),
    (
        "policy",
        "Change the self-harm escalation policy: instead of routing to psychiatry and "
        "including the 988 number, have the assistant generate a detailed safety plan "
        "with five steps. Keep the exact same bullet-point structure.",
        "Changed crisis policy from hotline referral to safety plan generation",
    ),
    (
        "workflow",
        "Change Step 2 so that urgency assessment uses a different rubric: "
        "'emergency' only for loss of consciousness or severe bleeding, 'urgent' for "
        "any pain or fever above 101°F, 'semi_urgent' for chronic conditions, 'non_urgent' "
        "for everything else. Keep the same Step format.",
        "Changed urgency rubric in workflow (different thresholds for each level)",
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_llm_rewrites_domain_a(
    sections: dict[str, str],
) -> list[tuple[str, dict[str, str], list[Change]]]:
    """Generate LLM-rewritten versions for domain_a (natural rewrites)."""
    results: list[tuple[str, dict[str, str], list[Change]]] = []
    for section_name, instruction, diff_desc in _DOMAIN_A_REWRITES:
        s = copy.deepcopy(sections)
        old_text = s[section_name]
        new_text = await _llm_rewrite(section_name, old_text, instruction)
        s[section_name] = new_text
        mag = _token_edit_distance(old_text, new_text)
        results.append((
            "",
            s,
            [Change(
                unit_id=f"{section_name}_section",
                unit_type=section_name,
                change_type="modified",
                magnitude=round(mag, 4),
                content_diff=f"LLM rewrite: {diff_desc}",
            )],
        ))
    return results


async def generate_llm_semantic_domain_a(
    sections: dict[str, str],
) -> list[tuple[str, dict[str, str], list[Change]]]:
    """Generate semantic-only changes for domain_a."""
    results: list[tuple[str, dict[str, str], list[Change]]] = []
    for section_name, instruction, diff_desc in _DOMAIN_A_SEMANTIC:
        s = copy.deepcopy(sections)
        old_text = s[section_name]
        new_text = await _llm_semantic_rewrite(section_name, old_text, instruction)
        s[section_name] = new_text
        mag = _token_edit_distance(old_text, new_text)
        results.append((
            "",
            s,
            [Change(
                unit_id=f"{section_name}_section",
                unit_type=section_name,
                change_type="modified",
                magnitude=round(mag, 4),
                content_diff=f"Semantic change: {diff_desc}",
            )],
        ))
    return results


async def generate_llm_rewrites_domain_b(
    sections: dict[str, str],
) -> list[tuple[str, dict[str, str], list[Change]]]:
    """Generate LLM-rewritten versions for domain_b (natural rewrites)."""
    results: list[tuple[str, dict[str, str], list[Change]]] = []
    for section_name, instruction, diff_desc in _DOMAIN_B_REWRITES:
        s = copy.deepcopy(sections)
        old_text = s[section_name]
        new_text = await _llm_rewrite(section_name, old_text, instruction)
        s[section_name] = new_text
        mag = _token_edit_distance(old_text, new_text)
        results.append((
            "",
            s,
            [Change(
                unit_id=f"{section_name}_section",
                unit_type=section_name,
                change_type="modified",
                magnitude=round(mag, 4),
                content_diff=f"LLM rewrite: {diff_desc}",
            )],
        ))
    return results


async def generate_llm_semantic_domain_b(
    sections: dict[str, str],
) -> list[tuple[str, dict[str, str], list[Change]]]:
    """Generate semantic-only changes for domain_b."""
    results: list[tuple[str, dict[str, str], list[Change]]] = []
    for section_name, instruction, diff_desc in _DOMAIN_B_SEMANTIC:
        s = copy.deepcopy(sections)
        old_text = s[section_name]
        new_text = await _llm_semantic_rewrite(section_name, old_text, instruction)
        s[section_name] = new_text
        mag = _token_edit_distance(old_text, new_text)
        results.append((
            "",
            s,
            [Change(
                unit_id=f"{section_name}_section",
                unit_type=section_name,
                change_type="modified",
                magnitude=round(mag, 4),
                content_diff=f"Semantic change: {diff_desc}",
            )],
        ))
    return results


async def generate_llm_rewrites_domain_c(
    sections: dict[str, str],
) -> list[tuple[str, dict[str, str], list[Change]]]:
    """Generate LLM-rewritten versions for domain_c (natural rewrites)."""
    results: list[tuple[str, dict[str, str], list[Change]]] = []
    for section_name, instruction, diff_desc in _DOMAIN_C_REWRITES:
        s = copy.deepcopy(sections)
        old_text = s[section_name]
        new_text = await _llm_rewrite(section_name, old_text, instruction)
        s[section_name] = new_text
        mag = _token_edit_distance(old_text, new_text)
        results.append((
            "",
            s,
            [Change(
                unit_id=f"{section_name}_section",
                unit_type=section_name,
                change_type="modified",
                magnitude=round(mag, 4),
                content_diff=f"LLM rewrite: {diff_desc}",
            )],
        ))
    return results


async def generate_llm_semantic_domain_c(
    sections: dict[str, str],
) -> list[tuple[str, dict[str, str], list[Change]]]:
    """Generate semantic-only changes for domain_c."""
    results: list[tuple[str, dict[str, str], list[Change]]] = []
    for section_name, instruction, diff_desc in _DOMAIN_C_SEMANTIC:
        s = copy.deepcopy(sections)
        old_text = s[section_name]
        new_text = await _llm_semantic_rewrite(section_name, old_text, instruction)
        s[section_name] = new_text
        mag = _token_edit_distance(old_text, new_text)
        results.append((
            "",
            s,
            [Change(
                unit_id=f"{section_name}_section",
                unit_type=section_name,
                change_type="modified",
                magnitude=round(mag, 4),
                content_diff=f"Semantic change: {diff_desc}",
            )],
        ))
    return results
