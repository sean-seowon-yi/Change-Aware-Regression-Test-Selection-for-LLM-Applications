#!/usr/bin/env python3
"""CLI: generate ~70 prompt versions for a domain and save to data/{domain}/versions/.

Produces five categories of mutations:
1. Mechanical (existing 50) -- deterministic string replacements
2. LLM-generated natural rewrites (5) -- realistic paragraph-level edits
3. Sequential chains (5) -- iterative edits building on previous versions
4. Intentional inconsistencies (3) -- update one section, forget another
5. Semantic-only changes (3) -- change meaning, preserve structure
6. New mutation types (4) -- meta-instructions, CoT, format shifts
"""

from __future__ import annotations

import asyncio
import copy
import sys
from dataclasses import asdict
from pathlib import Path

import typer
import yaml

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.phase1.prompts.loader import assemble_prompt, load_prompt_sections
from src.phase1.prompts.mutator import (
    MutationResult,
    _mutate_policy_replace,
    _mutate_remove_demo,
    _mutate_edit_demo_output,
    _mutate_workflow_reorder,
    _mutate_policy_add_clause,
    _STRENGTHEN_REPLACEMENTS,
    _NEW_POLICY_CLAUSES,
    _DOMAIN_B_STRENGTHEN,
    _DOMAIN_B_NEW_POLICY,
    _DOMAIN_C_STRENGTHEN,
    _DOMAIN_C_NEW_POLICY,
    generate_domain_a_versions,
    generate_domain_b_versions,
    generate_domain_c_versions,
    generate_inconsistency_versions_a,
    generate_inconsistency_versions_b,
    generate_inconsistency_versions_c,
    generate_new_type_versions_a,
    generate_new_type_versions_b,
    generate_new_type_versions_c,
)
from src.phase1.prompts.llm_mutator import (
    generate_llm_rewrites_domain_a,
    generate_llm_rewrites_domain_b,
    generate_llm_rewrites_domain_c,
    generate_llm_semantic_domain_a,
    generate_llm_semantic_domain_b,
    generate_llm_semantic_domain_c,
)
from src.phase1.models import Change

app = typer.Typer(add_completion=False)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _build_sequential_chains_a(
    base_sections: dict[str, str],
    existing_versions: list[tuple[str, dict[str, str], list[Change]]],
) -> list[tuple[str, str, dict[str, str], list[Change]]]:
    """Build sequential chains for domain_a.

    Returns (version_id_placeholder, parent_id, mutated_sections, changes).
    """
    chains: list[tuple[str, str, dict[str, str], list[Change]]] = []

    # Chain 1: v01 (rename action->task_type) -> also strengthen policy
    v01_sections = existing_versions[0][1]
    s, c = _mutate_policy_replace(
        v01_sections, _STRENGTHEN_REPLACEMENTS[0][0], _STRENGTHEN_REPLACEMENTS[0][1], "strengthen",
    )
    chains.append(("", "v01", s, existing_versions[0][2] + c))

    # Chain 2: v16 (remove demo) -> also reorder workflow
    v16_sections = existing_versions[15][1]
    s, c = _mutate_workflow_reorder(v16_sections, 0, 2)
    chains.append(("", "v16", s, existing_versions[15][2] + c))

    # Chain 3: v37 (reorder workflow) -> also edit demo output
    v37_sections = existing_versions[36][1]
    edits_a = [(0, '"priority": "high"', '"priority": "critical"')]
    s, c = _mutate_edit_demo_output(v37_sections, edits_a[0][0], edits_a[0][1], edits_a[0][2], "domain_a")
    chains.append(("", "v37", s, existing_versions[36][2] + c))

    # Chain 4: chain3 result -> also add policy clause
    chain3_sections = chains[-1][2]
    chain3_changes = chains[-1][3]
    s, c = _mutate_policy_add_clause(chain3_sections, _NEW_POLICY_CLAUSES[0])
    chains.append(("", "", s, chain3_changes + c))  # parent set later

    # Chain 5: v30 (strengthen policy) -> also remove demo
    v30_sections = existing_versions[29][1]
    s, c = _mutate_remove_demo(v30_sections, 3, "domain_a")
    chains.append(("", "v30", s, existing_versions[29][2] + c))

    return chains


def _build_sequential_chains_b(
    base_sections: dict[str, str],
    existing_versions: list[tuple[str, dict[str, str], list[Change]]],
) -> list[tuple[str, str, dict[str, str], list[Change]]]:
    """Build sequential chains for domain_b."""
    chains: list[tuple[str, str, dict[str, str], list[Change]]] = []

    # Chain 1: v01 (rename Explanation->Analysis) -> also strengthen policy
    v01_sections = existing_versions[0][1]
    s, c = _mutate_policy_replace(
        v01_sections, _DOMAIN_B_STRENGTHEN[0][0], _DOMAIN_B_STRENGTHEN[0][1], "strengthen",
    )
    chains.append(("", "v01", s, existing_versions[0][2] + c))

    # Chain 2: v16 (remove demo) -> also reorder workflow
    v16_sections = existing_versions[15][1]
    s, c = _mutate_workflow_reorder(v16_sections, 0, 1)
    chains.append(("", "v16", s, existing_versions[15][2] + c))

    # Chain 3: v37 (reorder workflow) -> also edit demo output
    v37_sections = existing_versions[36][1]
    edits_b = [(0, 'return "".join(ch for ch in text if ch not in vowels)',
                'return "".join(c for c in text if c.lower() not in "aeiou")')]
    s, c = _mutate_edit_demo_output(v37_sections, edits_b[0][0], edits_b[0][1], edits_b[0][2], "domain_b")
    chains.append(("", "v37", s, existing_versions[36][2] + c))

    # Chain 4: chain3 result -> also add policy clause
    chain3_sections = chains[-1][2]
    chain3_changes = chains[-1][3]
    s, c = _mutate_policy_add_clause(chain3_sections, _DOMAIN_B_NEW_POLICY[0])
    chains.append(("", "", s, chain3_changes + c))

    # Chain 5: v30 (strengthen policy) -> also remove demo
    v30_sections = existing_versions[29][1]
    s, c = _mutate_remove_demo(v30_sections, 2, "domain_b")
    chains.append(("", "v30", s, existing_versions[29][2] + c))

    return chains


def _build_sequential_chains_c(
    base_sections: dict[str, str],
    existing_versions: list[tuple[str, dict[str, str], list[Change]]],
) -> list[tuple[str, str, dict[str, str], list[Change]]]:
    """Build sequential chains for domain_c."""
    chains: list[tuple[str, str, dict[str, str], list[Change]]] = []

    # Chain 1: v01 (rename urgency->priority_level) -> also strengthen policy
    v01_sections = existing_versions[0][1]
    s, c = _mutate_policy_replace(
        v01_sections, _DOMAIN_C_STRENGTHEN[0][0], _DOMAIN_C_STRENGTHEN[0][1], "strengthen",
    )
    chains.append(("", "v01", s, existing_versions[0][2] + c))

    # Chain 2: v16 (remove demo) -> also reorder workflow
    v16_sections = existing_versions[15][1]
    s, c = _mutate_workflow_reorder(v16_sections, 0, 2)
    chains.append(("", "v16", s, existing_versions[15][2] + c))

    # Chain 3: v37 (reorder workflow) -> also edit demo output
    v37_sections = existing_versions[36][1]
    edits_c = [(0, '"urgency": "emergency"', '"urgency": "urgent"')]
    s, c = _mutate_edit_demo_output(v37_sections, edits_c[0][0], edits_c[0][1], edits_c[0][2], "domain_c")
    chains.append(("", "v37", s, existing_versions[36][2] + c))

    # Chain 4: chain3 result -> also add policy clause
    chain3_sections = chains[-1][2]
    chain3_changes = chains[-1][3]
    s, c = _mutate_policy_add_clause(chain3_sections, _DOMAIN_C_NEW_POLICY[0])
    chains.append(("", "", s, chain3_changes + c))

    # Chain 5: v30 (strengthen policy) -> also remove demo
    v30_sections = existing_versions[29][1]
    s, c = _mutate_remove_demo(v30_sections, 3, "domain_c")
    chains.append(("", "v30", s, existing_versions[29][2] + c))

    return chains


async def _generate_all(domain: str) -> None:
    """Generate all versions for a domain: mechanical + LLM + chains + inconsistencies + new types."""
    sections = load_prompt_sections(domain)
    base_prompt = assemble_prompt(sections)

    # --- Phase 1: Mechanical mutations (existing 50) ---
    if domain == "domain_a":
        mechanical = generate_domain_a_versions(sections)
    elif domain == "domain_b":
        mechanical = generate_domain_b_versions(sections)
    else:
        mechanical = generate_domain_c_versions(sections)

    typer.echo(f"[{domain}] Generated {len(mechanical)} mechanical versions")

    # --- Phase 2: LLM-generated (rewrites + semantic) ---
    typer.echo(f"[{domain}] Generating LLM-based mutations...")
    if domain == "domain_a":
        llm_rewrites = await generate_llm_rewrites_domain_a(sections)
        llm_semantic = await generate_llm_semantic_domain_a(sections)
    elif domain == "domain_b":
        llm_rewrites = await generate_llm_rewrites_domain_b(sections)
        llm_semantic = await generate_llm_semantic_domain_b(sections)
    else:
        llm_rewrites = await generate_llm_rewrites_domain_c(sections)
        llm_semantic = await generate_llm_semantic_domain_c(sections)
    typer.echo(f"[{domain}]   {len(llm_rewrites)} rewrites + {len(llm_semantic)} semantic")

    # --- Phase 3: Inconsistencies ---
    if domain == "domain_a":
        inconsistencies = generate_inconsistency_versions_a(sections)
    elif domain == "domain_b":
        inconsistencies = generate_inconsistency_versions_b(sections)
    else:
        inconsistencies = generate_inconsistency_versions_c(sections)
    typer.echo(f"[{domain}]   {len(inconsistencies)} inconsistency versions")

    # --- Phase 4: New mutation types ---
    if domain == "domain_a":
        new_types = generate_new_type_versions_a(sections)
    elif domain == "domain_b":
        new_types = generate_new_type_versions_b(sections)
    else:
        new_types = generate_new_type_versions_c(sections)
    typer.echo(f"[{domain}]   {len(new_types)} new mutation type versions")

    # --- Phase 5: Sequential chains (need mechanical versions first) ---
    if domain == "domain_a":
        chains_raw = _build_sequential_chains_a(sections, mechanical)
    elif domain == "domain_b":
        chains_raw = _build_sequential_chains_b(sections, mechanical)
    else:
        chains_raw = _build_sequential_chains_c(sections, mechanical)
    typer.echo(f"[{domain}]   {len(chains_raw)} sequential chain versions")

    # --- Assign version IDs and write ---
    versions_dir = _DATA_DIR / domain / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    # Clear old versions
    for old_file in versions_dir.glob("v*.yaml"):
        old_file.unlink()

    base_path = _DATA_DIR / domain / "base_prompt.txt"
    base_path.write_text(base_prompt)
    typer.echo(f"[{domain}] Saved base prompt ({len(base_prompt)} chars)")

    vid = 0

    # Write mechanical (v01-v50)
    for version_id, mutated_sections, changes in mechanical:
        prompt_text = assemble_prompt(mutated_sections)
        _write_version(versions_dir, version_id, domain, "base", changes, prompt_text)
        vid = max(vid, int(version_id[1:]))

    # Write LLM rewrites (v51-v55)
    for _, mutated_sections, changes in llm_rewrites:
        vid += 1
        version_id = f"v{vid:02d}"
        prompt_text = assemble_prompt(mutated_sections)
        _write_version(versions_dir, version_id, domain, "base", changes, prompt_text)

    # Write LLM semantic (v56-v58)
    for _, mutated_sections, changes in llm_semantic:
        vid += 1
        version_id = f"v{vid:02d}"
        prompt_text = assemble_prompt(mutated_sections)
        _write_version(versions_dir, version_id, domain, "base", changes, prompt_text)

    # Write inconsistencies (v59-v61)
    for _, mutated_sections, changes in inconsistencies:
        vid += 1
        version_id = f"v{vid:02d}"
        prompt_text = assemble_prompt(mutated_sections)
        _write_version(versions_dir, version_id, domain, "base", changes, prompt_text)

    # Write new types (v62-v65)
    for _, mutated_sections, changes in new_types:
        vid += 1
        version_id = f"v{vid:02d}"
        prompt_text = assemble_prompt(mutated_sections)
        _write_version(versions_dir, version_id, domain, "base", changes, prompt_text)

    # Write sequential chains (v66-v70)
    chain_vid_start = vid + 1
    for i, (_, parent_id, mutated_sections, changes) in enumerate(chains_raw):
        vid += 1
        version_id = f"v{vid:02d}"
        # Chain 4's parent is chain 3
        if i == 3:
            parent_id = f"v{chain_vid_start + 2:02d}"
        prompt_text = assemble_prompt(mutated_sections)
        _write_version(versions_dir, version_id, domain, parent_id, changes, prompt_text)

    total = vid
    typer.echo(f"\n[{domain}] Total: {total} versions in {versions_dir}")

    # Print change type distribution
    change_type_counts: dict[str, int] = {}
    category_counts = {
        "mechanical": len(mechanical),
        "llm_rewrite": len(llm_rewrites),
        "llm_semantic": len(llm_semantic),
        "inconsistency": len(inconsistencies),
        "new_type": len(new_types),
        "sequential_chain": len(chains_raw),
    }
    typer.echo(f"\n[{domain}] Category breakdown:")
    for cat, count in category_counts.items():
        typer.echo(f"  {cat}: {count}")


def _write_version(
    versions_dir: Path,
    version_id: str,
    domain: str,
    parent_id: str,
    changes: list[Change],
    prompt_text: str,
) -> None:
    version_data = {
        "version_id": version_id,
        "domain": domain,
        "parent_id": parent_id,
        "changes": [asdict(c) for c in changes],
        "prompt_text": prompt_text,
    }
    out_path = versions_dir / f"{version_id}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(version_data, f, default_flow_style=False, allow_unicode=True, width=120)


@app.command()
def generate(
    domain: str = typer.Option("domain_a", help="Domain to generate versions for (domain_a, domain_b, domain_c)"),
    all_domains: bool = typer.Option(False, "--all", help="Generate for all three domains"),
) -> None:
    """Generate ~70 prompt versions with typed change metadata."""
    domains = ["domain_a", "domain_b", "domain_c"] if all_domains else [domain]
    for d in domains:
        asyncio.run(_generate_all(d))


if __name__ == "__main__":
    app()
