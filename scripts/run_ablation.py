"""Ablation sensitivity profiling.

For each domain, replaces each prompt section with a placeholder and runs the
full test suite.  If a test's outcome flips compared to the unmodified base
prompt, the test is marked as sensitive to that section.

Output: data/{domain}/sensitivity_profiles.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from tqdm import tqdm

from src.phase1.harness.cache import EvalCache
from src.phase1.harness.runner import load_eval_suite, load_settings, run_suite
from src.phase1.prompts.loader import SECTION_ORDER, assemble_prompt, load_prompt_sections
from src.phase1.prompts.parser import parse_prompt

app = typer.Typer()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data"
_PLACEHOLDER = "[This section has been removed.]"


async def _run_ablation_domain(domain: str, settings: dict, cache: EvalCache) -> dict:
    """Run ablation profiling for a single domain.

    Returns {test_id: {section: bool, ...}} where True means the test is
    sensitive to that section.
    """
    base_prompt = assemble_prompt(load_prompt_sections(domain))
    sections = parse_prompt(base_prompt)
    test_cases = load_eval_suite(domain)

    model = settings.get("model", "gpt-4o-mini")
    temperature = settings.get("temperature", 0)
    max_tokens = settings.get("max_tokens", 1024)
    max_concurrent = settings.get("max_concurrent_requests", 20)
    num_retries = settings.get("num_retries", 5)

    runner_kwargs = dict(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrent=max_concurrent,
        num_retries=num_retries,
        cache=cache,
    )

    typer.echo(f"\n[{domain}] Running baseline ({len(test_cases)} tests)...")
    baseline_results = await run_suite(
        base_prompt, test_cases, version_id="base", **runner_kwargs,
    )
    baseline_outcomes = {r.test_id: r.outcome for r in baseline_results}

    profiles: dict[str, dict[str, bool]] = {
        tc.test_id: {sec: False for sec in SECTION_ORDER}
        for tc in test_cases
    }

    for section in tqdm(SECTION_ORDER, desc=f"[{domain}] Ablating"):
        if section not in sections:
            continue

        ablated_sections = dict(sections)
        ablated_sections[section] = _PLACEHOLDER
        ablated_prompt = assemble_prompt(ablated_sections)

        ablated_results = await run_suite(
            ablated_prompt, test_cases,
            version_id=f"ablation_{section}", **runner_kwargs,
        )

        for result in ablated_results:
            if result.outcome != baseline_outcomes.get(result.test_id):
                profiles[result.test_id][section] = True

    return profiles


@app.command()
def main(
    domain: str = typer.Option("all", help="Domain to profile (domain_a, domain_b, domain_c, or all)"),
) -> None:
    """Run ablation sensitivity profiling and write results."""
    settings = load_settings()
    cache = EvalCache(settings.get("cache_dir", "./cache/eval_cache"))

    domains = ["domain_a", "domain_b", "domain_c"] if domain == "all" else [domain]

    for d in domains:
        profiles = asyncio.run(_run_ablation_domain(d, settings, cache))

        out_path = _DATA_DIR / d / "sensitivity_profiles.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(profiles, f, indent=2)
        typer.echo(f"[{d}] Wrote {out_path} ({len(profiles)} tests)")

        sensitive_counts = {sec: 0 for sec in SECTION_ORDER}
        for tid, secs in profiles.items():
            for sec, val in secs.items():
                if val:
                    sensitive_counts[sec] += 1
        typer.echo(f"[{d}] Sensitivity counts: {sensitive_counts}")

    cache.close()
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
