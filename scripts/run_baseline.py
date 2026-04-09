#!/usr/bin/env python3
"""CLI: run the full-rerun baseline for a domain and write ground truth."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.phase1.baseline.full_rerun import load_versions, run_full_baseline, write_results
from src.phase1.harness.cache import EvalCache
from src.phase1.harness.runner import load_settings
from src.phase1.prompts.loader import load_prompt

app = typer.Typer(add_completion=False)


@app.command()
def run(
    domain: str = typer.Option("domain_a", help="Domain to run baseline for"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the result cache"),
) -> None:
    """Run full-rerun baseline: every test on every version, write ground truth."""
    settings = load_settings()

    typer.echo(f"Loading base prompt for {domain}...")
    base_prompt = load_prompt(domain)
    typer.echo(f"Base prompt loaded ({len(base_prompt)} chars)")

    versions = load_versions(domain)
    typer.echo(f"Found {len(versions)} versions")

    cache = None if no_cache else EvalCache(settings.get("cache_dir", "./cache/eval_cache"))

    typer.echo(f"Running baseline with model={settings['model']}, "
               f"concurrency={settings.get('max_concurrent_requests', 50)}, "
               f"retries={settings.get('num_retries', 10)}")

    results = asyncio.run(
        run_full_baseline(
            domain,
            base_prompt=base_prompt,
            settings=settings,
            cache=cache,
        )
    )

    if cache is not None:
        cache.close()

    detail_path, summary_path = write_results(results)

    summary = results["summary"]
    typer.echo("\n" + "=" * 70)
    typer.echo(f"Domain: {domain}")
    typer.echo(f"Versions: {summary['total_versions']}")
    typer.echo(f"Tests per version: {summary['total_tests']}")
    typer.echo(f"Total LLM calls: {summary['total_llm_calls']}")
    typer.echo(f"Total cost: ${summary['total_cost_usd']:.4f}")
    typer.echo(f"Wall clock: {summary['wall_clock_seconds']}s")
    typer.echo(f"Base pass rate: {summary['base_pass_rate']*100:.0f}%")

    impact = summary["impact_summary"]
    impacted_counts = [v["total_impacted"] for v in impact.values()]
    versions_with_impact = sum(1 for c in impacted_counts if c > 0)
    versions_with_5plus = sum(1 for c in impacted_counts if c >= 5)

    typer.echo(f"\nVersions with any impacted tests: {versions_with_impact}/{len(impact)}")
    typer.echo(f"Versions with >=5 impacted tests: {versions_with_5plus}/{len(impact)} "
               f"({versions_with_5plus/len(impact)*100:.0f}%)")
    typer.echo(f"Avg impacted tests: {sum(impacted_counts)/len(impacted_counts):.1f}")
    typer.echo(f"Max impacted: {max(impacted_counts)}")
    typer.echo(f"Min impacted: {min(impacted_counts)}")

    typer.echo(f"\nGround truth: {detail_path}")
    typer.echo(f"Summary: {summary_path}")


if __name__ == "__main__":
    app()
