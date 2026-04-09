#!/usr/bin/env python3
"""CLI: run baseline via OpenAI Batch API for one or both domains."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.phase1.baseline.batch_runner import run_batch_baseline, write_results
from src.phase1.harness.runner import load_settings

app = typer.Typer(add_completion=False)


def _print_summary(summary: dict) -> None:
    typer.echo("\n" + "=" * 70)
    typer.echo(f"Domain: {summary['domain']}")
    typer.echo(f"Versions: {summary['total_versions']}")
    typer.echo(f"Tests per version: {summary['total_tests']}")
    typer.echo(f"Total LLM calls: {summary['total_llm_calls']}")
    typer.echo(f"Total cost: ${summary['total_cost_usd']:.4f} (batch: 50% discount)")
    typer.echo(f"Wall clock: {summary['wall_clock_seconds']}s")
    typer.echo(f"Base pass rate: {summary['base_pass_rate'] * 100:.0f}%")
    typer.echo(f"Batch ID: {summary.get('batch_id', 'N/A')}")

    impact = summary["impact_summary"]
    impacted_counts = [v["total_impacted"] for v in impact.values()]
    versions_with_impact = sum(1 for c in impacted_counts if c > 0)
    versions_with_5plus = sum(1 for c in impacted_counts if c >= 5)
    error_versions = sum(1 for v in impact.values() if v["errors"] > 0)

    typer.echo(f"\nVersions with any impacted tests: {versions_with_impact}/{len(impact)}")
    typer.echo(f"Versions with >=5 impacted tests: {versions_with_5plus}/{len(impact)} "
               f"({versions_with_5plus / len(impact) * 100:.0f}%)")
    typer.echo(f"Versions with errors: {error_versions}/{len(impact)}")
    typer.echo(f"Avg impacted tests: {sum(impacted_counts) / len(impacted_counts):.1f}")
    typer.echo(f"Max impacted: {max(impacted_counts)}")
    typer.echo(f"Min impacted: {min(impacted_counts)}")


@app.command()
def run(
    domain: str = typer.Option("domain_a", help="Domain to run baseline for"),
    both: bool = typer.Option(False, "--both", help="Run all domains (domain_a, domain_b, domain_c)"),
) -> None:
    """Run full-rerun baseline via OpenAI Batch API."""
    settings = load_settings()
    domains = ["domain_a", "domain_b", "domain_c"] if both else [domain]

    for d in domains:
        typer.echo(f"\n{'='*70}")
        typer.echo(f"Starting batch baseline for {d}")
        typer.echo(f"Model: {settings['model']}")

        results = run_batch_baseline(d, settings=settings)
        detail_path, summary_path = write_results(results)

        _print_summary(results["summary"])
        typer.echo(f"\nGround truth: {detail_path}")
        typer.echo(f"Summary: {summary_path}")


if __name__ == "__main__":
    app()
