#!/usr/bin/env python3
"""CLI: run Phase 2 evaluation against Phase 1 ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.phase2.evaluator import (
    effective_means_from_aggregate,
    run_evaluation,
    run_sweep,
    write_outputs,
)

app = typer.Typer(help="Phase 2 evaluation: selector vs ground truth.")


def _print_summary(agg: dict) -> None:
    """Print a human-readable summary table to stdout."""
    a = agg["aggregate"]
    domain = agg["domain"]

    typer.echo(f"\n{'='*70}")
    typer.echo(f"  Evaluation Summary: {domain}")
    typer.echo(f"{'='*70}")
    typer.echo(f"  Versions evaluated:       {a['total_versions']}")
    typer.echo(f"  Tests per version:         {a['total_tests']}")
    typer.echo(f"  Perfect-recall versions:   {a['versions_with_perfect_recall']}")
    typer.echo(f"\n  {'Metric':<30} {'Value':>10}")
    typer.echo(f"  {'-'*42}")
    typer.echo(f"  {'Recall (predictor only)':<30} {a['mean_recall_predictor']:>9.4f}")
    typer.echo(f"  {'Recall (with sentinel)':<30} {a['mean_recall_with_sentinel']:>9.4f}")
    eff_r, eff_cr = effective_means_from_aggregate(a)
    typer.echo(f"  {'Effective recall':<30} {eff_r:>9.4f}")
    typer.echo(f"  {'Call reduction':<30} {a['mean_call_reduction']:>9.4f}")
    typer.echo(f"  {'Effective call reduction':<30} {eff_cr:>9.4f}")
    typer.echo(f"  {'False omission rate':<30} {a['mean_false_omission_rate']:>9.4f}")
    typer.echo(f"  {'Sentinel catch rate':<30} {a['sentinel_catch_rate']:>9.4f}")

    # Baselines
    baselines = agg.get("baselines", {})
    typer.echo(f"\n  {'Baseline Comparison':<30} {'Recall':>10} {'Reduction':>10}")
    typer.echo(f"  {'-'*52}")
    fr = baselines.get("full_rerun", {})
    typer.echo(
        f"  {'Full rerun':<30} {fr.get('mean_recall', 1.0):>10.4f}"
        f" {fr.get('call_reduction', 0.0):>10.4f}"
    )
    r50 = baselines.get("random_50", {})
    typer.echo(
        f"  {'Random 50%':<30} {r50.get('mean_recall', 0):>10.4f}"
        f" {r50.get('call_reduction', 0):>10.4f}"
    )
    mh = baselines.get("monitor_heuristic", {})
    typer.echo(
        f"  {'Monitor-type heuristic':<30} {mh.get('mean_recall', 0):>10.4f}"
        f" {mh.get('call_reduction', 0):>10.4f}"
    )

    # By change type
    typer.echo(f"\n  Breakdown by change type:")
    typer.echo(
        f"  {'Type':<18} {'Count':>6} {'RecallP':>8} {'RecallS':>8}"
        f" {'Reduc':>8} {'FOR':>8}"
    )
    typer.echo(f"  {'-'*58}")
    for ct, vals in sorted(agg.get("by_change_type", {}).items()):
        typer.echo(
            f"  {ct:<18} {vals['count']:>6}"
            f" {vals['mean_recall_predictor']:>8.4f}"
            f" {vals['mean_recall_with_sentinel']:>8.4f}"
            f" {vals['mean_call_reduction']:>8.4f}"
            f" {vals['mean_false_omission_rate']:>8.4f}"
        )

    # By magnitude
    typer.echo(f"\n  Breakdown by magnitude bucket:")
    typer.echo(
        f"  {'Bucket':<18} {'Count':>6} {'RecallP':>8} {'RecallS':>8}"
        f" {'Reduc':>8} {'FOR':>8}"
    )
    typer.echo(f"  {'-'*58}")
    for mb, vals in sorted(agg.get("by_magnitude_bucket", {}).items()):
        typer.echo(
            f"  {mb:<18} {vals['count']:>6}"
            f" {vals['mean_recall_predictor']:>8.4f}"
            f" {vals['mean_recall_with_sentinel']:>8.4f}"
            f" {vals['mean_call_reduction']:>8.4f}"
            f" {vals['mean_false_omission_rate']:>8.4f}"
        )

    # By category
    typer.echo(f"\n  Breakdown by mutation category:")
    typer.echo(
        f"  {'Category':<18} {'Count':>6} {'RecallP':>8} {'RecallS':>8}"
        f" {'Reduc':>8} {'FOR':>8}"
    )
    typer.echo(f"  {'-'*58}")
    for cat, vals in sorted(agg.get("by_category", {}).items()):
        typer.echo(
            f"  {cat:<18} {vals['count']:>6}"
            f" {vals['mean_recall_predictor']:>8.4f}"
            f" {vals['mean_recall_with_sentinel']:>8.4f}"
            f" {vals['mean_call_reduction']:>8.4f}"
            f" {vals['mean_false_omission_rate']:>8.4f}"
        )
    typer.echo()


def _print_sweep(sweep: list[dict]) -> None:
    typer.echo(
        f"\n  {'Sent%':>8} {'MagTh':>8} {'RecP':>8} {'RecS':>8} {'Reduc':>8}"
        f" {'EffRec':>8} {'EffCR':>8} {'FOR':>8}"
    )
    typer.echo(f"  {'-'*72}")
    for row in sweep:
        a = row["aggregate"]
        seff_r, seff_cr = effective_means_from_aggregate(a)
        typer.echo(
            f"  {row['sentinel_fraction']:>8.2f}"
            f" {row['magnitude_threshold']:>8.4f}"
            f" {a['mean_recall_predictor']:>8.4f}"
            f" {a['mean_recall_with_sentinel']:>8.4f}"
            f" {a['mean_call_reduction']:>8.4f}"
            f" {seff_r:>8.4f}"
            f" {seff_cr:>8.4f}"
            f" {a['mean_false_omission_rate']:>8.4f}"
        )
    typer.echo()


@app.command()
def evaluate(
    domain: str = typer.Option("", help="Domain (domain_a, domain_b, or domain_c). Empty with --all-domains runs all."),
    all_domains: bool = typer.Option(False, "--all-domains", help="Run all domains."),
    sweep: bool = typer.Option(False, "--sweep", help="Run parameter sweep."),
    sentinel_seed: int = typer.Option(42, help="Sentinel RNG seed."),
    sentinel_fraction: float = typer.Option(0.10, help="Sentinel fraction (default eval)."),
    magnitude_threshold: float = typer.Option(0.005, help="Magnitude threshold (default eval)."),
) -> None:
    """Evaluate the Phase 2 rule-based selector against Phase 1 ground truth."""

    domains: list[str] = []
    if all_domains:
        domains = ["domain_a", "domain_b", "domain_c"]
    elif domain:
        domains = [domain]
    else:
        typer.echo("Provide --domain NAME or --all-domains")
        raise typer.Exit(1)

    for dom in domains:
        typer.echo(f"\nRunning evaluation for {dom} ...")
        metrics, agg = run_evaluation(
            dom,
            sentinel_fraction=sentinel_fraction,
            magnitude_threshold=magnitude_threshold,
            sentinel_seed=sentinel_seed,
            progress=True,
        )
        _print_summary(agg)

        sweep_data: list[dict] | None = None
        if sweep:
            typer.echo(f"Running parameter sweep for {dom} ...")
            sweep_data = run_sweep(dom, sentinel_seed=sentinel_seed, progress=True)
            _print_sweep(sweep_data)

        out_dir = write_outputs(dom, metrics, agg, sweep_data)
        typer.echo(f"Results written to {out_dir}/")


if __name__ == "__main__":
    app()
