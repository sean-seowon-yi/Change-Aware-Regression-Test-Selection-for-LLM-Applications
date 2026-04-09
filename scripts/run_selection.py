#!/usr/bin/env python3
"""CLI: run the Phase 2 rule-based test selector on prompt versions."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
import yaml

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.phase1.models import TestCase
from src.phase1.prompts.loader import load_prompt_sections, assemble_prompt
from src.phase2.selector import select_tests

app = typer.Typer(help="Phase 2 rule-based test selector.")

DATA_DIR = Path(_project_root) / "data"


def _load_base_prompt(domain: str) -> str:
    return assemble_prompt(load_prompt_sections(domain))


def _load_version_prompt(domain: str, version_id: str) -> str:
    path = DATA_DIR / domain / "versions" / f"{version_id}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["prompt_text"]


def _load_test_cases(domain: str) -> list[TestCase]:
    path = DATA_DIR / domain / "eval_suite.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        TestCase(
            test_id=t["test_id"],
            input_text=t.get("input_text", ""),
            monitor_type=t["monitor_type"],
            monitor_config=t.get("monitor_config", {}),
            tags=t.get("tags", []),
            sensitive_to=t.get("sensitive_to", []),
        )
        for t in data["tests"]
    ]


@app.command()
def run(
    domain: str = typer.Option(..., help="Domain name (domain_a, domain_b, or domain_c)"),
    version: str = typer.Option(
        "", help="Single version ID (e.g. v01). Empty = run all."
    ),
    all_versions: bool = typer.Option(
        False, "--all", help="Run selector on every version."
    ),
    sentinel_seed: int = typer.Option(42, help="Sentinel RNG seed."),
    magnitude_threshold: float = typer.Option(
        0.005, help="Magnitude gating threshold."
    ),
    sentinel_fraction: float = typer.Option(
        0.10, help="Sentinel sampling fraction."
    ),
) -> None:
    """Run the rule-based selector on one or all versions for a domain."""
    base = _load_base_prompt(domain)
    cases = _load_test_cases(domain)

    if version:
        versions = [version]
    elif all_versions:
        vdir = DATA_DIR / domain / "versions"
        versions = sorted(p.stem for p in vdir.glob("v*.yaml"))
    else:
        typer.echo("Provide --version VID or --all")
        raise typer.Exit(1)

    results = []
    for vid in versions:
        v_prompt = _load_version_prompt(domain, vid)
        res = select_tests(
            base,
            v_prompt,
            cases,
            sentinel_fraction=sentinel_fraction,
            magnitude_threshold=magnitude_threshold,
            sentinel_seed=sentinel_seed,
        )
        results.append((vid, res))

    if len(results) == 1:
        vid, res = results[0]
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Domain: {domain}  Version: {vid}")
        typer.echo(f"{'='*60}")
        typer.echo(f"Changes detected: {len(res.changes)}")
        for c in res.changes:
            typer.echo(
                f"  {c.change.unit_type}:{c.change.change_type} "
                f"(mag={c.change.magnitude:.4f})"
            )
            if c.affected_keys:
                typer.echo(f"    affected_keys: {c.affected_keys}")
            if c.affected_demo_labels:
                typer.echo(f"    affected_demos: {c.affected_demo_labels}")
        typer.echo(f"\nTotal tests:     {res.total_tests}")
        typer.echo(f"Predicted:       {len(res.predicted_ids)}")
        typer.echo(f"Sentinel:        {len(res.sentinel_ids)}")
        typer.echo(f"Selected:        {len(res.selected_ids)}")
        typer.echo(f"Call reduction:  {res.call_reduction:.1%}")
        typer.echo(f"\nPredicted IDs:  {sorted(res.predicted_ids)}")
        typer.echo(f"Sentinel IDs:   {sorted(res.sentinel_ids)}")
    else:
        typer.echo(f"\n{'='*60}")
        typer.echo(
            f"Domain: {domain}  |  {len(results)} versions  |  "
            f"{len(cases)} tests"
        )
        typer.echo(f"{'='*60}")
        typer.echo(
            f"{'Version':<10} {'Changes':>8} {'Predicted':>10} "
            f"{'Sentinel':>10} {'Selected':>10} {'Reduction':>10}"
        )
        typer.echo("-" * 60)

        total_reduction = 0.0
        for vid, res in results:
            typer.echo(
                f"{vid:<10} {len(res.changes):>8} "
                f"{len(res.predicted_ids):>10} "
                f"{len(res.sentinel_ids):>10} "
                f"{len(res.selected_ids):>10} "
                f"{res.call_reduction:>9.1%}"
            )
            total_reduction += res.call_reduction

        avg = total_reduction / len(results) if results else 0
        typer.echo("-" * 60)
        typer.echo(f"{'AVERAGE':<10} {'':>8} {'':>10} {'':>10} {'':>10} {avg:>9.1%}")


if __name__ == "__main__":
    app()
