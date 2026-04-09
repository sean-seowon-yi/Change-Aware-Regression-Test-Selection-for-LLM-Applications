#!/usr/bin/env python3
"""CLI entrypoint: load a domain prompt, run its eval suite, print results."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import typer

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.phase1.harness.cache import EvalCache
from src.phase1.harness.runner import load_eval_suite, load_settings, run_suite
from src.phase1.prompts.loader import load_prompt

app = typer.Typer(add_completion=False)


def _print_results(results, settings) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.outcome == "pass")
    failed = total - passed
    total_cost = sum(r.cost_usd for r in results)
    cached_count = sum(1 for r in results if r.cached)

    typer.echo("\n" + "=" * 80)
    typer.echo(f"{'TEST ID':<12} {'OUTCOME':<10} {'LATENCY':<12} {'CACHED':<8} {'DETAILS'}")
    typer.echo("-" * 80)
    for r in results:
        details_str = json.dumps(r.monitor_details, default=str)
        if len(details_str) > 40:
            details_str = details_str[:37] + "..."
        typer.echo(
            f"{r.test_id:<12} {r.outcome:<10} {r.latency_ms:>8.1f} ms  {'yes' if r.cached else 'no':<8} {details_str}"
        )
    typer.echo("=" * 80)
    typer.echo(f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Cached: {cached_count}  |  Cost: ${total_cost:.4f}")
    typer.echo(f"Model: {settings.get('model', 'N/A')}  |  Temperature: {settings.get('temperature', 'N/A')}")


@app.command()
def run(
    domain: str = typer.Option("domain_a", help="Domain to evaluate"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the result cache"),
    output_file: str = typer.Option(None, "--output", "-o", help="Write results as JSONL"),
) -> None:
    """Load a domain prompt, run its eval suite, and print results."""
    settings = load_settings()

    typer.echo(f"Loading prompt for {domain}...")
    prompt_text = load_prompt(domain)
    typer.echo(f"Prompt loaded ({len(prompt_text)} chars)")

    typer.echo(f"Loading eval suite for {domain}...")
    test_cases = load_eval_suite(domain)
    typer.echo(f"Loaded {len(test_cases)} test cases")

    cache = None if no_cache else EvalCache(settings.get("cache_dir", "./cache/eval_cache"))

    typer.echo(f"Running eval suite with model={settings['model']}...")
    results = asyncio.run(
        run_suite(
            prompt_text,
            test_cases,
            version_id="base",
            model=settings["model"],
            temperature=settings["temperature"],
            max_tokens=settings.get("max_tokens", 1024),
            max_concurrent=settings.get("max_concurrent_requests", 50),
            num_retries=settings.get("num_retries", 10),
            cache=cache,
        )
    )

    if cache is not None:
        cache.close()

    _print_results(results, settings)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(asdict(r), default=str) + "\n")
        typer.echo(f"Results written to {out_path}")


if __name__ == "__main__":
    app()
