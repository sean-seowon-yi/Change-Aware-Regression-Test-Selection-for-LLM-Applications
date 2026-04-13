#!/usr/bin/env python3
"""Unified CLI: run each pipeline phase via the existing phase scripts.

Each subcommand spawns the same ``python -m scripts.<module> ...`` you would run
manually, so all options from the underlying CLIs still apply if you call them
directly.

Examples:
    python -m scripts.run_phases list
    python -m scripts.run_phases phase1 --both
    python -m scripts.run_phases phase2 --all-domains --sweep
    python -m scripts.run_phases phase3-train
    python -m scripts.run_phases phase3-eval
    python -m scripts.run_phases phase3-eval --kfold
    python -m scripts.run_phases phase4 full
    python -m scripts.run_phases phase4 collect
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

_ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(
    help="Run CARTS pipeline phases (delegates to scripts.run_batch_baseline, "
    "run_evaluation, run_phase3_train, run_phase3_eval, run_phase4_analysis).",
    no_args_is_help=True,
)

phase4_app = typer.Typer(help="Phase 4 analysis (individual steps or full pipeline).")
app.add_typer(phase4_app, name="phase4")


def _spawn(module_args: list[str]) -> None:
    cmd = [sys.executable, "-m", *module_args]
    typer.secho(f"→ {' '.join(cmd)}", fg=typer.colors.CYAN)
    code = subprocess.call(cmd, cwd=str(_ROOT))
    raise typer.Exit(code)


@app.command("list")
def cmd_list() -> None:
    """Print what each phase command runs."""
    rows = [
        ("phase1", "scripts.run_batch_baseline run", "Phase 1: Batch API baseline → ground truth JSONL"),
        ("phase2", "scripts.run_evaluation evaluate", "Phase 2: Rule-based selector eval → results/selection/"),
        ("phase3-train", "scripts.run_phase3_train train-all", "Phase 3: Train LightGBM / XGBoost / logreg → models/"),
        ("phase3-eval", "scripts.run_phase3_eval evaluate-all", "Phase 3: Evaluate learned selectors → results/phase3/"),
        ("phase3-eval --kfold", "scripts.run_phase3_eval kfold-evaluate-all", "Phase 3: K-fold CV eval"),
        ("phase3-eval --compare", "scripts.run_phase3_eval compare", "Phase 3: Rule vs learned comparison JSON"),
        ("phase4 full", "scripts.run_phase4_analysis full", "Phase 4: collect → stats → ablations → cost → figures → tables"),
        ("phase4 <step>", "scripts.run_phase4_analysis <step>", "Steps: collect, statistics, ablations, cost-analysis, figures, tables"),
    ]
    typer.echo("\nPhase commands → underlying invocation\n")
    for name, cmd, desc in rows:
        typer.echo(f"  {name:<28} {desc}")
        typer.echo(f"    {'python -m ' + cmd}")
    typer.echo("\nUse --help on each subcommand for flags.\n")


@app.command("phase1")
def phase1(
    both: bool = typer.Option(
        False,
        "--both",
        help="Run domain_a, domain_b, and domain_c (default: single --domain).",
    ),
    domain: str = typer.Option("domain_a", "--domain", help="Domain when not using --both."),
) -> None:
    """Phase 1: full-rerun baseline (OpenAI Batch API) → ground truth."""
    args = ["scripts.run_batch_baseline", "run", "--domain", domain]
    if both:
        args.append("--both")
    _spawn(args)


@app.command("phase2")
def phase2(
    all_domains: bool = typer.Option(
        False,
        "--all-domains",
        help="Evaluate domain_a, domain_b, and domain_c.",
    ),
    domain: str = typer.Option("", "--domain", help="Single domain when not using --all-domains."),
    sweep: bool = typer.Option(False, "--sweep", help="Also run sentinel × magnitude sweep."),
    sentinel_seed: int = typer.Option(42, "--sentinel-seed"),
    sentinel_fraction: float = typer.Option(0.10, "--sentinel-fraction"),
    magnitude_threshold: float = typer.Option(0.005, "--magnitude-threshold"),
) -> None:
    """Phase 2: rule-based selector evaluation → results/selection/."""
    args = [
        "scripts.run_evaluation",
        "evaluate",
        "--sentinel-seed",
        str(sentinel_seed),
        "--sentinel-fraction",
        str(sentinel_fraction),
        "--magnitude-threshold",
        str(magnitude_threshold),
    ]
    if all_domains:
        args.append("--all-domains")
    elif domain:
        args.extend(["--domain", domain])
    else:
        typer.echo("Error: pass --all-domains or --domain <name>", err=True)
        raise typer.Exit(1)
    if sweep:
        args.append("--sweep")
    _spawn(args)


@app.command("phase3-train")
def phase3_train(
    pca_components: int = typer.Option(24, "--pca-components"),
    n_trials: int = typer.Option(50, "--n-trials", help="Optuna trials per boosted model."),
) -> None:
    """Phase 3: train all configured models (temporal + cross-domain splits)."""
    _spawn(
        [
            "scripts.run_phase3_train",
            "train-all",
            "--pca-components",
            str(pca_components),
            "--n-trials",
            str(n_trials),
        ],
    )


@app.command("phase3-eval")
def phase3_eval(
    kfold: bool = typer.Option(False, "--kfold", help="Run kfold-evaluate-all instead of evaluate-all."),
    compare_only: bool = typer.Option(
        False,
        "--compare",
        help="Only rebuild comparison_summary.json from existing results.",
    ),
) -> None:
    """Phase 3: evaluate learned selectors (or k-fold, or compare-only)."""
    if compare_only and kfold:
        typer.echo("Error: use either --compare or --kfold, not both.", err=True)
        raise typer.Exit(1)
    if compare_only:
        sub = "compare"
    elif kfold:
        sub = "kfold-evaluate-all"
    else:
        sub = "evaluate-all"
    _spawn(["scripts.run_phase3_eval", sub])


@phase4_app.command("full")
def phase4_full(
    skip_ablations: bool = typer.Option(False, "--skip-ablations"),
    n_trials: int = typer.Option(30, "--n-trials"),
) -> None:
    """Run full Phase 4 pipeline (collect → statistics → ablations → cost → figures → tables)."""
    args = ["scripts.run_phase4_analysis", "full"]
    if skip_ablations:
        args.append("--skip-ablations")
    args.extend(["--n-trials", str(n_trials)])
    _spawn(args)


@phase4_app.command("collect")
def phase4_collect() -> None:
    _spawn(["scripts.run_phase4_analysis", "collect"])


@phase4_app.command("statistics")
def phase4_statistics() -> None:
    _spawn(["scripts.run_phase4_analysis", "statistics"])


@phase4_app.command("ablations")
def phase4_ablations(
    domain: str = typer.Option("all", "--domain", help="domain_a | domain_b | domain_c | all"),
    model_type: str = typer.Option("lightgbm", "--model-type"),
    n_trials: int = typer.Option(30, "--n-trials"),
) -> None:
    _spawn(
        [
            "scripts.run_phase4_analysis",
            "ablations",
            "--domain",
            domain,
            "--model-type",
            model_type,
            "--n-trials",
            str(n_trials),
        ],
    )


@phase4_app.command("cost-analysis")
def phase4_cost() -> None:
    _spawn(["scripts.run_phase4_analysis", "cost-analysis"])


@phase4_app.command("figures")
def phase4_figures() -> None:
    _spawn(["scripts.run_phase4_analysis", "figures"])


@phase4_app.command("tables")
def phase4_tables() -> None:
    _spawn(["scripts.run_phase4_analysis", "tables"])


if __name__ == "__main__":
    app()
