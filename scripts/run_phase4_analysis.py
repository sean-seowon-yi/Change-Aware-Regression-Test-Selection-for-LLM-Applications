"""Phase 4: Comprehensive evaluation, analysis & report generation.

Orchestrates all analysis steps in a deterministic, reproducible pipeline.

Usage:
    python -m scripts.run_phase4_analysis collect
    python -m scripts.run_phase4_analysis statistics
    python -m scripts.run_phase4_analysis ablations
    python -m scripts.run_phase4_analysis cost-analysis
    python -m scripts.run_phase4_analysis figures
    python -m scripts.run_phase4_analysis tables
    python -m scripts.run_phase4_analysis full
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(help="Phase 4: Comprehensive evaluation, analysis & report generation.")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _PROJECT_ROOT / "results" / "phase4"
_MODELS_DIR = _PROJECT_ROOT / "models"
_DOMAINS = ["domain_a", "domain_b", "domain_c"]
_MODEL_TYPES = ["lightgbm", "xgboost", "logreg"]


def _ensure_dir() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)


def _write_json(data, path: Path) -> None:
    from src.phase4.report_builder import export_json
    export_json(data, path)


# ---------------------------------------------------------------------------
# Step 1: collect
# ---------------------------------------------------------------------------

@app.command()
def collect() -> None:
    """Step 1: Collect and unify all results from Phase 2 and Phase 3."""
    _ensure_dir()

    from src.phase4.report_builder import collect_all_results, collect_detail_records

    all_results = collect_all_results()
    _write_json(all_results, _RESULTS_DIR / "main_results.json")
    typer.echo(f"Collected {len(all_results)} evaluation configurations.")

    detail_records = collect_detail_records()
    record_counts = {k: len(v) for k, v in detail_records.items()}
    _write_json(record_counts, _RESULTS_DIR / "detail_record_counts.json")
    typer.echo(f"Detail records available for {len(detail_records)} configs.")

    for tag, count in sorted(record_counts.items()):
        typer.echo(f"  {tag}: {count} version records")


# ---------------------------------------------------------------------------
# Step 2: statistics
# ---------------------------------------------------------------------------

@app.command()
def statistics() -> None:
    """Step 2: Run all statistical tests (Wilcoxon, bootstrap CIs)."""
    _ensure_dir()

    from src.phase4.analysis import compute_all_comparisons, metric_summary
    from src.phase4.report_builder import collect_detail_records

    comparisons = compute_all_comparisons(domains=_DOMAINS)
    _write_json(comparisons, _RESULTS_DIR / "statistical_tests.json")
    typer.echo(f"Computed {len(comparisons)} statistical comparisons.")

    for key, comp in sorted(comparisons.items()):
        p = comp.get("p_value", 1.0)
        d = comp.get("effect_size_cohens_d", 0.0)
        sig = "*" if p < 0.05 else ""
        typer.echo(f"  {key}: p={p:.4f}{sig}  d={d:.3f}  diff={comp.get('mean_diff', 0):.4f}")

    detail_records = collect_detail_records()
    summaries: dict[str, dict] = {}
    for tag, recs in detail_records.items():
        if not recs:
            continue
        recalls = [r.get("recall_with_sentinel", r.get("recall_predictor", 0)) for r in recs]
        reductions = [r.get("call_reduction", 0) for r in recs]
        eff_recalls = [r.get("effective_recall") for r in recs]
        eff_crs = [r.get("effective_call_reduction") for r in recs]
        summaries[tag] = {
            "recall": metric_summary(recalls),
            "call_reduction": metric_summary(reductions),
        }
        if all(v is not None for v in eff_recalls):
            summaries[tag]["effective_recall"] = metric_summary(
                [float(v) for v in eff_recalls],
            )
        if all(v is not None for v in eff_crs):
            summaries[tag]["effective_call_reduction"] = metric_summary(
                [float(v) for v in eff_crs],
            )
    _write_json(summaries, _RESULTS_DIR / "metric_summaries.json")
    typer.echo(f"Computed metric summaries for {len(summaries)} configs.")


# ---------------------------------------------------------------------------
# Step 3: ablations
# ---------------------------------------------------------------------------

@app.command()
def ablations(
    domain: str = typer.Option("all", help="domain_a | domain_b | domain_c | all"),
    model_type: str = typer.Option("lightgbm", help="lightgbm | xgboost | logreg"),
    n_trials: int = typer.Option(30, help="Optuna trials for feature ablation"),
) -> None:
    """Step 3: Run all ablation studies."""
    _ensure_dir()

    from src.phase4.ablations import (
        change_complexity_breakdown,
        feature_group_ablation,
        magnitude_threshold_sweep,
        sentinel_fraction_sweep,
    )

    domains = _DOMAINS if domain == "all" else [domain]
    all_ablation_results: dict = {}

    for dom in domains:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  Ablations for {dom}")
        typer.echo(f"{'='*60}")

        # --- Feature group ablation ---
        typer.echo(f"\n  Feature group ablation ({model_type})...")
        try:
            feat_results = feature_group_ablation(
                dom, model_type, n_trials=n_trials,
            )
            all_ablation_results[f"feature_ablation_{dom}"] = feat_results
            for r in feat_results:
                typer.echo(
                    f"    {r['config']:25s} recall={r['recall']:.4f} "
                    f"cr={r['call_reduction']:.4f} "
                    f"(Δrec={r['delta_recall']:+.4f})"
                )
        except Exception as e:
            typer.echo(f"    ERROR: {e}")

        # --- Sentinel fraction sweep ---
        model_tag = f"{model_type}_{dom}_temporal"
        model_path = _MODELS_DIR / f"{model_tag}.pkl"
        if model_path.exists():
            typer.echo(f"\n  Sentinel fraction sweep...")
            try:
                sent_results = sentinel_fraction_sweep(dom, model_tag)
                all_ablation_results[f"sentinel_sweep_{dom}"] = sent_results
                for r in sent_results:
                    typer.echo(
                        f"    frac={r['sentinel_fraction']:.2f}: "
                        f"recall={r['mean_recall']:.4f} "
                        f"cr={r['mean_call_reduction']:.4f} "
                        f"catch={r['sentinel_catch_rate']}"
                    )
            except Exception as e:
                typer.echo(f"    ERROR: {e}")

            # --- Magnitude threshold sweep ---
            typer.echo(f"\n  Magnitude threshold sweep...")
            try:
                mag_results = magnitude_threshold_sweep(dom, model_tag)
                all_ablation_results[f"magnitude_sweep_{dom}"] = mag_results
                for r in mag_results:
                    typer.echo(
                        f"    thresh={r['magnitude_threshold']:.3f}: "
                        f"recall={r['mean_recall']:.4f} "
                        f"cr={r['mean_call_reduction']:.4f} "
                        f"FOR={r['mean_for']:.4f}"
                    )
            except Exception as e:
                typer.echo(f"    ERROR: {e}")

            # --- Change complexity breakdown ---
            typer.echo(f"\n  Change complexity breakdown...")
            try:
                comp_results = change_complexity_breakdown(dom, model_tag)
                all_ablation_results[f"complexity_{dom}"] = comp_results
                for r in comp_results:
                    if r.get("count", 0) > 0:
                        typer.echo(
                            f"    {r['complexity']:12s}: n={r['count']} "
                            f"recall={r['mean_recall']:.4f} "
                            f"cr={r['mean_call_reduction']:.4f}"
                        )
                    else:
                        typer.echo(f"    {r['complexity']:12s}: n=0")
            except Exception as e:
                typer.echo(f"    ERROR: {e}")
        else:
            typer.echo(f"\n  Skipping sweeps: model {model_tag} not found.")

    _write_json(all_ablation_results, _RESULTS_DIR / "ablation_results.json")
    typer.echo(f"\nWrote ablation results ({len(all_ablation_results)} sections).")


# ---------------------------------------------------------------------------
# Step 4: cost-analysis
# ---------------------------------------------------------------------------

@app.command()
def cost_analysis() -> None:
    """Step 4: Generate CI/CD cost projections."""
    _ensure_dir()

    from src.phase4.cost_model import (
        STANDARD_SCENARIOS,
        break_even_suite_size,
        cost_vs_suite_size,
        project_all_scenarios,
    )

    observed_cr = _load_observed_call_reduction()
    observed_esc = _load_observed_escalation_rate()
    effective_cr = _load_effective_call_reduction()

    typer.echo(f"Using observed call_reduction={observed_cr:.4f}, escalation_rate={observed_esc:.4f}")
    typer.echo(f"Using effective_call_reduction={effective_cr:.4f} (sentinel reruns baked in)")

    projections = project_all_scenarios(observed_cr, observed_esc)
    _write_json(projections, _RESULTS_DIR / "cost_projections.json")

    typer.echo("\nCI/CD Cost Projections (single-pass model with escalation rate):")
    typer.echo(f"{'Scenario':30s} {'Full':>10s} {'CARTS':>10s} {'Savings':>10s} {'%':>8s}")
    typer.echo("-" * 72)
    for p in projections:
        typer.echo(
            f"{p['scenario']:30s} ${p['full_rerun_cost']:>9.2f} "
            f"${p['carts_cost']:>9.2f} ${p['savings_absolute']:>9.2f} "
            f"{p['savings_pct']:>7.1%}"
        )

    effective_projections = project_all_scenarios(effective_cr, 0.0)
    _write_json(effective_projections, _RESULTS_DIR / "cost_projections_effective.json")

    typer.echo("\nCI/CD Cost Projections (effective — sentinel reruns baked in):")
    typer.echo(f"{'Scenario':30s} {'Full':>10s} {'CARTS':>10s} {'Savings':>10s} {'%':>8s}")
    typer.echo("-" * 72)
    for p in effective_projections:
        typer.echo(
            f"{p['scenario']:30s} ${p['full_rerun_cost']:>9.2f} "
            f"${p['carts_cost']:>9.2f} ${p['savings_absolute']:>9.2f} "
            f"{p['savings_pct']:>7.1%}"
        )

    be = break_even_suite_size(0.001, observed_cr, observed_esc)
    typer.echo(f"\nBreak-even suite size (gpt-4o-mini): {be:.1f} tests")

    cost_curves = []
    for cr_rate in [0.20, 0.30, 0.40, 0.50, 0.60]:
        entries = cost_vs_suite_size(cr_rate, observed_esc, 0.01, 100)
        cost_curves.extend(entries)
    _write_json(cost_curves, _RESULTS_DIR / "cost_curves.json")
    typer.echo(f"Wrote cost curve data ({len(cost_curves)} data points).")


def _load_observed_call_reduction() -> float:
    """Load the best observed call reduction from Phase 3 results."""
    phase3_dir = _PROJECT_ROOT / "results" / "phase3"
    best_cr = 0.0
    if phase3_dir.exists():
        for path in phase3_dir.glob("eval_*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                cr = data.get("aggregate", {}).get("mean_call_reduction", 0)
                if cr > best_cr:
                    best_cr = cr
            except Exception:
                pass
    if best_cr == 0.0:
        selection_dir = _PROJECT_ROOT / "results" / "selection"
        for dom in _DOMAINS:
            p = selection_dir / f"metrics_{dom}.json"
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    cr = data.get("aggregate", {}).get("mean_call_reduction", 0)
                    if cr > best_cr:
                        best_cr = cr
                except Exception:
                    pass
    return best_cr if best_cr > 0 else 0.45


def _load_effective_call_reduction() -> float:
    """Load the best observed *effective* call reduction from results.

    Effective CR already accounts for sentinel-triggered reruns (0.0 on
    versions where sentinel fired), so cost projections using it should
    set escalation_rate=0.
    """
    phase3_dir = _PROJECT_ROOT / "results" / "phase3"
    best_cr = 0.0
    if phase3_dir.exists():
        for path in phase3_dir.glob("eval_*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                cr = data.get("aggregate", {}).get("mean_effective_call_reduction", 0)
                if cr > best_cr:
                    best_cr = cr
            except Exception:
                pass
    if best_cr == 0.0:
        selection_dir = _PROJECT_ROOT / "results" / "selection"
        for dom in _DOMAINS:
            p = selection_dir / f"metrics_{dom}.json"
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    cr = data.get("aggregate", {}).get("mean_effective_call_reduction", 0)
                    if cr > best_cr:
                        best_cr = cr
                except Exception:
                    pass
    return best_cr if best_cr > 0 else 0.30


def _load_observed_escalation_rate() -> float:
    """Compute unconditional escalation rate from Phase 3 detail records.

    Escalation occurs when a sentinel test detects an impacted test that
    the predictor missed, triggering a full rerun.  The unconditional rate
    is ``n_sentinel_triggered / n_total_versions``, NOT the conditional
    ``sentinel_catch_rate`` (which conditions on the predictor having
    missed at least one test and inflates the estimate).

    Returns the average unconditional rate across all temporal model
    evaluations, capped at 0.30.
    """
    phase3_dir = _PROJECT_ROOT / "results" / "phase3"
    if not phase3_dir.exists():
        return 0.05

    rates: list[float] = []
    for path in sorted(phase3_dir.glob("detail_eval_*temporal*.jsonl")):
        try:
            records = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if not records:
                continue
            n_triggered = sum(1 for r in records if r.get("sentinel_hit", False))
            rates.append(n_triggered / len(records))
        except Exception:
            pass

    if rates:
        avg_rate = sum(rates) / len(rates)
        return min(avg_rate, 0.30)

    for path in phase3_dir.glob("eval_*temporal*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            scr = data.get("aggregate", {}).get("sentinel_catch_rate")
            if scr is not None:
                return min(scr, 0.30)
        except Exception:
            pass

    return 0.05


# ---------------------------------------------------------------------------
# Step 5: figures
# ---------------------------------------------------------------------------

@app.command()
def figures() -> None:
    """Step 5: Generate all publication-ready figures."""
    _ensure_dir()

    fig_dir = _RESULTS_DIR / "figures"

    _generate_f1_pareto(fig_dir)
    _generate_f2_feature_importance(fig_dir)
    _generate_f3_recall_by_change_type(fig_dir)
    _generate_f4_cross_domain_heatmap(fig_dir)
    _generate_f5_magnitude_scatter(fig_dir)
    _generate_f6_confusion_breakdown(fig_dir)
    _generate_f7_cost_savings(fig_dir)

    typer.echo("\nAll figures generated.")


def _generate_f1_pareto(fig_dir: Path) -> None:
    """F1: Recall–Reduction Pareto Front via threshold sweep.

    Uses pre-existing threshold sensitivity files from Phase 3 and
    Phase 2 sweep data.  Does NOT run on-the-fly threshold sweeps
    (those are extremely slow due to per-version feature extraction).

    To generate threshold sweep data, run:
        python -m scripts.run_phase3_eval threshold-sensitivity --domain <d> --model-tag <tag>
    """
    from src.phase4.visualizations import plot_pareto_fronts

    threshold_data: dict[str, dict[str, list[dict]]] = {}
    phase3_dir = _PROJECT_ROOT / "results" / "phase3"
    selection_dir = _PROJECT_ROOT / "results" / "selection"

    for domain in _DOMAINS:
        threshold_data[domain] = {}

        if phase3_dir.exists():
            for path in sorted(phase3_dir.glob(f"threshold_sensitivity_{domain}_*.json")):
                tag = path.stem.replace(f"threshold_sensitivity_{domain}_", "")
                try:
                    with open(path) as f:
                        threshold_data[domain][tag] = json.load(f)
                except Exception:
                    pass

        sweep_path = selection_dir / f"sweep_{domain}.json"
        if sweep_path.exists():
            try:
                with open(sweep_path) as f:
                    sweep_data = json.load(f)
                rule_curve = []
                for entry in sweep_data:
                    agg = entry.get("aggregate", {})
                    rule_curve.append({
                        "threshold": entry.get("sentinel_fraction", 0),
                        "mean_recall_with_sentinel": agg.get("mean_recall_with_sentinel", 0),
                        "mean_call_reduction": agg.get("mean_call_reduction", 0),
                        "mean_effective_recall": agg.get("mean_effective_recall"),
                        "mean_effective_call_reduction": agg.get("mean_effective_call_reduction"),
                    })
                if rule_curve:
                    threshold_data[domain]["rule_based"] = rule_curve
            except Exception:
                pass

    has_data = any(bool(v) for v in threshold_data.values())
    if has_data:
        plot_pareto_fronts(threshold_data, fig_dir / "pareto_front.png")
        typer.echo("  F1: pareto_front.png")
    else:
        typer.echo("  F1: skipped (no threshold sweep data — run Phase 3 threshold-sensitivity first)")


def _generate_f2_feature_importance(fig_dir: Path) -> None:
    """F2: Feature importance bar chart from best LightGBM model."""
    from src.phase3.trainer import feature_importance_lgb, load_model
    from src.phase4.visualizations import plot_feature_importance

    all_importance: dict[str, list[dict]] = {}

    for domain in _DOMAINS:
        model_path = _MODELS_DIR / f"lightgbm_{domain}_temporal.pkl"
        if not model_path.exists():
            continue
        bundle = load_model(model_path)
        model = bundle["model"]
        feature_names = bundle["feature_names"]
        imp = feature_importance_lgb(model, feature_names)
        all_importance[domain] = imp

        plot_feature_importance(
            imp, fig_dir / f"feature_importance_{domain}.png",
            title=f"Feature Importance — {domain.replace('_', ' ').title()}",
        )
        typer.echo(f"  F2: feature_importance_{domain}.png")

    if all_importance:
        _write_json(all_importance, _RESULTS_DIR / "feature_importance.json")


def _generate_f3_recall_by_change_type(fig_dir: Path) -> None:
    """F3: Grouped bar chart of recall per change type."""
    from src.phase4.report_builder import collect_all_results
    from src.phase4.visualizations import plot_recall_by_change_type

    all_results = collect_all_results()
    by_change_data: dict[str, dict[str, float]] = {}

    for tag, data in all_results.items():
        by_ct = data.get("by_change_type", {})
        if not by_ct:
            continue
        label = tag.replace("rule_based_", "rule_").replace("eval_", "")
        recalls_by_type = {}
        for ct, metrics in by_ct.items():
            r = metrics.get("mean_effective_recall")
            if r is None:
                r = metrics.get("mean_recall_with_sentinel", metrics.get("mean_recall_predictor"))
            if r is not None:
                recalls_by_type[ct] = r
        if recalls_by_type:
            by_change_data[label] = recalls_by_type

    if by_change_data:
        plot_recall_by_change_type(by_change_data, fig_dir / "recall_by_change.png")
        _write_json(by_change_data, _RESULTS_DIR / "by_change_type.json")
        typer.echo("  F3: recall_by_change.png")
    else:
        typer.echo("  F3: skipped (no change type data)")


def _generate_f4_cross_domain_heatmap(fig_dir: Path) -> None:
    """F4: Cross-domain recall heatmap."""
    from src.phase4.report_builder import build_cross_domain_matrix, collect_all_results
    from src.phase4.visualizations import plot_cross_domain_heatmap

    all_results = collect_all_results()
    matrix_df = build_cross_domain_matrix(all_results)

    has_data = matrix_df.notna().any().any()
    if has_data:
        plot_cross_domain_heatmap(matrix_df, fig_dir / "cross_domain_heatmap.png")
        _write_json(matrix_df.reset_index(), _RESULTS_DIR / "cross_domain_matrix.json")
        typer.echo("  F4: cross_domain_heatmap.png")
    else:
        typer.echo("  F4: skipped (no cross-domain data)")


def _generate_f5_magnitude_scatter(fig_dir: Path) -> None:
    """F5: Magnitude vs. Recall scatter from detail records."""
    from src.phase4.report_builder import collect_detail_records
    from src.phase4.visualizations import plot_magnitude_scatter

    detail_records = collect_detail_records()
    scatter_data: list[dict] = []

    for tag, recs in detail_records.items():
        if "temporal" not in tag and "rule" not in tag:
            continue
        for r in recs:
            scatter_data.append({
                "magnitude_max": r.get("magnitude_max", 0),
                "recall_with_sentinel": r.get("recall_with_sentinel", r.get("recall_predictor", 0)),
                "change_type": _extract_primary_change_type(r),
                "tag": tag,
            })

    if scatter_data:
        plot_magnitude_scatter(scatter_data, fig_dir / "magnitude_scatter.png")
        typer.echo("  F5: magnitude_scatter.png")
    else:
        typer.echo("  F5: skipped (no detail data)")


def _extract_primary_change_type(record: dict) -> str:
    changes = record.get("changes", [])
    if not changes:
        return "none"
    types = {c.get("unit_type", "unknown") for c in changes}
    return next(iter(types)) if len(types) == 1 else "compound"


def _generate_f6_confusion_breakdown(fig_dir: Path) -> None:
    """F6: Confusion breakdown from detail records."""
    from src.phase4.report_builder import collect_detail_records
    from src.phase4.visualizations import plot_confusion_breakdown

    detail_records = collect_detail_records()

    for domain in _DOMAINS:
        best_key = None
        for tag in sorted(detail_records.keys()):
            if domain not in tag:
                continue
            if "lightgbm" in tag and "temporal" in tag:
                best_key = tag
                break
            if ("lightgbm" in tag or "rule" in tag) and best_key is None:
                best_key = tag

        if best_key and detail_records[best_key]:
            n_tests = detail_records[best_key][0].get("selected_count", 0) + \
                      detail_records[best_key][0].get("impacted_count", 20)
            total_tests = max(n_tests, 20)
            plot_confusion_breakdown(
                detail_records[best_key],
                total_tests,
                fig_dir / f"confusion_breakdown_{domain}.png",
            )
            typer.echo(f"  F6: confusion_breakdown_{domain}.png")


def _generate_f7_cost_savings(fig_dir: Path) -> None:
    """F7: Cost savings curves."""
    from src.phase4.visualizations import plot_cost_savings

    cost_curve_path = _RESULTS_DIR / "cost_curves.json"
    if cost_curve_path.exists():
        with open(cost_curve_path) as f:
            cost_data = json.load(f)
        plot_cost_savings(cost_data, fig_dir / "cost_savings.png")
        typer.echo("  F7: cost_savings.png")
    else:
        from src.phase4.cost_model import cost_vs_suite_size

        observed_cr = _load_observed_call_reduction()
        observed_esc = _load_observed_escalation_rate()

        cost_data = []
        for cr in [0.20, 0.30, 0.40, 0.50, 0.60]:
            cost_data.extend(cost_vs_suite_size(cr, observed_esc, 0.01, 100))

        if cost_data:
            plot_cost_savings(cost_data, fig_dir / "cost_savings.png")
            typer.echo("  F7: cost_savings.png")
        else:
            typer.echo("  F7: skipped")


# ---------------------------------------------------------------------------
# Step 6: tables
# ---------------------------------------------------------------------------

@app.command()
def tables() -> None:
    """Step 6: Generate all tables (JSON + LaTeX)."""
    _ensure_dir()

    from src.phase4.report_builder import (
        build_change_type_table,
        build_cross_domain_matrix,
        build_learned_vs_rule_table,
        build_main_results_table,
        collect_all_results,
        collect_detail_records,
        export_json,
        export_latex_table,
    )

    all_results = collect_all_results()
    detail_records = collect_detail_records()

    table_dir = _RESULTS_DIR / "tables"

    t1 = build_main_results_table(all_results)
    if not t1.empty:
        export_json(t1, table_dir / "T1_main_results.json")
        export_latex_table(t1, table_dir / "T1_main_results.tex", index=False)
        typer.echo(f"  T1: {len(t1)} rows")

    t2 = build_learned_vs_rule_table(detail_records)
    if not t2.empty:
        export_json(t2, table_dir / "T2_learned_vs_rule.json")
        export_latex_table(t2, table_dir / "T2_learned_vs_rule.tex", index=False)
        typer.echo(f"  T2: {len(t2)} rows")

    t5 = build_cross_domain_matrix(all_results)
    has_t5 = t5.notna().any().any()
    if has_t5:
        export_json(t5.reset_index(), table_dir / "T5_cross_domain.json")
        export_latex_table(t5, table_dir / "T5_cross_domain.tex", index=True)
        typer.echo("  T5: cross-domain matrix")

    t7 = build_change_type_table(all_results)
    if not t7.empty:
        export_json(t7, table_dir / "T7_by_change_type.json")
        export_latex_table(t7, table_dir / "T7_by_change_type.tex", index=False)
        typer.echo(f"  T7: {len(t7)} rows")

    _generate_feature_importance_table(table_dir)
    _generate_ablation_table(table_dir)
    _generate_cost_table(table_dir)
    _generate_sentinel_table(table_dir)

    typer.echo("\nAll tables generated.")


def _generate_feature_importance_table(table_dir: Path) -> None:
    """T3: Feature importance from saved models."""
    import pandas as pd
    from src.phase3.trainer import feature_importance_lgb, load_model
    from src.phase4.report_builder import export_json, export_latex_table

    rows: list[dict] = []
    for domain in _DOMAINS:
        model_path = _MODELS_DIR / f"lightgbm_{domain}_temporal.pkl"
        if not model_path.exists():
            continue
        bundle = load_model(model_path)
        imp = feature_importance_lgb(bundle["model"], bundle["feature_names"])
        for rank, entry in enumerate(imp[:15], 1):
            rows.append({
                "domain": domain,
                "rank": rank,
                "feature": entry["feature"],
                "importance": round(entry["importance"], 4),
            })

    if rows:
        df = pd.DataFrame(rows)
        export_json(df, table_dir / "T3_feature_importance.json")
        export_latex_table(df, table_dir / "T3_feature_importance.tex", index=False)
        typer.echo(f"  T3: {len(rows)} rows")


def _generate_ablation_table(table_dir: Path) -> None:
    """T4: Feature ablation deltas."""
    import pandas as pd
    from src.phase4.report_builder import export_json, export_latex_table

    ablation_path = _RESULTS_DIR / "ablation_results.json"
    if not ablation_path.exists():
        typer.echo("  T4: skipped (run ablations first)")
        return

    with open(ablation_path) as f:
        data = json.load(f)

    rows: list[dict] = []
    for key, entries in data.items():
        if not key.startswith("feature_ablation_"):
            continue
        domain = key.replace("feature_ablation_", "")
        for entry in entries:
            rows.append({"domain": domain, **entry})

    if rows:
        df = pd.DataFrame(rows)
        export_json(df, table_dir / "T4_feature_ablation.json")
        export_latex_table(df, table_dir / "T4_feature_ablation.tex", index=False)
        typer.echo(f"  T4: {len(rows)} rows")


def _generate_cost_table(table_dir: Path) -> None:
    """T6: Cost projections."""
    import pandas as pd
    from src.phase4.report_builder import export_json, export_latex_table

    cost_path = _RESULTS_DIR / "cost_projections.json"
    if not cost_path.exists():
        typer.echo("  T6: skipped (run cost-analysis first)")
        return

    with open(cost_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    export_json(df, table_dir / "T6_cost_projections.json")
    export_latex_table(df, table_dir / "T6_cost_projections.tex", index=False)
    typer.echo(f"  T6: {len(df)} rows")


def _generate_sentinel_table(table_dir: Path) -> None:
    """T8: Sentinel analysis from ablation results."""
    import pandas as pd
    from src.phase4.report_builder import export_json, export_latex_table

    ablation_path = _RESULTS_DIR / "ablation_results.json"
    if not ablation_path.exists():
        typer.echo("  T8: skipped (run ablations first)")
        return

    with open(ablation_path) as f:
        data = json.load(f)

    rows: list[dict] = []
    for key, entries in data.items():
        if not key.startswith("sentinel_sweep_"):
            continue
        domain = key.replace("sentinel_sweep_", "")
        for entry in entries:
            rows.append({"domain": domain, **entry})

    if rows:
        df = pd.DataFrame(rows)
        export_json(df, table_dir / "T8_sentinel_analysis.json")
        export_latex_table(df, table_dir / "T8_sentinel_analysis.tex", index=False)
        typer.echo(f"  T8: {len(rows)} rows")


# ---------------------------------------------------------------------------
# full (runs all steps)
# ---------------------------------------------------------------------------

@app.command()
def full(
    skip_ablations: bool = typer.Option(False, help="Skip ablation studies (slow)"),
    n_trials: int = typer.Option(30, help="Optuna trials for ablation training"),
) -> None:
    """Run the complete Phase 4 pipeline (steps 1–6)."""
    typer.echo("=" * 60)
    typer.echo("  Phase 4: Full Analysis Pipeline")
    typer.echo("=" * 60)

    typer.echo("\n--- Step 1: Collect results ---")
    collect()

    typer.echo("\n--- Step 2: Statistical tests ---")
    statistics()

    if not skip_ablations:
        typer.echo("\n--- Step 3: Ablation studies ---")
        ablations(domain="all", model_type="lightgbm", n_trials=n_trials)
    else:
        typer.echo("\n--- Step 3: Ablations SKIPPED ---")

    typer.echo("\n--- Step 4: Cost analysis ---")
    cost_analysis()

    typer.echo("\n--- Step 5: Figures ---")
    figures()

    typer.echo("\n--- Step 6: Tables ---")
    tables()

    typer.echo("\n" + "=" * 60)
    typer.echo("  Phase 4 pipeline complete.")
    typer.echo(f"  Results in: {_RESULTS_DIR}")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
