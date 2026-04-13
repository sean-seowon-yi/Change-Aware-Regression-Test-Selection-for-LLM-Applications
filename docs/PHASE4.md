# Phase 4: Comprehensive Evaluation, Analysis & Report

**Goal:** Consolidate all evaluation results from Phases 1–3, answer the five primary research questions with statistical rigor, run ablation studies that isolate the contribution of each design decision, produce publication-ready tables and figures, and write the final report.

**Where the numbers live:** [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) is the narrative + tables tied to `results/` (half-up rounding). Phase 4 scripts regenerate `results/phase4/*.json`, figures, and `tables/*.tex`.

---

## 1. Problem Statement

### 1.1 What Phase 4 Is

Phases 1–3 built and evaluated the CARTS system incrementally: Phase 1 produced ground truth, Phase 2 built a rule-based selector, Phase 3 trained learned models. Each phase ran evaluations in its own context — version-holdout splits, k-fold CV, cross-domain transfers — but no single analysis has unified all results into a coherent story.

Phase 4 is the **analysis and synthesis** layer. It:

1. Aggregates Phase 2 and Phase 3 JSON already on disk (and can trigger optional slow steps such as ablations via CLI flags).
2. Performs systematic ablation studies that isolate individual contributions.
3. Computes statistical significance and confidence intervals for all key comparisons.
4. Produces a unified comparison across **3 domains × 3 model types × 4 split strategies × multiple ablation dimensions**.
5. Answers the five research questions stated in the proposal with concrete evidence.
6. Generates all tables, figures, and narratives for the final report.

### 1.2 What Phase 4 Does Not Do

Phase 4 does **not** build new models, add new features, or create new data. The entire model pipeline is frozen. If a gap is discovered (e.g., a missing cross-domain configuration), Phase 4 runs the existing training and evaluation scripts — it does not modify them.

---

## 2. Research Questions

| ID | Question | Primary Evidence |
|---|---|---|
| **RQ1** | How much call reduction does change-aware selection achieve while maintaining ≥95% regression detection recall? | Version-holdout and k-fold results across all domains and model types |
| **RQ2** | Does the learned selector outperform the rule-based selector? | Head-to-head comparison on identical version sets, with statistical significance |
| **RQ3** | How effective is the sentinel sample at catching missed regressions? | Sentinel catch rate and escalation rate, across sentinel fractions |
| **RQ4** | How does selection quality vary by change type (format-only vs. compound changes)? | Metrics broken down by `unit_type`, `change_type`, magnitude bucket, and mutation category |
| **RQ5** | What is the cost savings in realistic CI/CD scenarios? | Dollar-amount projections for representative team configurations |

---

## 3. Evaluation Configurations

### 3.1 Full Configuration Matrix

Phase 3 produces the per-run JSON under `results/phase3/`; Phase 4 **collects** them via `report_builder.collect_all_results()`. Re-run Phase 2/3 if you change evaluator semantics. **Full numeric report:** [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md).

| Split Strategy | Domain A | Domain B | Domain C | Combined |
|---|---|---|---|---|
| **Version holdout** (train v01–v50, test v51–v70) | LightGBM, XGBoost, LogReg | LightGBM, XGBoost, LogReg | LightGBM, XGBoost, LogReg | — |
| **K-Fold CV** (5-fold by version, single domain) | LightGBM, XGBoost, LogReg | LightGBM, XGBoost, LogReg | LightGBM, XGBoost, LogReg | — |
| **K-Fold Combined** (target + other domains) | LightGBM, XGBoost, LogReg | LightGBM, XGBoost, LogReg | LightGBM, XGBoost, LogReg | — |
| **Cross-Domain** (train on one, test on another) | A→B, A→C | B→A, B→C | C→A, C→B | — |

**Model types:** LightGBM (primary), XGBoost (secondary tree model), Logistic Regression (interpretable baseline).

**Baselines (change-agnostic):**
- Full rerun (100% recall, 0% reduction)
- Random 50% (change-agnostic random sample)
- Monitor-type heuristic (select by monitor family matching)
- Rule-based selector (Phase 2)

**Scale:** dozens of learned configurations (version holdout via `--split temporal`, k-fold, k-fold combined, six cross-domain directions × three model types) plus Phase 2 rule metrics and baselines — exact counts depend on which `run_phase3_eval` targets were executed.

### 3.2 Evaluation Metrics

All metrics are computed using the Phase 2 evaluator framework (`src/phase2/evaluator.py`), ensuring exact comparability across all configurations.

| Metric | Formula | Description |
|---|---|---|
| **Recall (predictor)** | \|I ∩ predicted\| / \|I\| | Impacted tests caught by model alone |
| **Recall (with sentinel)** | \|I ∩ selected\| / \|I\| | Caught by model + sentinel |
| **Call reduction** | 1 − \|selected\| / \|T\| | Fraction of tests skipped |
| **False omission rate (FOR)** | \|I ∩ (T ∖ S)\| / \|T ∖ S\| | Among tests **not selected**, fraction that were impacted (same as Phase 2 `_evaluate_baseline`) |
| **Sentinel catch rate** | (versions where sentinel caught) / (versions with any miss) | Safety net effectiveness |
| **AUROC** | Area under ROC curve | Threshold-independent discrimination |
| **AUPRC** | Area under precision-recall curve | Robust under class imbalance |
| **Effective recall** | Mean of per-version effective recall (sentinel-hit → 1.0) | Aligns cost / Pareto plots with complement-pass policy |
| **Effective call reduction** | Mean of per-version effective CR (sentinel-hit → 0) | Used in `cost_projections_effective.json` and some Phase 4 tables |

### 3.3 Aggregation Dimensions

Every metric is reported at four levels of granularity:

1. **Overall** — single number per (domain, model, split) configuration
2. **By change type** — grouped by `unit_type` (format, demonstration, demo_item, policy, workflow, role, compound)
3. **By magnitude bucket** — low (<0.05), medium (0.05–0.20), high (>0.20)
4. **By mutation category** — `llm_generated` vs. `mechanical` (and related tags in version YAML / evaluator strata; exact mapping is data-dependent)

---

## 4. Ablation Studies

Each ablation isolates a single design decision by varying it while holding everything else constant. All ablations use the version-holdout split (v01–v50 / v51–v70; CLI `temporal`) on all three domains.

### 4.1 Feature Group Ablation

**Question:** How much does each feature group contribute to model performance?

| Configuration | Features Included | Features Removed |
|---|---|---|
| Full model | All (15 change + 22 test + 5 pairwise + PCA embedding dims) | None |
| No change features | Test + pairwise + embeddings | 15 change scalars |
| No test features | Change + pairwise + embeddings | 22 test scalars |
| No pairwise features | Change + test + embeddings | 5 pairwise scalars |
| No embeddings | Change + test + pairwise (scalars only) | 48 PCA embedding dims |
| No ablation profiles | Change + test (no ablation cols) + pairwise + embeddings | 5 ablation binary columns |
| Pairwise only | 5 pairwise scalars + embeddings | Change + test scalars |

**Implementation:**

```python
def run_feature_ablation(
    domain: str,
    model_type: str = "lightgbm",
    *,
    n_trials: int = 50,
) -> list[dict]:
    """Train and evaluate with each feature group removed."""
    # 1. Load full dataset
    X, y, meta = build_dataset(domain, split="temporal")
    # 2. Define column groups
    groups = {
        "change": [c for c in X.columns if c.startswith(("unit_type_", "change_type_", "magnitude", "num_affected", "is_compound"))],
        "test": [c for c in X.columns if c.startswith(("monitor_type_", "sensitivity_cat_", "num_inferred", "has_sensitive", "ablation_"))],
        "pairwise": ["cosine_similarity", "key_overlap", "demo_overlap", "section_match", "rule_based_prediction"],
        "embeddings": [c for c in X.columns if c.startswith(("change_emb_", "test_emb_"))],
        "ablation": [c for c in X.columns if c.startswith("ablation_")],
    }
    # 3. For each ablation config, remove columns, retrain, evaluate
    ...
```

**Expected output:** A table showing recall and call_reduction for each configuration. The delta from the full model quantifies each group's contribution.

### 4.2 Sentinel Size Sweep

**Question:** How does the sentinel fraction affect the recall-reduction tradeoff and escalation rate?

| Sentinel Fraction | Expected Behavior |
|---|---|
| 0.00 | No safety net. Recall = predictor-only recall. Highest call reduction. |
| 0.05 | Minimal safety net. Slight recall boost, small call reduction loss. |
| 0.10 | Default. Balanced safety-efficiency tradeoff. |
| 0.15 | Stronger safety. More escalations, lower call reduction. |
| 0.20 | Conservative. Catches most misses, significant call reduction penalty. |

**Implementation:** Re-run the evaluator with each sentinel fraction on the holdout test slice (v51–v70), using the already-trained models.

### 4.3 Threshold Sensitivity

**Question:** How does the classification threshold τ control the recall-reduction Pareto front?

Sweep τ from 0.01 to 0.99 in steps of 0.01. For each threshold, compute recall and call_reduction across all test versions. Plot the Pareto front.

Report explicit results at fixed thresholds: τ ∈ {0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90}.

**Key insight:** The Pareto front shows the fundamental tradeoff. A steep initial slope means the model can safely skip many tests at near-perfect recall. A flat curve means aggressive thresholds are needed to gain any reduction.

### 4.4 Change Complexity Ablation

**Question:** How does the number of simultaneous changes affect selection quality?

Group versions by number of distinct `unit_type` values in their change list:
- 1 change type (single-section edits)
- 2 change types (e.g., format + demo_item)
- 3+ change types (complex compound mutations)

Report recall and call_reduction per bucket. Compound changes are expected to have lower call reduction (more tests are impacted) but potentially lower recall (harder to predict).

### 4.5 Magnitude Gating Ablation

**Question:** What is the optimal magnitude threshold for filtering trivial changes?

Sweep `magnitude_threshold` from 0.0 to 0.05 in steps of 0.005. At each threshold, versions whose max change magnitude falls below the gate are treated as "no change" (empty selected set). Measure the effect on recall and false omissions.

### 4.6 Domain Generalization Analysis

**Question:** How well do learned patterns transfer across application domains?

Compare three generalization regimes:
- **Within-domain** (k-fold CV on the target domain)
- **Cross-domain** (train on source, test on target)
- **Combined** (train on target + all other domains, k-fold CV on target)

For each target domain, plot recall and call_reduction for all three regimes. The gap between within-domain and cross-domain quantifies domain specificity. The gap between within-domain and combined quantifies the benefit of pooling data.

---

## 5. Statistical Analysis

### 5.1 Significance Testing

For each comparison (learned vs. rule-based, LightGBM vs. LogReg, within-domain vs. cross-domain), compute:

1. **Paired Wilcoxon signed-rank test** on per-version recall values. This is non-parametric and appropriate for the small sample sizes (20–70 versions per domain).
2. **Bootstrap 95% confidence intervals** on mean recall and mean call_reduction (10,000 bootstrap resamples of version-level metrics).
3. **Effect size** (Cohen's d) for the primary comparison (learned vs. rule-based).

**Implementation:**

```python
from scipy.stats import wilcoxon
import numpy as np

def significance_test(
    recalls_a: list[float],
    recalls_b: list[float],
) -> dict:
    """Paired comparison of two selector configurations."""
    stat, p_value = wilcoxon(recalls_a, recalls_b, alternative="greater")
    diff = np.array(recalls_a) - np.array(recalls_b)
    effect_size = diff.mean() / max(diff.std(), 1e-9)
    
    # Bootstrap CI
    boot_diffs = []
    rng = np.random.default_rng(42)
    for _ in range(10_000):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot_diffs.append(diff[idx].mean())
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    
    return {
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "effect_size_cohens_d": float(effect_size),
        "mean_diff": float(diff.mean()),
        "ci_95": [float(ci_lo), float(ci_hi)],
    }
```

### 5.2 Confidence Intervals for Key Metrics

For each (domain, model, split) configuration, report:
- Mean recall ± 95% CI
- Mean call_reduction ± 95% CI
- Mean FOR ± 95% CI

CIs are computed via bootstrap resampling over versions.

---

## 6. Cost Analysis

### 6.1 Cost Model

Define a parameterized cost model for realistic CI/CD scenarios:

| Parameter | Symbol | Values Considered |
|---|---|---|
| Suite size | N | 100, 200, 300, 500 |
| PRs per month | P | 50, 100, 200 |
| Cost per LLM call | c | $0.001 (gpt-4o-mini), $0.01 (gpt-4o), $0.05 (gpt-4-turbo) |
| Call reduction rate | r | Observed from evaluation |
| Sentinel fraction | s | 0.10 |
| Escalation rate | e | Observed from evaluation |

**Full rerun cost per month:**
```
C_full = P × N × c
```

**CARTS cost per month (single-pass reduction `r`):**
```
C_carts = P × [(1 − e) × N × (1 − r) + e × N] × c
       = P × N × c × [1 − r × (1 − e)]
```

The `(1 − e)` fraction of PRs uses the reduced suite; the `e` fraction escalates to full rerun.

**Effective reduction:** When building `cost_projections_effective.json`, Phase 4 uses **mean effective call reduction** from aggregates (sentinel-hit versions contribute 0 reduction), so dollar savings reflect the same policy as effective recall in the evaluator.

**Monthly savings:**
```
Savings = C_full − C_carts = P × N × c × r × (1 − e)
```

### 6.2 Scenario Projections

| Scenario | N | P | c | r (observed) | e (observed) | Monthly Full | Monthly CARTS | Savings |
|---|---|---|---|---|---|---|---|---|
| Small team, cheap model | 100 | 50 | $0.001 | 0.45 | 0.05 | $5.00 | $2.86 | $2.14 |
| Medium team, cheap model | 300 | 100 | $0.001 | 0.45 | 0.05 | $30.00 | $17.15 | $12.85 |
| Large team, expensive model | 300 | 200 | $0.01 | 0.45 | 0.05 | $600.00 | $343.00 | $257.00 |
| Enterprise, GPT-4-turbo | 500 | 200 | $0.05 | 0.45 | 0.05 | $5,000 | $2,858 | $2,142 |

*Note: `r` and `e` values above are placeholders — they will be filled with actual observed values from evaluation.*

### 6.3 Break-Even Analysis

Compute the **minimum call reduction** needed to justify the system's overhead (embedding computation, model inference, change classification). The overhead per PR is:
- Change classification: ~50ms (CPU, no LLM call)
- Feature extraction + embedding: ~2s (sentence-transformer inference for N test inputs, cached after first run)
- Model prediction: ~10ms (LightGBM inference on N rows)
- Total overhead: ~2–3 seconds per PR

This overhead is negligible compared to LLM call latency. Even 1% call reduction produces net positive value for suites with N ≥ 50.

---

## 7. Visualization & Reporting

### 7.1 Tables

| Table | Content | Format |
|---|---|---|
| **T1: Main Results** | Recall, call reduction, FOR, AUROC for all (domain, model, split) configs | 3-panel table (one per domain), rows = model × split |
| **T2: Learned vs. Rule-Based** | Head-to-head on identical version sets, with p-values and CIs | One row per domain |
| **T3: Feature Importance** | Top-15 features by gain (LightGBM) and coefficient magnitude (LogReg) | Per-domain, sorted by importance |
| **T4: Feature Ablation** | Recall/CR delta when each feature group is removed | 7 rows × 3 domains |
| **T5: Cross-Domain Transfer** | Recall matrix: rows = train domain, columns = test domain | 3×3 matrix |
| **T6: Cost Projections** | Savings for 4 scenarios | As in Section 6.2 |
| **T7: By Change Type** | Recall and CR per unit_type | Aggregated across domains |
| **T8: Sentinel Analysis** | Catch rate, escalation rate, recall boost per sentinel fraction | 5 rows × 3 domains |

### 7.2 Figures

| Figure | Content | Type |
|---|---|---|
| **F1: Recall–Reduction Pareto Front** | Threshold sweep curves for all 3 models + rule-based + baselines | Line plot, one panel per domain |
| **F2: Feature Importance Bar Chart** | Top-20 features for the best LightGBM model | Horizontal bar chart |
| **F3: Recall by Change Type** | Grouped bar chart of recall per unit_type for learned vs. rule-based | Grouped bars |
| **F4: Cross-Domain Heatmap** | 3×3 heatmap of recall (rows=train, cols=test) | Heatmap with annotations |
| **F5: Magnitude vs. Recall Scatter** | Each version as a point, x=max magnitude, y=recall, color=unit_type | Scatter plot |
| **F6: Confusion-Style Diagram** | Selected vs. impacted breakdown (TP, FP, FN, TN counts) | Stacked bar or confusion matrix |
| **F7: Cost Savings Curve** | Monthly savings vs. suite size for different call reduction rates | Multi-line plot |

### 7.3 Implementation

All figures are generated via `matplotlib` / `seaborn` and saved to `results/phase4/figures/`. Tables are generated as both JSON (for programmatic use) and LaTeX (for the report).

```python
def generate_pareto_figure(
    results: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Plot recall vs. call_reduction Pareto fronts for all selectors."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, domain in zip(axes, ["domain_a", "domain_b", "domain_c"]):
        for label, curve in results[domain].items():
            taus = [r["threshold"] for r in curve]
            recalls = [r["mean_recall_with_sentinel"] for r in curve]
            reductions = [r["mean_call_reduction"] for r in curve]
            ax.plot(reductions, recalls, label=label, marker="o", markersize=2)
        ax.set_xlabel("Call Reduction")
        ax.set_ylabel("Recall (with sentinel)")
        ax.set_title(domain.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

---

## 8. Project Structure Additions

```
src/
  phase4/
    __init__.py
    analysis.py              # Statistical analysis: significance tests, CIs, effect sizes
    ablations.py             # Feature ablation, sentinel sweep, magnitude gating
    cost_model.py            # CI/CD cost projections and break-even analysis
    report_builder.py        # Collect all results, build unified comparison tables
    visualizations.py        # Generate all figures (matplotlib/seaborn)
scripts/
  run_phases.py              # wrapper: phase4 <step> → run_phase4_analysis …
  run_phase4_analysis.py     # CLI: run full Phase 4 analysis pipeline
results/
  phase4/
    main_results.json        # T1: full configuration matrix results
    learned_vs_rule.json     # T2: head-to-head comparison with statistics
    feature_importance.json  # T3: per-domain feature rankings
    feature_ablation.json    # T4: ablation deltas
    cross_domain_matrix.json # T5: transfer matrix
    cost_projections.json    # T6: scenario projections (single-pass r)
    cost_projections_effective.json  # T6 variant using effective call reduction
    by_change_type.json      # T7: per-unit_type breakdown
    sentinel_analysis.json   # T8: sentinel fraction sweep
    statistical_tests.json   # All p-values, CIs, effect sizes
    figures/
      pareto_front.png       # F1
      feature_importance.png # F2
      recall_by_change.png   # F3
      cross_domain_heatmap.png # F4
      magnitude_scatter.png  # F5
      confusion_breakdown.png # F6
      cost_savings.png       # F7
```

---

## 9. Component-by-Component Implementation

### 9.1 Statistical Analysis Module (`src/phase4/analysis.py`)

Provides reusable functions for all statistical comparisons.

```python
def paired_comparison(
    metric_a: list[float],
    metric_b: list[float],
    *,
    alternative: str = "greater",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired Wilcoxon test + bootstrap CI for two selector configurations.
    
    Returns:
        {wilcoxon_stat, p_value, effect_size, mean_diff, ci_95_lo, ci_95_hi}
    """

def bootstrap_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a mean."""

def compute_all_comparisons(
    results_dir: Path,
) -> dict:
    """Run all pairwise comparisons across configurations.
    
    Compares:
      - Learned (LightGBM) vs. rule-based (per domain)
      - Learned (LightGBM) vs. learned (LogReg) (per domain)
      - Learned (LightGBM) vs. learned (XGBoost) (per domain)
      - Within-domain vs. cross-domain (per domain pair)
      - Combined vs. within-domain (per domain)
    """
```

### 9.2 Ablation Module (`src/phase4/ablations.py`)

Orchestrates all ablation experiments.

```python
def feature_group_ablation(
    domain: str,
    model_type: str = "lightgbm",
    *,
    split: str = "temporal",
    n_trials: int = 50,
) -> list[dict]:
    """Train and evaluate with each feature group removed.
    
    Returns list of {config_name, recall, call_reduction, auroc, delta_recall, delta_cr}.
    """

def sentinel_fraction_sweep(
    domain: str,
    model_tag: str,
    *,
    fractions: list[float] = [0.0, 0.05, 0.10, 0.15, 0.20],
    split: str = "temporal",
) -> list[dict]:
    """Evaluate the same model at different sentinel fractions.
    
    Returns list of {fraction, recall_with_sentinel, call_reduction, 
                     sentinel_catch_rate, escalation_rate}.
    """

def magnitude_threshold_sweep(
    domain: str,
    model_tag: str,
    *,
    thresholds: list[float] = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
) -> list[dict]:
    """Evaluate at different magnitude gating thresholds."""

def change_complexity_breakdown(
    domain: str,
    model_tag: str,
) -> list[dict]:
    """Break down metrics by number of change types per version."""
```

### 9.3 Cost Model Module (`src/phase4/cost_model.py`)

Parameterized CI/CD cost projections.

```python
@dataclass
class CICDScenario:
    name: str
    suite_size: int
    prs_per_month: int
    cost_per_call: float

def project_savings(
    scenario: CICDScenario,
    call_reduction: float,
    escalation_rate: float,
) -> dict:
    """Compute monthly costs and savings for a CI/CD scenario.
    
    Returns:
        {full_rerun_cost, carts_cost, savings_absolute, savings_pct,
         calls_saved_per_month, break_even_suite_size}
    """

STANDARD_SCENARIOS = [
    CICDScenario("small_team_cheap", 100, 50, 0.001),
    CICDScenario("medium_team_cheap", 300, 100, 0.001),
    CICDScenario("large_team_expensive", 300, 200, 0.01),
    CICDScenario("enterprise_premium", 500, 200, 0.05),
]
```

### 9.4 Report Builder (`src/phase4/report_builder.py`)

Collects all results into a unified JSON structure and generates LaTeX tables.

```python
def collect_all_results(results_dir: Path) -> dict:
    """Scan results/phase3/ and results/selection/ for all evaluation JSONs.
    
    Returns a nested dict keyed by (domain, model_type, split_strategy).
    """

def build_main_results_table(all_results: dict) -> pd.DataFrame:
    """T1: Main results table with all configs."""

def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """T2: Learned vs. rule-based with significance tests."""

def build_cross_domain_matrix(all_results: dict) -> pd.DataFrame:
    """T5: 3x3 recall transfer matrix."""

def export_latex_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Export all tables as .tex files for inclusion in the report."""
```

### 9.5 Visualization Module (`src/phase4/visualizations.py`)

All figure-generation functions.

```python
def plot_pareto_fronts(
    threshold_data: dict,
    output_path: Path,
) -> None:
    """F1: Recall vs. call_reduction Pareto front for all selectors."""

def plot_feature_importance(
    importance_data: dict,
    domain: str,
    output_path: Path,
    *,
    top_k: int = 20,
) -> None:
    """F2: Horizontal bar chart of feature importance."""

def plot_recall_by_change_type(
    by_change_data: dict,
    output_path: Path,
) -> None:
    """F3: Grouped bar chart comparing learned vs. rule-based per change type."""

def plot_cross_domain_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """F4: 3x3 heatmap of cross-domain recall."""

def plot_magnitude_scatter(
    per_version_data: list[dict],
    output_path: Path,
) -> None:
    """F5: Version-level scatter of magnitude vs. recall."""

def plot_confusion_breakdown(
    per_version_data: list[dict],
    output_path: Path,
) -> None:
    """F6: Stacked bar of TP/FP/FN/TN per change type."""

def plot_cost_savings(
    cost_data: list[dict],
    output_path: Path,
) -> None:
    """F7: Monthly savings vs. suite size at different call reduction rates."""
```

### 9.6 CLI Script (`scripts/run_phase4_analysis.py`)

A single entry point that orchestrates the entire Phase 4 pipeline.

```python
app = typer.Typer(help="Phase 4: Comprehensive evaluation, analysis & report generation.")

@app.command()
def collect():
    """Step 1: Collect and unify all results from Phase 2 and Phase 3."""

@app.command()
def statistics():
    """Step 2: Run all statistical tests (Wilcoxon, bootstrap CIs)."""

@app.command()
def ablations(
    domain: str = typer.Option("all", help="domain_a | domain_b | domain_c | all"),
    model_type: str = typer.Option("lightgbm", help="lightgbm | xgboost | logreg"),
):
    """Step 3: Run all ablation studies."""

@app.command()
def cost_analysis():
    """Step 4: Generate CI/CD cost projections (CLI: ``cost-analysis``)."""

@app.command()
def figures():
    """Step 5: Generate all publication-ready figures."""

@app.command()
def tables():
    """Step 6: Generate all tables (JSON + LaTeX)."""

@app.command()
def full():
    """Run the complete Phase 4 pipeline (steps 1-6)."""
```

---

## 10. Implementation status

The modules below exist in `src/phase4/` (`analysis.py`, `report_builder.py`, `visualizations.py`, `cost_model.py`, `ablations.py`) with `python -m scripts.run_phase4_analysis <command>` or `python -m scripts.run_phases phase4 <step>`. The sprint checklist that follows is the original plan; treat unchecked items as **optional extensions** unless you are bootstrapping a new checkout.

### Sprint 1: Result Collection & Statistical Framework

**Goal:** Build the infrastructure to load, unify, and statistically compare all existing results.

**Tasks:**

- [ ] Create `src/phase4/__init__.py`
- [ ] Implement `src/phase4/analysis.py`
  - Paired Wilcoxon signed-rank test
  - Bootstrap confidence intervals
  - Cohen's d effect size
  - All pairwise comparisons for learned vs. rule-based, model type comparisons, and domain transfer comparisons
- [ ] Implement `src/phase4/report_builder.py`
  - Result collection: scan `results/phase3/` and `results/selection/` for all JSON files
  - Unified data structure keyed by (domain, model, split)
  - Main results table builder (T1)
  - Learned vs. rule-based comparison builder (T2)
- [ ] Implement `scripts/run_phase4_analysis.py` — `collect` and `statistics` commands
- [ ] Re-run any Phase 3 evaluations that used the old FOR formula or the old sentinel_catch_rate logic, to ensure all stored results reflect the corrected evaluator
- [ ] Verify: load results for all 3 domains × 3 models × 4 splits, confirm no gaps

**Validation:** `collect` produces a unified JSON with at least 27 entries (3 domains × 3 models × 3 splits). `statistics` produces p-values and CIs for all comparisons.

---

### Sprint 2: Ablation Studies

**Goal:** Run all ablation experiments and produce quantitative evidence for each design decision.

**Tasks:**

- [ ] Implement `src/phase4/ablations.py`
  - Feature group ablation (7 configurations × 3 domains)
  - Sentinel fraction sweep (5 fractions × 3 domains)
  - Magnitude threshold sweep (6 thresholds × 3 domains)
  - Change complexity breakdown (3 complexity buckets × 3 domains)
- [ ] Run feature group ablations
  - For each domain, train 7 LightGBM models (full, −change, −test, −pairwise, −embeddings, −ablation, pairwise-only) on version holdout
  - Evaluate each on v51–v70 test set
  - Record recall and call_reduction deltas
- [ ] Run sentinel fraction sweep
  - Use pre-trained version-holdout LightGBM models (`*_temporal.pkl`)
  - Re-run evaluation at each sentinel fraction
  - Record recall, call_reduction, sentinel_catch_rate, escalation_rate
- [ ] Run magnitude threshold sweep
  - Re-run evaluation at each magnitude gating threshold
  - Record recall and false omissions
- [ ] Run change complexity breakdown
  - Group versions by number of distinct unit_types
  - Report per-bucket metrics
- [ ] Extend `scripts/run_phase4_analysis.py` — `ablations` command

**Validation:**
- Feature ablation: removing pairwise features should cause the largest recall drop (strongest signal)
- Sentinel sweep: recall should monotonically increase with sentinel fraction
- Magnitude sweep: FOR should increase as the threshold increases (more versions gated out)
- Complexity: compound changes should have lower call reduction than single-section changes

---

### Sprint 3: Cost Analysis, Visualizations & Report

**Goal:** Produce all figures, tables, cost projections, and the final synthesis.

**Tasks:**

- [ ] Implement `src/phase4/cost_model.py`
  - Parameterized cost projections for all standard scenarios
  - Break-even analysis
  - Sensitivity analysis (cost vs. suite size, cost vs. reduction rate)
- [ ] Implement `src/phase4/visualizations.py`
  - F1: Pareto front (threshold sweep curves, 3-panel)
  - F2: Feature importance bar chart
  - F3: Recall by change type (grouped bars)
  - F4: Cross-domain heatmap
  - F5: Magnitude vs. recall scatter
  - F6: Confusion breakdown
  - F7: Cost savings curves
- [ ] Generate all figures and save to `results/phase4/figures/`
- [ ] Generate all tables (JSON + LaTeX) and save to `results/phase4/`
- [ ] Add `matplotlib` and `seaborn` to `requirements.txt`
- [ ] Run the full pipeline: `python -m scripts.run_phase4_analysis full`
- [ ] Write the final interpretation (see Section 11)

**Validation:**
- All 7 figures render without errors
- All 8 tables have complete data (no NaN or missing cells)
- Cost projections show positive savings for all scenarios with call_reduction > 0
- LaTeX tables compile cleanly

---

## 11. Research Question Analysis Framework

### RQ1: Call Reduction at ≥95% Recall

**Evidence:** Report the operating point on the Pareto front (F1) where recall (with sentinel) first exceeds 0.95. Read off the corresponding call reduction.

**Expected narrative:**
- "At the 95% recall operating point, the version-holdout LightGBM model achieves X% call reduction on domain_a, Y% on domain_b, and Z% on domain_c."
- "This represents a Nx reduction in LLM evaluation calls per prompt change."

**Supporting evidence:** Threshold sweep table (Section 4.3), cost projections (T6).

### RQ2: Learned vs. Rule-Based

**Evidence:** T2 (head-to-head comparison), with Wilcoxon p-value and bootstrap CI.

**Expected narrative:**
- "The learned selector achieves X±Y% recall vs. the rule-based selector's A±B% recall (Wilcoxon p=P, Cohen's d=D), representing a Z percentage-point improvement."
- "The improvement is concentrated in [change types where rules fail — policy, workflow, cross-section effects]."

**Supporting evidence:** F3 (recall by change type), T7.

### RQ3: Sentinel Effectiveness

**Evidence:** T8 (sentinel analysis), sentinel fraction sweep (Section 4.2).

**Expected narrative:**
- "The 10% sentinel sample catches X% of predictor misses, adding Y percentage points of recall at a cost of Z percentage points of call reduction."
- "Increasing sentinel fraction to 20% catches A% of misses but reduces call savings by B percentage points — diminishing returns."

**Supporting evidence:** Escalation rate across configurations.

### RQ4: Variation by Change Type

**Evidence:** T7 (by change type), F3 (grouped bars), F5 (magnitude scatter).

**Expected narrative:**
- "Format changes achieve the highest call reduction (X%) because they impact a narrow, predictable set of tests."
- "Role and compound changes achieve the lowest call reduction (Y%) because their impact is diffuse."
- "The learned model closes the recall gap on policy and workflow changes, where rules previously failed."

**Supporting evidence:** Feature importance (T3, F2) — the `section_match` and `key_overlap` features should rank highly for targeted changes; embeddings should rank highly for diffuse changes.

### RQ5: Cost Savings

**Evidence:** T6 (cost projections), F7 (cost curves).

**Expected narrative:**
- "For a medium-sized team (300-test suite, 100 PRs/month, gpt-4o-mini), CARTS reduces monthly eval cost from $X to $Y, saving $Z/month."
- "For enterprises using GPT-4o (500-test suite, 200 PRs/month), annual savings exceed $A."

**Supporting evidence:** Break-even analysis shows CARTS is cost-positive for suites with N ≥ B tests.

---

## 12. Success Criteria

| Criterion | Target | Notes |
|---|---|---|
| Phase 4 pipeline | `full` run completes | `results/phase4/` populated; cross-check [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) |
| RQ1 answered with concrete numbers | Call reduction at ≥95% recall, per domain | With 95% CIs |
| RQ2 answered with significance test | Learned vs. rule-based, p < 0.05 | Wilcoxon test on per-version recall |
| RQ3 sentinel analysis complete | Catch rate, escalation rate per sentinel fraction | 5 fractions × 3 domains |
| RQ4 change-type breakdown complete | Per-unit_type metrics for all selectors | 7 change types × 3 domains |
| RQ5 cost projections for ≥4 scenarios | With concrete dollar amounts | Based on observed call reduction and escalation rate |
| Feature ablation shows interpretable contributions | Each group's removal causes measurable recall drop | Pairwise features expected to be most important |
| All figures render without errors | 7 publication-quality figures | PNG at 150+ DPI |
| All tables exportable to LaTeX | 8 tables | Compilable .tex files |
| No oracle leakage confirmed | Verified across all evaluation paths | No ground truth features in any model input |
| Report narrative for each RQ | Written interpretation with evidence citations | Clear, concise, evidence-backed |

---

## 13. Dependencies

### New Python Packages

| Package | Purpose | Version |
|---|---|---|
| `matplotlib` | Figure generation | ≥3.8 |
| `seaborn` | Statistical visualizations | ≥0.13 |
| `scipy` | Wilcoxon test, statistical functions | ≥1.11 |

All other dependencies are already installed from Phases 1–3.

### Data Dependencies

| Artifact | Source | Required |
|---|---|---|
| Ground truth JSONL files | Phase 1 | Yes — all 3 domains |
| Trained model `.pkl` bundles | Phase 3 | Yes — all 27 model configs |
| Phase 2 evaluation results | Phase 2 | Yes — rule-based metrics per domain |
| Phase 3 evaluation results | Phase 3 | Yes — learned metrics per domain |
| Sensitivity profiles | Phase 3 ablation | Yes — per domain |

---

## 14. What Phase 4 Produces

By the end of this phase:

1. **A unified results database** — all evaluation configurations in a single JSON structure, queryable by domain, model type, and split strategy.
2. **Statistical evidence** — p-values, confidence intervals, and effect sizes for all key comparisons.
3. **Ablation evidence** — quantified contribution of each feature group, sentinel fraction, magnitude threshold, and change complexity.
4. **Cost projections** — concrete dollar savings for realistic CI/CD scenarios, with break-even analysis.
5. **Publication-ready artifacts** — 7 figures (PNG) and 8 tables (JSON + LaTeX) ready for the final report.
6. **Research question answers** — concrete, evidence-backed narrative for each of the 5 RQs.
7. **A reproducible analysis pipeline** — `python -m scripts.run_phase4_analysis full` (or `python -m scripts.run_phases phase4 full`) regenerates everything from scratch.

---

## 15. Relationship to Earlier Phases

| Phase | Role in Phase 4 |
|---|---|
| **Phase 1** | Provides ground truth (the "answer key" for all evaluation). No changes. |
| **Phase 2** | Provides rule-based selector results (the baseline). Evaluator framework is reused for all Phase 4 metrics. No changes to the code. |
| **Phase 3** | Provides trained models and learned selector results. Training and evaluation scripts are invoked by Phase 4 for ablations but not modified. |
| **Phase 4** | Pure analysis layer. Reads outputs from Phases 1–3, computes statistics, generates figures and tables. The only new code is in `src/phase4/` and `scripts/run_phase4_analysis.py`. |
