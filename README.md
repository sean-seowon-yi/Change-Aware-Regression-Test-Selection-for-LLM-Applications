# Change-Aware Regression Test Selection for LLM Applications

**CARTS** is a research prototype for change-aware regression test selection in LLM applications. Instead of rerunning a full eval suite after every prompt/config change, it classifies what changed, predicts which tests are likely impacted, adds a random sentinel safety sample, and measures the safety/cost tradeoff against full-rerun ground truth.

The core question is:

> Given a typed prompt/config change, can we select a much smaller test subset while still catching the tests whose outcomes would change under a full rerun?

## Current Snapshot

This repository contains an end-to-end benchmark and analysis pipeline:

- **3 benchmark domains:** `domain_a`, `domain_b`, `domain_c`
- **270 eval tests total:** 100 / 80 / 90 tests by domain
- **210 prompt versions:** 70 versions per domain (`v01`–`v70`)
- **19,170 Phase 1 LLM calls** in the reported full-rerun baseline
- **Rule-based selector** with change classifier, test tagger, impact rules, sentinels, and evaluation
- **Learned selectors:** LightGBM, XGBoost, and logistic regression
- **Analysis layer:** statistical tests, ablations, figures, tables, and cost projections

The main numeric report is [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md). It is the best starting point for interpreting the reference run, metrics, and definitions. Large `results/` outputs and `models/*.pkl` bundles are generated locally when you rerun the pipeline.

## Headline Findings

From the documented report snapshot:

- The **rule-based selector** is transparent but uneven: recall with sentinels is about **0.65** on `domain_a`, **0.62** on `domain_b`, and **0.94** on `domain_c`, with meaningful call reduction.
- On the **version-holdout split** (`v01`–`v50` train, `v51`–`v70` test), learned models improve the hard domains. `domain_b` XGBoost reaches **1.00 recall with sentinels** at about **0.60 call reduction**.
- **K-fold results** are stronger than holdout: LightGBM/XGBoost reach at least **0.92 recall with sentinels** in the reported k-fold rows.
- **Cross-domain transfer is not reliably safe**. Some directions work, but `domain_c → domain_b` is a clear failure mode with near-zero predictor recall and heavy reliance on sentinels.
- **False omission rate (FOR)** is reported alongside recall because impact is sparse. Low FOR can coexist with low recall; for safety, the target is **high recall and low FOR**, not low FOR alone.
- Phase 4 cost projections are illustrative. `cost_projections_effective.json` uses effective call reduction, where sentinel-triggered complement passes count as zero net savings for that version.

See the report for confidence intervals, Wilcoxon tests, by-change-type breakdowns, and exact artifact paths.

## Pipeline Overview

**Phase 1: Full-rerun ground truth**

- Runs every eval test against each prompt version using the harness / OpenAI Batch API path.
- Records which tests changed outcome versus the baseline.
- Outputs `results/baseline/ground_truth_domain_*.jsonl` and `summary_domain_*.json`.

**Phase 2: Rule-based selection**

- Classifies prompt diffs into typed changes such as `format`, `policy`, `workflow`, `demo_item`, and `role`.
- Tags tests by sensitivity using only test metadata and prompt text: monitor type, `tags`, `sensitive_to`, JSON key dependencies, and few-shot demo dependencies.
- Selects predicted impacted tests plus sentinels from the non-predicted pool.
- Outputs metrics, per-version details, and sentinel/magnitude sweeps under `results/selection/`.

**Phase 3: Learned selection**

- Builds one row per `(version, test)` with scalar change/test/pairwise features.
- Embeds change summaries and test inputs with `sentence-transformers/all-MiniLM-L6-v2`; raw 384-d embeddings are used for cosine similarity and PCA-compressed for model features.
- Trains/evaluates LightGBM, XGBoost, and logistic regression on version holdout, k-fold, combined k-fold, and cross-domain settings.
- Outputs model bundles in `models/` and evaluation artifacts in `results/phase3/`.

**Phase 4: Analysis and report artifacts**

- Collects Phase 2/3 results, computes statistical comparisons and metric summaries, runs ablations, builds figures/tables, and generates cost projections.
- Outputs `results/phase4/main_results.json`, `metric_summaries.json`, `statistical_tests.json`, `cost_projections*.json`, `figures/*.png`, and `tables/*.tex`.

## Metrics In Plain English

- **Recall (predictor):** of the tests that actually changed outcome, how many did the predictor select before sentinels?
- **Recall (+ sentinel):** same measurement after adding the sentinel sample.
- **Call reduction:** the fraction of tests skipped, `1 - selected / total`.
- **False omission rate (FOR):** among skipped tests, what fraction were actually impacted?
- **Sentinel catch rate:** when the predictor missed impacted tests, how often did the sentinel sample catch one?
- **Effective recall / effective call reduction:** deployment-style accounting. If a sentinel catches a miss, the policy assumes a second-wave complement pass runs the remaining suite, so that version counts as recall **1.0** and call reduction **0.0**.

For formal definitions, see [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md) §1 and `compute_effective_metrics` in `src/phase2/evaluator.py`.

## Repository Layout

- `config/settings.yaml` — model, cache, concurrency, results directory, sentinel defaults
- `data/domain_{a,b,c}/` — base prompts, eval suites, and version YAML files
- `src/phase1/` — eval harness, monitors, prompt loading/parsing/mutation, batch/full-rerun baseline
- `src/phase2/` — change classifier, test tagger, impact rules, sentinel sampler, selector, evaluator
- `src/phase3/` — feature extraction, dataset building, training, learned predictor/selector
- `src/phase4/` — result collection, statistical analysis, ablations, cost model, plots, table generation
- `scripts/` — Typer CLIs; `run_phases.py` is the unified wrapper
- `models/` — locally generated trained selector bundles (`*.pkl`)
- `results/` — locally generated benchmark outputs consumed by the analysis/report scripts
- `docs/` — phase docs, proposal summary, and full evaluation report
- `tests/` — unit/integration tests for harness, selectors, features, analysis, and metrics

## Setup

Use Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` for paths that call LLM APIs, especially Phase 1 baseline generation and any fresh eval harness runs. The learned-model feature pipeline uses local `sentence-transformers`; the first run may download `all-MiniLM-L6-v2`.

## Unified CLI

`python -m scripts.run_phases` delegates to the underlying Typer scripts and is the easiest way to run the full pipeline.

```bash
python -m scripts.run_phases list
python -m scripts.run_phases phase1 --both
python -m scripts.run_phases phase2 --all-domains --sweep
python -m scripts.run_phases phase3-train
python -m scripts.run_phases phase3-eval
python -m scripts.run_phases phase3-eval --kfold
python -m scripts.run_phases phase3-eval --compare
python -m scripts.run_phases phase4 full
```

Important CLI details:

- `phase1` delegates to `scripts.run_batch_baseline run`.
- `phase2` requires either `--all-domains` or `--domain <name>`.
- `phase3-train` delegates to `train-all` with default `--pca-components 24` and `--n-trials 50`.
- `phase3-eval --compare` rebuilds comparison summaries from existing results.
- `phase3-eval --kfold` and `--compare` are mutually exclusive.
- `phase3-eval` expects matching `models/*.pkl` bundles from training; missing model configs are skipped by the direct evaluator.
- `phase4 full` runs collect → statistics → ablations → cost → figures → tables with default ablation `--n-trials 30`. Use `--skip-ablations` for a faster report refresh.

## Direct Reproduction Commands

These are equivalent to the unified CLI and useful when you need lower-level options.

```bash
# Phase 1: full-rerun labels via OpenAI Batch API
python -m scripts.run_batch_baseline run --both

# Phase 2: rule selector + sentinel/magnitude sweep
python -m scripts.run_evaluation evaluate --all-domains --sweep

# Phase 3: train and evaluate learned selectors
# evaluate-all expects model bundles produced by train-all
python -m scripts.run_phase3_train train-all
python -m scripts.run_phase3_eval evaluate-all
python -m scripts.run_phase3_eval kfold-evaluate-all
python -m scripts.run_phase3_eval compare

# Phase 4: rebuild analysis/report artifacts
python -m scripts.run_phase4_analysis full
```

For combined k-fold only:

```bash
python -m scripts.run_phase3_eval kfold-evaluate-combined --target-domain domain_a
```

For faster Phase 4 refreshes when ablation tables are not needed:

```bash
python -m scripts.run_phase4_analysis full --skip-ablations
```

## Key Artifacts

- `results/baseline/ground_truth_domain_*.jsonl` — Phase 1 labels at `(version, test)` granularity
- `results/selection/metrics_domain_*.json` — Phase 2 rule-selector aggregate metrics
- `results/selection/detail_domain_*.jsonl` — Phase 2 per-version details
- `results/phase3/training_summary.json` — model training diagnostics
- `results/phase3/eval_summary.json` — learned selector evaluation summary
- `results/phase3/comparison_summary.json` — rule vs learned comparison inputs
- `results/phase3/kfold_summary.json` — k-fold and combined k-fold summaries
- `results/phase4/main_results.json` — collected result database
- `results/phase4/statistical_tests.json` — paired tests, CIs, and effect sizes
- `results/phase4/cost_projections.json` and `cost_projections_effective.json` — cost scenarios
- `results/phase4/figures/` and `results/phase4/tables/` — report-ready outputs

## Documentation

- [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md) — full report and source-of-truth interpretation for current numbers
- [`docs/PROPOSAL_SUMMARY.md`](docs/PROPOSAL_SUMMARY.md) — problem framing, related work, and design
- [`docs/PHASE1.md`](docs/PHASE1.md) — harness, data layout, baseline, and ground truth
- [`docs/PHASE2.md`](docs/PHASE2.md) — rule-based selector, test tagging, sentinels, and evaluator
- [`docs/PHASE3.md`](docs/PHASE3.md) — learned features, models, splits, and training/eval strategy
- [`docs/PHASE4.md`](docs/PHASE4.md) — analysis, statistics, cost model, figures, and tables

## Tests

```bash
python -m pytest tests/ -q
```

Some end-to-end reproduction paths are intentionally heavier than unit tests: Phase 1 requires API access, Phase 3 training can run Optuna/boosted models, and Phase 4 ablations may retrain models.
