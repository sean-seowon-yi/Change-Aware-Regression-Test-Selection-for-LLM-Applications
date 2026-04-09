# CARTS evaluation results (full report)

This document ties **research goals**, **definitions**, **every numeric result** in the committed `results/` tree, and **interpretation** for the end-to-end pipeline: Phase 1 (LLM ground truth), Phase 2 (rule-based selector), Phase 3 (learned selectors and cross-domain transfer), and Phase 4 (statistics and cost projections).

**Reading guide.** If a column header or term is unclear (e.g. *version*, *impact*, *n*), skim **§1** (selector metrics) and **§1.1** (dataset vocabulary) first. Phase 1 (**§3**) explains how ground-truth labels are produced; Phases 2–4 assume those definitions.

**Source of truth:** JSON and JSONL under `results/baseline/`, `results/selection/`, `results/phase3/`, and `results/phase4/`. Display values use **half-up** (“regular” rounding away from zero at the last retained digit), matching `decimal.Decimal.quantize(..., ROUND_HALF_UP)`: **two** decimal places for recall, call reduction, and other means unless noted; **three** for false omission rate (FOR), Cohen’s *d*, mean differences, and confidence-interval endpoints where shown. (Python’s built-in `round()` uses banker's rounding on ties and is **not** used here.)

---

## 0. Research goals and what each phase tests

**Problem.** LLM-backed applications change often (prompts, tools, policies). Running a full evaluation suite on every change is expensive; skipping tests risks missing regressions. CARTS asks whether **change-aware test selection**—predicting which tests are impacted, optionally augmented with **random sentinels**—can preserve safety while cutting invocations.

**Goals (mapped to evidence).**

1. **Obtain trustworthy labels** for "which tests fail after a version change" so selectors can be scored fairly. *Phase 1* reruns the full suite per version with a fixed model and records pass/fail and impacted sets.

2. **Establish a transparent, non-learned baseline** using diff- and metadata-driven rules plus sentinels. *Phase 2* reports aggregate and stratified (change type, magnitude, category) behavior.

3. **Learn from history** whether gradient-boosted trees or a linear model can **outperform rules** on **held-out** versions (version holdout) and under **cross-validation** when more training data is available (k-fold), including **multi-domain** training.

4. **Quantify transfer failure modes** when a model trained on domain *X* is evaluated on domain *Y* without retraining. *Phase 3* cross-domain rows and `eval_summary.json` provide the full matrix.

5. **Stress-test statistical claims** with paired tests on per-version recall (+ sentinel). *Phase 4* `statistical_tests.json` records Wilcoxon statistics, *p*-values, Cohen's *d*, and bootstrap CIs on the mean paired difference.

6. **Translate selection rates into illustrative cost** under stated assumptions. *Phase 4* `cost_projections.json` holds scenario-level dollars and call counts; the **formula** and **code loader** behavior are documented below so numbers are not misread.

---

## 1. Metrics and definitions

| Metric | Definition | How to read it |
|--------|------------|----------------|
| **Recall (predictor)** | Fraction of **ground-truth impacted** tests (§1.1) that appear in the model's selected set **before** sentinels are added. | Measures **pure ranking/threshold** quality. |
| **Recall (+ sentinel)** | Recall after unioning the predictor's set with a **random** subset of tests **not** in the predictor's set. Implementation: `sample_sentinels` in `src/phase2/sentinel.py` draws `max(5, ⌈0.10 × pool_size⌉)` from the non-predicted pool (Phase 2 `selector.py`; same helper in Phase 3 `learned_selector.py`). | Safety net: even a weak predictor can recover recall if the non-predicted pool is large enough for sentinels to sample. |
| **Call reduction** | `1 − selected_count / all_tests_count`. | Higher ⇒ fewer monitor/LLM calls per version. |
| **FOR (false omission rate)** | Among tests **not** selected, the fraction that were **actually** impacted: `impacted ∩ not_selected` / `not_selected`. | Risk in the **skipped** pool; complements recall. Lower is better. |
| **Sentinel catch rate** | When the predictor misses impacted tests, the fraction of those misses caught by sentinels (version-level aggregation in metrics JSON). | High catch rate ⇒ sentinels compensate for predictor gaps on that stratum. |

**Baselines (Phase 2 `metrics_domain_*.json`, `baselines`):**

- **full_rerun:** recall 1.0, call reduction 0 (upper bound on safety, no savings).
- **random_50:** random 50% of tests; recall near 0.5× stochastic expectation, call reduction 0.5.
- **monitor_heuristic:** a simple monitoring-oriented heuristic from the harness; interpret as a **non-learned** alternative to rules.

### 1.1 Vocabulary: versions, domains, and “impact”

These terms appear in Phase 1 summaries and throughout the report. They describe the **synthetic benchmark**, not a live production app.

| Term | Meaning |
|------|--------|
| **Domain** (`domain_a`, `domain_b`, `domain_c`) | Three separate **scenario families** (different base prompts, test suites, and YAML “version” timelines). Metrics are **never** mixed across domains unless a table explicitly says **combined** training. |
| **Version** (e.g. `v01` … `v70`) | One **snapshot** of a prompt/configuration after a scripted change from the previous version. Each version is a row in the dataset: same **test IDs**, possibly different pass/fail outcomes vs. the **baseline**. |
| **Baseline (for labeling)** | The **prior** version’s outcomes (or stored reference). Phase 1 compares the current version’s eval results to this reference to decide if a test **flipped**. |
| **Test / suite** | A single eval case (one LLM call in the harness). **Tests per version** is the suite size (fixed within a domain). |
| **Impacted test** | A test whose outcome **differs** from the baseline for that version (treated as “would need re-checking” in production). Synonym in JSON: ground-truth **positive** for selection. |
| **Impacted count (per version)** | Number of impacted tests in that version (`total_impacted` in `summary_domain_*.json`). **Zero** means the change did not flip any test vs. baseline. |
| **Version with impact** | Any version where **impacted count ≥ 1**. The fraction **x / 70** is “how many of the 70 versions had at least one regression/outcome change.” |
| **Stratum *n* (Phase 2–4 tables)** | Unless stated otherwise, **number of versions** in that row’s slice (e.g. all versions tagged `policy`, or all in a magnitude bucket)—**not** the number of tests. |

**Version holdout:** **Train** versions are **v01–v50**; **held-out test** versions are **v51–v70** (20 versions). This is a **fixed version-range** split in the synthetic benchmark (later versions use harder mutation families), not literal wall-clock time. **K-fold** and **all-versions** rows use all 70 unless noted. *Code:* CLI and artifacts still use the string `temporal` (e.g. `--split temporal`, `*_temporal.pkl`, `detail_eval_*temporal*.jsonl`).

---

## 2. Executive summary (one page)

1. **Rules** reach moderate aggregate recall+s on **domain_a** and **domain_b** (0.65 and 0.62) with FOR ≈ 0.013 (a) and 0.030 (b). **domain_c** is much easier for rules: recall+s **0.94**, call reduction **0.47**, FOR **0.007**.

2. **Version-holdout learned models** (train v01–v50, test v51–v70, *n* = 20 per domain) **improve** recall+s on **domain_a** and **domain_b** versus the **rule baseline on the same v51–v70 slice** (see §5.2). On **domain_b**, **XGBoost** reaches **1.0** recall+s with call reduction **0.60**; **logistic regression** keeps high recall+s (**0.91**) but **sacrifices** call reduction (**0.28**) versus trees on that slice.

3. **Cross-domain** evaluation shows **severe transfer degradation** in several directions (e.g. **domain_c → domain_b**: predictor recall **0.01–0.02**, recall+s **≈0.27**, heavy reliance on sentinels). **domain_a → domain_b** with **logreg** is an exception: recall+s **0.93** but call reduction only **0.06** (near full suite).

4. **5-fold GroupKFold** (single-domain and **combined** multi-domain training) yields **high** recall+s for **LightGBM/XGBoost** (≥ **0.92** in every row of `kfold_summary.json` in this run). **Combined** **XGBoost** on **domain_c** reaches **0.997** recall+s (raw **0.9971** in JSON; **1.00** at two-decimal rounding) and **0.78** call reduction.

5. **Wilcoxon** paired tests on per-version recall+s: **domain_a** holdout vs. rule is **not** significant at α = 0.05 (small *n*); **k-fold** vs. rule is **highly** significant. **domain_b**: version-holdout and k-fold both **favor** learned models (*p* < 0.05). **domain_c**: most comparisons **non-significant**; **logreg (holdout)** vs. rule shows a **negative** mean difference (rules better on this slice).

6. **Cost projections** in committed `cost_projections.json` use **call_reduction = 0.9** and **escalation_rate = 0.3**, hence **savings_pct = 0.63** via `r × (1 − e)`. Current code can recompute escalation from version-holdout detail files (`detail_eval_*temporal*.jsonl`); **re-run** `cost-analysis` to align JSON with the loader (§8.4).

7. **Feature importance** (LightGBM split importance) identifies **`key_overlap`** as dominant on **domain_a** and **domain_c**, while **`num_affected_demo_labels`** leads on **domain_b**, reflecting structurally different change–test interactions across domains.

---

## 3. Phase 1 — Ground truth (LLM baseline)

### 3.1 Goal

Provide **version-level** ground truth: after each synthetic **version** change, the harness records which tests **differ** from the **baseline** (prior version). That set is the ground truth for “impacted” in Phases 2–3. Phase 1 uses a **single** evaluator model and batch setup so labels are reproducible.

### 3.2 Observed scale (latest committed summaries)

Each row is one **domain** (scenario family). **LLM calls** = suite size × number of versions run in Phase 1 (full suite per version).

| Domain   | Tests per suite | Version count | Total LLM calls | Cost (USD) | Wall clock (s) | Base pass rate |
|----------|-----------------|---------------|-----------------|------------|----------------|----------------|
| domain_a | 100             | 70            | 7,100           | $0.95      | 8,437          | 0.95           |
| domain_b | 80              | 70            | 5,680           | $1.15      | 3,278          | 0.82           |
| domain_c | 90              | 70            | 6,390           | $1.52      | 1,897          | 0.94           |

**Base pass rate:** fraction of **test runs** that pass when averaged across versions (a coarse difficulty signal for the domain, not a selector metric).

### 3.3 How often do versions change outcomes? (Impact distribution)

**Impacted tests** are defined in §1.1. Here we summarize **per domain**, across all **70 versions**.

| Domain   | Versions with ≥1 impacted test (of 70) | Mean impacted tests per version | Max impacted (one version) | Min impacted | Version counts by impacted-test bucket: 0 \| 1–5 \| 6–15 \| 16–30 \| 31+ |
|----------|------------------------------------------|-----------------------------------|----------------------------|--------------|--------------------------------------------------------------------------|
| domain_a | 67 / 70                                  | 10.86                             | 48                         | 0            | 3 / 50 / 3 / 1 / 13                                                      |
| domain_b | 69 / 70                                  | 6.20                              | 28                         | 0            | 1 / 56 / 0 / 13 / 0                                                      |
| domain_c | 24 / 70                                  | 7.13                              | 35                         | 0            | 46 / 9 / 1 / 0 / 14                                                      |

**How to read the last column:** five numbers separated by `/`. They are **counts of versions** (not tests). Example for **domain_a**: **3** versions have **zero** impacted tests; **50** versions have **1–5** impacted tests; **3** have **6–15**; **1** has **16–30**; **13** have **31+**. So the row answers “how many versions fall in each impact-size bucket?”

**Interpretation:** **domain_a** has widespread but often small impacts (50 versions with 1–5 impacted tests, but a long tail up to 48). **domain_b** is densely impacted (69/70 versions) with a cluster of 16–30 impact sizes. **domain_c** is bimodal: **46 out of 70** versions have **zero** impacted tests, but when impact occurs it can be large (14 versions with 31+ impacted tests). This bimodality explains why domain_c is "easy" for rules on aggregate—most versions need no selection at all—but challenging on the impacted subset. **Mean impacted** uses **two-decimal** half-up over the 70 versions.

### 3.4 Artifacts

`results/baseline/ground_truth_domain_*.jsonl`, `summary_domain_*.json` (counts, costs, wall time, batch ids, per-version impact sizes).

Phase 1 does **not** score a selector; it **fixes the yardstick**. **Downstream** recall numbers are only meaningful relative to this labeling.

---

## 4. Phase 2 — Rule-based selector

**Settings (aggregate table):** **sentinel fraction 0.10** on the **non-predicted** pool with **minimum 5** sentinels (`sample_sentinels`), magnitude threshold **0.005** (`run_evaluation` defaults in `scripts/run_evaluation.py` / `src/phase2/evaluator.py`), evaluated over **all 70 versions** per domain.

**Row scope:** Unless a section title says **version holdout** or **k-fold**, Phase 2 metrics are **averages over all 70 versions** in that domain (one datapoint per version per metric).

### 4.1 Goal

Show what a **transparent** change-diff rule set achieves without training, and **where** it breaks (change type, magnitude, LLM vs. mechanical versions).

### 4.2 Aggregate metrics (all versions)

**Perfect recall (versions):** Count of versions where **recall with sentinel** reached **1.0** (selector + sentinels caught every ground-truth impacted test for that version). **x / 70** compares that count to all versions in the domain.

| Domain   | Recall (pred) | Recall (+s) | Call reduction | FOR (3dp) | Sentinel catch | Perfect recall (versions) |
|----------|---------------|-------------|----------------|-----------|----------------|----------------------------|
| domain_a | 0.57          | 0.65        | 0.46           | 0.013     | 0.33           | 35 / 70                    |
| domain_b | 0.54          | 0.62        | 0.37           | 0.030     | 0.33           | 30 / 70                    |
| domain_c | 0.93          | 0.94        | 0.47           | 0.007     | 0.29           | 64 / 70                    |

**Confidence intervals** (from `metric_summaries.json`, **95% CI** on the **per-version** recall+s and call-reduction series; **n = 70** versions per domain):

| Domain   | Recall+s mean | Std dev | 95% CI             | CR mean | CR 95% CI          |
|----------|---------------|---------|---------------------|---------|---------------------|
| domain_a | 0.654         | 0.427   | [0.553, 0.753]      | 0.456   | [0.380, 0.529]      |
| domain_b | 0.624         | 0.378   | [0.537, 0.712]      | 0.371   | [0.305, 0.438]      |
| domain_c | 0.938         | 0.229   | [0.880, 0.986]      | 0.471   | [0.407, 0.532]      |

**Interpretation:** The wide standard deviations (especially **domain_a**: 0.43) reflect high version-to-version variance in impact magnitude and rule applicability. The CIs show that domain_c's recall advantage over a/b is statistically clear, while the overlap between domain_a and domain_b CIs suggests they may not differ significantly.

*Note:* JSON `mean_false_omission_rate` for domain_a is **0.0129** (displays as **0.013** at three decimals).

### 4.3 Heuristic baselines (same harness)

| Domain   | Baseline           | Mean recall | Call reduction |
|----------|--------------------|------------|----------------|
| domain_a | random_50          | 0.50       | 0.50           |
| domain_a | monitor_heuristic  | 0.76       | 0.46           |
| domain_b | random_50          | 0.56       | 0.50           |
| domain_b | monitor_heuristic  | 0.53       | 0.41           |
| domain_c | random_50          | 0.84       | 0.50           |
| domain_c | monitor_heuristic  | 0.93       | 0.47           |

**Interpretation:** On **domain_a**, the **monitor_heuristic** achieves **higher** recall than rules at similar call reduction—rules are **not** unbeatable. On **domain_b**, **monitor_heuristic** underperforms rules on recall. **domain_c** approaches **full-rerun** recall for both rules and monitor_heuristic, reflecting easier structure.

### 4.4 By change type (rule-based)

**Column *n*:** number of **versions** whose primary change type matches the row (see §1.1). Metrics are **means over those versions only**.

**domain_a:**

| Type           | Versions (*n*) | R_pred | R+s  | CR   | FOR   | Sent. catch |
|----------------|----------------|--------|------|------|-------|-------------|
| compound       | 18  | 0.97   | 0.98 | 0.15 | 0.001 | 0.90        |
| demo_item      | 5   | 0.40   | 0.47 | 0.84 | 0.016 | 0.33        |
| demonstration  | 10  | 0.49   | 0.49 | 0.72 | 0.017 | 0.00        |
| format         | 11  | 0.40   | 0.45 | 0.51 | 0.031 | 0.10        |
| policy         | 11  | 0.13   | 0.13 | 0.76 | 0.024 | 0.00        |
| role           | 8   | 1.00   | 1.00 | 0.00 | 0.00  | —           |
| workflow       | 7   | 0.29   | 0.93 | 0.54 | 0.005 | 0.83        |

**domain_b:**

| Type           | Versions (*n*) | R_pred | R+s  | CR   | FOR   | Sent. catch |
|----------------|----------------|--------|------|------|-------|-------------|
| compound       | 19  | 0.86   | 0.90 | 0.11 | 0.025 | 0.70        |
| demo_item      | 5   | 0.40   | 0.40 | 0.61 | 0.025 | 0.00        |
| demonstration  | 10  | 0.34   | 0.34 | 0.50 | 0.043 | 0.00        |
| format         | 9   | 0.35   | 0.35 | 0.25 | 0.072 | 0.00        |
| policy         | 11  | 0.38   | 0.83 | 0.79 | 0.007 | 0.91        |
| role           | 8   | 1.00   | 1.00 | 0.00 | 0.00  | —           |
| workflow       | 8   | 0.09   | 0.09 | 0.61 | 0.043 | 0.00        |

**domain_c:**

| Type           | Versions (*n*) | R_pred | R+s  | CR   | FOR   | Sent. catch |
|----------------|----------------|--------|------|------|-------|-------------|
| compound       | 22  | 0.91   | 0.92 | 0.36 | 0.020 | 0.50        |
| demo_item      | 5   | 1.00   | 1.00 | 0.72 | 0.00  | —           |
| demonstration  | 6   | 0.67   | 0.67 | 0.72 | 0.005 | 0.00        |
| format         | 10  | 0.99   | 1.00 | 0.55 | 0.003 | 0.50        |
| policy         | 12  | 0.96   | 0.96 | 0.68 | 0.001 | 0.00        |
| role           | 8   | 1.00   | 1.00 | 0.00 | 0.00  | —           |
| workflow       | 7   | 1.00   | 1.00 | 0.50 | 0.00  | —           |

**Interpretation:** **Policy** and **demonstration** changes on **domain_a** are **catastrophic** for rules unless sentinels help; **workflow** is saved almost entirely by **sentinels** (catch **0.83**). **domain_b workflow** remains **unsalvaged** by sentinels in this run (**0.09** recall+s). **domain_c** is uniformly strong except **demonstration** (still **0.67** recall+s). The **role** change type achieves **perfect** recall across all three domains because the rule selector selects **all** tests for role changes (CR = 0.00).

### 4.5 By magnitude bucket

**Column *n*:** versions in that **magnitude** bucket (size of change signal from metadata).

| Domain   | Bucket | Versions (*n*) | R_pred | R+s  | CR   | FOR   |
|----------|--------|----------------|--------|------|------|-------|
| domain_a | high   | 30  | 0.58   | 0.72 | 0.51 | 0.010 |
| domain_a | medium | 21  | 0.57   | 0.62 | 0.39 | 0.017 |
| domain_a | low    | 19  | 0.57   | 0.60 | 0.45 | 0.012 |
| domain_b | high   | 24  | 0.42   | 0.45 | 0.43 | 0.038 |
| domain_b | medium | 22  | 0.60   | 0.70 | 0.29 | 0.027 |
| domain_b | low    | 24  | 0.60   | 0.73 | 0.38 | 0.025 |
| domain_c | high   | 26  | 0.87   | 0.87 | 0.51 | 0.003 |
| domain_c | medium | 22  | 1.00   | 1.00 | 0.42 | 0.001 |
| domain_c | low    | 22  | 0.95   | 0.96 | 0.47 | 0.019 |

**Interpretation:** **domain_b** **high-magnitude** rows are **hardest** for rules (recall+s **0.45**). **domain_c** **medium** bucket shows **saturated** recall with aggressive selection (CR **0.42**).

### 4.6 By category (LLM-generated vs. mechanical)

**Column *n*:** versions labeled with that **category** (how the version’s edits were produced in the data generator).

| Domain   | Category      | Versions (*n*) | R_pred | R+s  | CR   | FOR   |
|----------|---------------|----------------|--------|------|------|-------|
| domain_a | llm_generated | 20  | 0.74   | 0.75 | 0.44 | 0.014 |
| domain_a | mechanical    | 50  | 0.51   | 0.62 | 0.46 | 0.012 |
| domain_b | llm_generated | 20  | 0.59   | 0.66 | 0.37 | 0.041 |
| domain_b | mechanical    | 50  | 0.52   | 0.61 | 0.37 | 0.025 |
| domain_c | llm_generated | 20  | 0.92   | 0.93 | 0.43 | 0.002 |
| domain_c | mechanical    | 50  | 0.94   | 0.94 | 0.49 | 0.009 |

**Interpretation:** LLM-generated changes tend to show slightly higher predictor recall than mechanical changes on domain_a and domain_b, consistent with the hypothesis that LLM-generated edits are more semantically coherent and thus easier for rules to classify. Domain_c shows no meaningful split between categories.

### 4.7 Parameter sensitivity sweep

The pipeline evaluates all combinations of **4** sentinel fractions × **4** magnitude thresholds (**16** configurations per domain). Key rows:

| Domain   | Sent. frac | Mag. thresh | R_pred | R+s  | CR   | FOR    |
|----------|-----------|-------------|--------|------|------|--------|
| domain_a | 0.05      | 0.000       | 0.57   | 0.65 | 0.47 | 0.013  |
| domain_a | 0.10      | 0.005       | 0.57   | 0.65 | 0.46 | 0.013  |
| domain_a | 0.20      | 0.000       | 0.57   | 0.73 | 0.41 | 0.012  |
| domain_a | 0.10      | 0.020       | 0.37   | 0.49 | 0.59 | 0.103  |
| domain_b | 0.05      | 0.000       | 0.54   | 0.62 | 0.38 | 0.030  |
| domain_b | 0.20      | 0.000       | 0.54   | 0.69 | 0.34 | 0.028  |
| domain_b | 0.10      | 0.020       | 0.36   | 0.47 | 0.51 | 0.087  |
| domain_c | 0.05      | 0.000       | 0.95   | 0.95 | 0.45 | 0.001  |
| domain_c | 0.10      | 0.005       | 0.93   | 0.94 | 0.47 | 0.007  |
| domain_c | 0.20      | 0.020       | 0.77   | 0.82 | 0.53 | 0.071  |

**Key findings from the sweep:**

- **Magnitude threshold** is the dominant lever: raising it from **0.005 → 0.02** dramatically increases call reduction but **collapses** recall and inflates FOR (e.g. domain_a: recall+s drops from **0.65 → 0.49**, FOR rises from **0.013 → 0.103**). The default **0.005** is a safe operating point.
- **Sentinel fraction** has a **smaller** effect on recall+s but trades off against call reduction. Increasing from **0.05 → 0.20** boosts domain_a recall+s from **0.65 → 0.73** at the cost of **6 pp** in call reduction. The default **0.10** balances safety and savings.
- **domain_c** is robust to both parameters: even at (sf=0.05, mt=0.005) it maintains **0.94** recall+s.

**Artifacts:** `results/selection/metrics_domain_*.json`, `sweep_domain_*.json`, `detail_domain_*.jsonl`.

---

## 5. Phase 3 — Learned selectors

### 5.1 Training and evaluation modes

- **Version holdout:** The model sees **training versions v01–v50** only; metrics on **v51–v70** are the **held-out** slice (**n = 20** versions per eval domain). This mimics deploying after observing a long history, then scoring on new PRs. Thresholds per model are in `comparison_summary.json`.
- **K-fold:** **5** folds; **GroupKFold** keeps each **version** entirely in train **or** test (no leakage of the same version into both). **single_domain** = train only on that domain’s versions; **combined** = train on **all three** domains’ versions but still report metrics on the **named target** domain.
- **Cross-domain:** Model trained on **source** domain’s versions, evaluated on **every** version of the **target** domain (**70** test points). Model tags encode direction, e.g. `lightgbm_domain_b_cross_ba` → trained on **B**, tested on **A**.

### 5.2 Training diagnostics (`training_summary.json`)

| Domain   | Split        | Model   | Train AUROC | Test AUROC | Train recall | Train CR |
|----------|-------------|---------|-------------|------------|-------------|---------|
| domain_a | holdout     | LGBM    | 1.000       | 0.880      | 1.00        | 0.84    |
| domain_a | holdout     | XGB     | 1.000       | 0.826      | 1.00        | 0.85    |
| domain_a | holdout     | logreg  | 0.994       | 0.842      | 1.00        | 0.71    |
| domain_b | holdout     | LGBM    | 1.000       | 0.936      | 1.00        | 0.75    |
| domain_b | holdout     | XGB     | 1.000       | 0.946      | 1.00        | 0.63    |
| domain_b | holdout     | logreg  | 0.942       | 0.685      | 1.00        | 0.28    |
| domain_c | holdout     | LGBM    | 1.000       | 0.997      | 1.00        | 0.88    |
| domain_c | holdout     | XGB     | 1.000       | 0.997      | 1.00        | 0.83    |
| domain_c | holdout     | logreg  | 0.996       | 0.928      | 0.98        | 0.89    |

**Cross-domain training AUROC** (selected — full table in `training_summary.json`):

| Source → Target | Model   | Train AUROC | Test AUROC |
|-----------------|---------|-------------|------------|
| A → B           | LGBM    | 1.000       | 0.555      |
| A → B           | XGB     | 1.000       | 0.680      |
| A → C           | LGBM    | 1.000       | 0.975      |
| A → C           | XGB     | 1.000       | 0.978      |
| B → A           | LGBM    | 1.000       | 0.842      |
| B → A           | XGB     | 1.000       | 0.859      |
| C → A           | LGBM    | 1.000       | 0.910      |
| C → B           | LGBM    | 1.000       | 0.672      |
| C → B           | XGB     | 1.000       | 0.495      |
| C → B           | logreg  | 0.997       | 0.448      |

**Interpretation:**

- **Version-holdout split**: All tree models achieve **1.000 train AUROC** (perfect separation on training data) with some **generalization gap** to test. The gap is largest for **domain_a** (test AUROC **0.826–0.880**) and smallest for **domain_c** (**0.997**). This confirms domain_c's structural regularity.
- **Logreg** shows lower train AUROC (**0.94** on domain_b) indicating the linear model cannot perfectly separate the training features. Its test AUROC (**0.685** on domain_b) is substantially below trees, consistent with the recall gaps seen at evaluation time.
- **Cross-domain** test AUROC shows the transfer problem quantitatively: **C → B** drops to **0.495** (XGB) and **0.448** (logreg)—at or **below** random-guess level (AUROC **0.5** for balanced binary labels)—matching the near-zero predictor recall in §5.4.

### 5.3 Decision thresholds (`comparison_summary.json`)

The pipeline selects per-model probability thresholds that optimize recall on the version-holdout validation set:

| Domain   | Model   | Threshold |
|----------|---------|-----------|
| domain_a | LGBM    | 0.02      |
| domain_a | XGB     | 0.01      |
| domain_a | logreg  | 0.08      |
| domain_b | LGBM    | 0.01      |
| domain_b | XGB     | 0.01      |
| domain_b | logreg  | 0.01      |
| domain_c | LGBM    | 0.01      |
| domain_c | XGB     | 0.01      |
| domain_c | logreg  | 0.33      |

**Interpretation:** Trees use very low thresholds (**0.01–0.02**), reflecting their ability to produce sharp probability estimates that separate impacted and non-impacted tests. **Logreg** requires higher thresholds on domain_a (**0.08**) and domain_c (**0.33**) because its probability distribution is more diffuse—a consequence of the linear decision boundary.

### 5.4 Version holdout (*n* = 20): full table

Each row is one **evaluated domain** × **model family**. All numbers are **only** for versions **v51–v70** (the **20-version** test split; see §5.1)—not averaged over all 70.

| Eval domain | Model   | R_pred | R+s   | Call red. | FOR    | Sent. catch |
|-------------|---------|--------|-------|-----------|--------|-------------|
| domain_a    | LGBM    | 0.83   | 0.84  | 0.82      | 0.019  | 0.20        |
| domain_a    | XGB     | 0.82   | 0.82  | 0.82      | 0.019  | 0.20        |
| domain_a    | logreg  | 0.86   | 0.86  | 0.76      | 0.021  | 0.20        |
| domain_b    | LGBM    | 0.95   | 0.95  | 0.70      | 0.005  | 0.00        |
| domain_b    | XGB     | 0.98   | **1.00** | 0.60   | **0.000** | **1.00** |
| domain_b    | logreg  | 0.90   | 0.91  | 0.28      | 0.007  | 0.33        |
| domain_c    | LGBM    | 0.92   | 0.93  | 0.83      | 0.001  | 0.33        |
| domain_c    | XGB     | 0.98   | 0.98  | 0.80      | 0.001  | 0.00        |
| domain_c    | logreg  | 0.80   | 0.80  | 0.82      | 0.003  | 0.20        |

**Version-holdout confidence intervals** (from `metric_summaries.json`):

| Eval domain | Model   | R+s 95% CI          | CR 95% CI           |
|-------------|---------|----------------------|----------------------|
| domain_a    | LGBM    | [0.702, 0.955]       | [0.757, 0.860]       |
| domain_a    | XGB     | [0.675, 0.955]       | [0.763, 0.870]       |
| domain_a    | logreg  | [0.723, 0.973]       | [0.687, 0.822]       |
| domain_b    | LGBM    | [0.891, 0.991]       | [0.626, 0.768]       |
| domain_b    | XGB     | [1.000, 1.000]       | [0.506, 0.684]       |
| domain_b    | logreg  | [0.775, 1.000]       | [0.183, 0.400]       |
| domain_c    | LGBM    | [0.800, 1.000]       | [0.760, 0.891]       |
| domain_c    | XGB     | [0.925, 1.000]       | [0.724, 0.858]       |
| domain_c    | logreg  | [0.600, 0.950]       | [0.712, 0.900]       |

**Apples-to-apples vs. rules on v51–v70** (from `T2_learned_vs_rule.json`): rule recall+s on the **holdout slice** is **0.746 / 0.662 / 0.925** for **a / b / c** (vs. all-version aggregates of 0.654 / 0.624 / 0.938). Learned models **beat** rules on **a** and **b** (all three families); on **c**, **trees** match or **edge** rules while **logreg** **lags** (**0.80** vs. **0.93** rule slice).

**Interpretation:** **Trees** dominate the **recall vs. call reduction** frontier on **domain_b** under version holdout; **logreg** **over-selects** (low call reduction) on **b** while staying relatively strong on **a**. XGBoost on domain_b achieves the **only perfect holdout recall+s** (**1.00**) with the sentinel catching every missed test (sentinel catch rate **1.00**). LGBM on domain_b has a sentinel catch rate of **0.00** despite lower recall—it missed tests on versions where sentinels did not happen to sample them.

### 5.5 Cross-domain: full matrix (`eval_summary.json`)

**Convention:** `train_domain → eval_domain`. FOR shown at three decimals.

**Compact cross-domain recall matrix** (XGBoost per direction, from `cross_domain_matrix.json`):

| Train ↓ / Eval → | domain_a | domain_b | domain_c |
|-------------------|----------|----------|----------|
| domain_a          | —        | 0.35     | 0.87     |
| domain_b          | 0.44     | —        | 0.83     |
| domain_c          | 0.28     | 0.28     | —        |

**Interpretation of the heatmap:** **A → C** and **B → C** are the strongest transfer directions (**0.87**, **0.83**). **C → anything** is the weakest source, reflecting domain_c's unusual bimodal impact structure. These values correspond to XGBoost (the matrix builder records one model tag per direction; see `report_builder.py`).

#### Eval **domain_a** (trained on B or C)

| Train | Model   | R_pred | R+s   | CR   | FOR   |
|-------|---------|--------|-------|------|-------|
| B     | LGBM    | 0.23   | 0.36  | 0.75 | 0.039 |
| B     | XGB     | 0.21   | 0.44  | 0.76 | 0.059 |
| B     | logreg  | 0.68   | 0.71  | 0.52 | 0.171 |
| C     | LGBM    | 0.23   | 0.27  | 0.82 | 0.028 |
| C     | XGB     | 0.24   | 0.28  | 0.81 | 0.027 |
| C     | logreg  | 0.13   | 0.20  | 0.86 | 0.076 |

**Interpretation:** **B→A** with **logreg** is a **partial** transfer success (recall+s **0.71**) but **high FOR** in the skipped pool (**0.17**). **C→A** stays **weak** on recall+s (**0.20–0.28**) despite **high** call reduction—**sentinels** are doing most safety work.

#### Eval **domain_b** (trained on A or C)

| Train | Model   | R_pred | R+s   | CR   | FOR   |
|-------|---------|--------|-------|------|-------|
| A     | LGBM    | 0.08   | 0.32  | 0.83 | 0.073 |
| A     | XGB     | 0.13   | 0.35  | 0.80 | 0.063 |
| A     | logreg  | 0.91   | 0.93  | 0.06 | 0.012 |
| C     | LGBM    | 0.01   | 0.27  | 0.90 | 0.072 |
| C     | XGB     | 0.02   | 0.28  | 0.88 | 0.073 |
| C     | logreg  | 0.01   | 0.27  | 0.90 | 0.072 |

**Interpretation:** **A→B logreg** memorizes **enough shared structure** to get **0.93** recall+s but **abandons** efficiency (**0.06** call reduction). **C→B** is a **failure mode** for **impact prediction** (pred recall **≈0.01–0.02**): recall+s **≈0.27** implies **sentinel-dominated** safety.

#### Eval **domain_c** (trained on A or B)

| Train | Model   | R_pred | R+s   | CR   | FOR   |
|-------|---------|--------|-------|------|-------|
| A     | LGBM    | 0.93   | 0.93  | 0.77 | 0.006 |
| A     | XGB     | 0.87   | 0.87  | 0.81 | 0.007 |
| A     | logreg  | 0.96   | 0.96  | 0.16 | 0.002 |
| B     | LGBM    | 0.86   | 0.86  | 0.64 | 0.014 |
| B     | XGB     | 0.82   | 0.83  | 0.77 | 0.019 |
| B     | logreg  | 0.81   | 0.87  | 0.43 | 0.172 |

**Interpretation:** **A→C** **trees** achieve **strong** recall+s with **high** call reduction; **logreg** again trades **tiny** call reduction (**0.16**) for recall. **B→C logreg** shows **elevated FOR** (**0.17**) despite **0.87** recall+s—risk is concentrated in **non-selected** tests.

### 5.6 Five-fold cross-validation: complete `kfold_summary.json`

Each row summarizes **out-of-fold** scores pooled across folds (each **version** appears in the test fold exactly once). **Eval domain** is whose versions the metrics refer to; **combined** mode still reports that target domain’s performance after training on pooled data. Abbreviations match §1 (**R_pred**, **R+s**, call reduction, FOR).

| Eval domain | Mode          | Model   | R_pred | R+s   | Call red. | FOR   |
|-------------|---------------|---------|--------|-------|-----------|-------|
| domain_a    | single_domain | LGBM    | 0.92   | 0.92  | 0.70      | 0.012 |
| domain_a    | single_domain | XGB     | 0.92   | 0.92  | 0.71      | 0.009 |
| domain_a    | single_domain | logreg  | 0.96   | 0.96  | 0.31      | 0.007 |
| domain_b    | single_domain | LGBM    | 0.90   | 0.92  | 0.70      | 0.010 |
| domain_b    | single_domain | XGB     | 0.91   | 0.93  | 0.68      | 0.010 |
| domain_b    | single_domain | logreg  | 0.84   | 0.88  | 0.40      | 0.014 |
| domain_c    | single_domain | LGBM    | 0.96   | 0.97  | 0.81      | 0.008 |
| domain_c    | single_domain | XGB     | 0.96   | 0.97  | 0.80      | 0.007 |
| domain_c    | single_domain | logreg  | 0.90   | 0.90  | 0.69      | 0.002 |
| domain_a    | combined      | LGBM    | 0.97   | 0.97  | 0.47      | 0.005 |
| domain_a    | combined      | XGB     | 0.93   | 0.93  | 0.72      | 0.007 |
| domain_a    | combined      | logreg  | 0.98   | 0.98  | 0.39      | 0.005 |
| domain_b    | combined      | LGBM    | 0.97   | 0.98  | 0.42      | 0.013 |
| domain_b    | combined      | XGB     | 0.96   | 0.97  | 0.58      | 0.009 |
| domain_b    | combined      | logreg  | 0.91   | 0.92  | 0.26      | 0.008 |
| domain_c    | combined      | LGBM    | 0.98   | 0.98  | 0.70      | 0.005 |
| domain_c    | combined      | XGB     | **1.00** | **1.00** | **0.78** | 0.002 |
| domain_c    | combined      | logreg  | 0.90   | 0.92  | 0.63      | 0.004 |

**K-fold confidence intervals** (selected, from `metric_summaries.json`):

| Config | R+s 95% CI          | CR 95% CI           |
|--------|----------------------|----------------------|
| domain_a / single / LGBM | [0.872, 0.968] | [0.661, 0.741] |
| domain_b / single / XGB  | [0.885, 0.969] | [0.629, 0.727] |
| domain_c / combined / XGB | [0.991, 1.000] | [0.740, 0.819] |
| domain_a / combined / LGBM | [0.936, 1.000] | [0.416, 0.520] |
| domain_b / combined / LGBM | [0.959, 0.995] | [0.372, 0.475] |

**Interpretation:** **More data** (k-fold and especially **combined**) **compresses variance** and pushes recall+s toward **saturation** for **tree** models. **Logreg** systematically **selects larger** sets (lower call reduction) except where **threshold tuning** interacts with **class balance**. **domain_c + combined + XGB** is the **best** joint recall+s / call reduction point in this artifact (JSON: recall_predictor **0.9955**, recall_with_sentinel **0.9971**—both **1.00** at two decimals; call_reduction **0.7806**).

The **combined** mode trades call reduction for higher recall stability: e.g. domain_a combined LGBM (**0.97** R+s, **0.47** CR) vs. single LGBM (**0.92** R+s, **0.70** CR). Combined training adds ~3× data volume but the extra domains introduce feature noise that reduces the model's ability to prune aggressively, explaining the CR drop.

**Artifacts:** `models/*.pkl`, `results/phase3/eval_summary.json`, `comparison_summary.json`, `kfold_summary.json`, `training_summary.json`, `detail_eval_*.jsonl`, per-run `eval_*.json`.

---

## 6. Phase 4 — Feature importance

### 6.1 Goal

Understand **which features** drive the learned models' predictions, to validate that the models are learning meaningful change–test relationships rather than spurious correlations.

### 6.2 Top features by domain (LightGBM split importance)

**domain_a:**

| Rank | Feature              | Importance |
|------|----------------------|------------|
| 1    | key_overlap          | 161,334    |
| 2    | test_pca_0           | 17,279     |
| 3    | num_affected_keys    | 15,418     |
| 4    | test_pca_11          | 13,892     |
| 5    | change_pca_17        | 12,604     |

**domain_b:**

| Rank | Feature                  | Importance |
|------|--------------------------|------------|
| 1    | num_affected_demo_labels | 17,048     |
| 2    | test_pca_8               | 11,778     |
| 3    | test_pca_3               | 10,130     |
| 4    | is_compound              | 8,826      |
| 5    | unit_type_workflow       | 5,855      |

**domain_c:**

| Rank | Feature              | Importance |
|------|----------------------|------------|
| 1    | key_overlap          | 157,849    |
| 2    | section_match        | 16,955     |
| 3    | num_affected_keys    | 10,358     |
| 4    | change_pca_1         | 10,247     |
| 5    | num_inferred_key_deps| 7,544      |

**Interpretation:** **`key_overlap`** (the overlap between changed prompt keys and test-relevant keys) dominates both domain_a and domain_c by an **order of magnitude** over the next feature, confirming that the key-dependency structure is the primary signal for predicting test impact. **domain_b** has a flatter feature distribution led by **`num_affected_demo_labels`** (count of demonstration labels touched by the change), reflecting domain_b's richer demonstration-based prompt structure. The presence of **PCA components** in all three domains (test_pca_*, change_pca_*) indicates the models also leverage learned embeddings of test and change representations.

**Artifact:** `results/phase4/feature_importance.json`, per-domain figures at `results/phase4/figures/feature_importance_domain_*.png`.

---

## 7. Phase 4 — Statistical tests (`statistical_tests.json`)

**Procedure:** Paired **Wilcoxon signed-rank** on per-version **recall (+ sentinel)**; learned series aligned to rule series on the **intersection** of version IDs in `detail_domain_*.jsonl` and the corresponding learned detail file (**20** versions for **holdout**, **70** for **k-fold** where coverage is full). Effect size: **Cohen's *d*** on paired differences; **95% bootstrap CI** on mean difference (`mean_diff` = learned − rule).

### 7.1 Learned vs. rule (all rows)

**Column *n*:** number of **paired** versions (same version IDs for rule and learned series). **Mean diff** = learned recall+s minus rule recall+s, averaged over those versions; **95% CI** is the bootstrap interval for that paired mean (see procedure note above the table).

| Comparison | Versions (*n*) | *p* | Cohen's *d* | Mean diff | 95% CI |
|------------|----------------|-----|-------------|-----------|--------|
| LGBM holdout vs. rule, domain_a | 20 | 0.165 | 0.17 | +0.093 | [−0.131, 0.334] |
| XGB holdout vs. rule, domain_a | 20 | 0.211 | 0.14 | +0.078 | [−0.148, 0.325] |
| logreg holdout vs. rule, domain_a | 20 | 0.128 | 0.21 | +0.113 | [−0.105, 0.348] |
| LGBM k-fold vs. rule, domain_a | 70 | 2.42e−5 | 0.56 | +0.270 | [0.156, 0.383] |
| XGB k-fold vs. rule, domain_a | 70 | 2.65e−5 | 0.54 | +0.268 | [0.152, 0.383] |
| logreg k-fold vs. rule, domain_a | 70 | 3.10e−6 | 0.65 | +0.307 | [0.197, 0.418] |
| LGBM holdout vs. rule, domain_b | 20 | 0.002 | 0.84 | +0.284 | [0.146, 0.425] |
| XGB holdout vs. rule, domain_b | 20 | 0.001 | 0.97 | +0.338 | [0.200, 0.488] |
| logreg holdout vs. rule, domain_b | 20 | 0.034 | 0.48 | +0.246 | [0.021, 0.450] |
| LGBM k-fold vs. rule, domain_b | 70 | 1.38e−6 | 0.68 | +0.299 | [0.197, 0.399] |
| XGB k-fold vs. rule, domain_b | 70 | 8.62e−7 | 0.73 | +0.306 | [0.211, 0.402] |
| logreg k-fold vs. rule, domain_b | 70 | 1.01e−4 | 0.55 | +0.252 | [0.143, 0.359] |
| LGBM holdout vs. rule, domain_c | 20 | 0.500 | 0.00 | 0.000 | [−0.150, 0.150] |
| XGB holdout vs. rule, domain_c | 20 | 0.159 | 0.22 | +0.050 | [0.000, 0.150] |
| logreg holdout vs. rule, domain_c | 20 | 0.949 | −0.39 | −0.125 | [−0.275, 0.000] |
| LGBM k-fold vs. rule, domain_c | 70 | 0.261 | 0.10 | +0.028 | [−0.037, 0.094] |
| XGB k-fold vs. rule, domain_c | 70 | 0.140 | 0.11 | +0.029 | [−0.033, 0.094] |
| logreg k-fold vs. rule, domain_c | 70 | 0.935 | −0.17 | −0.039 | [−0.096, 0.011] |

**Interpretation:** **Statistical significance tracks domain difficulty and sample size.** **domain_a** version-holdout improvements are **visible in means** but **not** significant at α = 0.05 with **20** pairs; **k-fold** on **70** versions **is** significant (**medium** effect **0.54–0.65**). **domain_b** shows **large** version-holdout effects for **trees** (*d* **0.84–0.97**). **domain_c** tests are **mostly inconclusive** for **trees**; **logreg** is **worse** than rules on the holdout slice (**negative** mean diff, *p* **0.95**).

### 7.2 Learned vs. rule with confidence intervals (`T2_learned_vs_rule.json`)

This table adds **95% CIs on both rule and learned means** for direct comparison. **Versions (*n*)** is the number of versions underlying each mean (paired where applicable).

| Domain   | Model   | Split    | Versions (*n*) | Rule R+s [CI]           | Learned R+s [CI]         | Diff  | *p*     | *d*  |
|----------|---------|----------|----------------|-------------------------|--------------------------|-------|---------|------|
| domain_a | LGBM    | holdout  | 20  | 0.746 [0.558, 0.904]    | 0.839 [0.702, 0.955]     | +0.09 | 0.165   | 0.17 |
| domain_a | LGBM    | kfold    | 70  | 0.654 [0.553, 0.753]    | 0.973 [0.935, 1.000]     | +0.32 | 1.3e−6  | 0.70 |
| domain_a | XGB     | kfold    | 70  | 0.654 [0.553, 0.753]    | 0.931 [0.882, 0.973]     | +0.28 | 1.1e−5  | 0.58 |
| domain_a | logreg  | kfold    | 70  | 0.654 [0.553, 0.753]    | 0.977 [0.947, 1.000]     | +0.32 | 1.0e−6  | 0.71 |
| domain_b | XGB     | holdout  | 20  | 0.662 [0.512, 0.800]    | 1.000 [1.000, 1.000]     | +0.34 | 0.001   | 0.97 |
| domain_b | XGB     | kfold    | 70  | 0.624 [0.537, 0.712]    | 0.970 [0.937, 0.995]     | +0.35 | 9.0e−8  | 0.86 |
| domain_c | XGB     | kfold    | 70  | 0.938 [0.880, 0.986]    | 0.997 [0.991, 1.000]     | +0.06 | 0.022   | 0.26 |
| domain_c | logreg  | holdout  | 20  | 0.925 [0.800, 1.000]    | 0.800 [0.600, 0.950]     | −0.13 | 0.949   | −0.39|

**Interpretation:** The CIs make it visually clear that **k-fold** improvements on domain_a and domain_b have **non-overlapping** CI ranges for rule vs. learned. On domain_c, the intervals overlap substantially for holdout comparisons, confirming the non-significant *p*-values.

### 7.3 Model vs. model (selected)

**Versions (*n*):** paired versions in the Wilcoxon comparison. **Mean diff** = first-named model minus second (e.g. LGBM − XGB).

| Comparison | Versions (*n*) | *p* | Mean diff | Note |
|------------|------------------|-----|-----------|------|
| LGBM vs. XGB, domain_a holdout | 20 | 0.09 | +0.015 | LightGBM slightly higher recall+s on average (`mean_diff` = LGBM − XGB; n.s.) |
| LGBM vs. XGB, domain_a k-fold | 70 | 0.500 | +0.001 | Essentially tied |
| LGBM vs. logreg, domain_a holdout | 20 | 0.500 | −0.021 | Tied on recall+s |
| LGBM vs. logreg, domain_a k-fold | 70 | 0.988 | −0.037 | Logreg higher recall+s |
| LGBM vs. XGB, domain_b holdout | 20 | 0.971 | −0.054 | Tied (XGB slightly higher) |
| LGBM vs. XGB, domain_b k-fold | 70 | 0.600 | −0.007 | Tied |
| LGBM vs. logreg, domain_b holdout | 20 | 0.231 | +0.038 | n.s. (LGBM slightly higher on average) |
| LGBM vs. logreg, domain_b k-fold | 70 | 0.110 | +0.047 | n.s. |
| LGBM vs. logreg, domain_c holdout | 20 | 0.051 | +0.125 | Borderline; LGBM higher |
| LGBM vs. logreg, domain_c k-fold | 70 | 0.050 | +0.066 | Significant at 0.05; LGBM higher |

**Interpretation:** No tree-vs-tree comparison reaches significance, confirming that **LightGBM and XGBoost are interchangeable** for this task. The **tree vs. logreg** gap is most visible on **domain_c** where the nonlinear models can exploit the bimodal impact structure that a linear decision boundary cannot capture. On domain_a/b, logreg compensates by over-selecting (lower call reduction).

---

## 8. Cost projections (`results/phase4/cost_projections.json`)

### 8.1 Goal

Illustrate **relative** savings of selective evaluation vs. full suite re-run under **stated** assumptions, not to claim measured production totals.

### 8.2 Formula (as implemented)

With **call reduction** `r` and **escalation rate** `e` (fraction of runs that trigger a full fallback), **savings_pct ≈ r × (1 − e)**. Committed JSON uses **r = 0.9**, **e = 0.3** ⇒ **savings_pct = 0.63** for every scenario row.

### 8.3 All scenarios (committed snapshot)

| Scenario            | Suite | PRs/mo | $/call | Full $ | CARTS $ | Save $ | Save % | Calls saved/mo | Annual $ |
|---------------------|-------|--------|--------|--------|---------|--------|--------|----------------|----------|
| small_team_cheap    | 100   | 50     | 0.001  | 5.0    | 1.85    | 3.15   | 0.63   | 3,150           | 37.8     |
| medium_team_cheap   | 300   | 100    | 0.001  | 30.0   | 11.1    | 18.9   | 0.63   | 18,900          | 226.8    |
| large_team_standard | 300   | 200    | 0.01   | 600.0  | 222.0   | 378.0  | 0.63   | 37,800          | 4,536.0  |
| enterprise_premium  | 500   | 200    | 0.05   | 5000.0 | 1850.0  | 3150.0 | 0.63   | 63,000          | 37,800.0 |

### 8.4 Cost sensitivity curves (`cost_curves.json`)

The pipeline also generates **45** data points varying suite size (50–1000) and call reduction (0.2–0.6) with fixed escalation rate 0.3 and cost/call $0.01. Selected rows:

| Suite size | Call reduction | Savings % | Calls saved/mo | Annual savings |
|------------|---------------|-----------|----------------|----------------|
| 50         | 0.2           | 14%       | 700            | $84            |
| 50         | 0.6           | 42%       | 2,100          | $252           |
| 300        | 0.2           | 14%       | 4,200          | $504           |
| 300        | 0.6           | 42%       | 12,600         | $1,512         |
| 1000       | 0.2           | 14%       | 14,000         | $1,680         |
| 1000       | 0.6           | 42%       | 42,000         | $5,040         |

**Interpretation:** Savings scale **linearly** with suite size and call reduction at fixed escalation rate. The 0.63 savings in §8.3 assumes 0.9 call reduction (optimistic). Version-holdout evaluations in §5.4 span call reduction **0.28–0.83** across models; with **e = 0.3**, illustrative savings **`r × (1 − e)`** span about **0.20–0.58** (e.g. **0.28 × 0.7 ≈ 0.20** for domain_b logreg up to **0.83 × 0.7 ≈ 0.58** for domain_c LGBM).

### 8.5 Loader note (code vs. committed JSON)

`scripts/run_phase4_analysis.py` **cost-analysis** can set **call_reduction** from Phase 3 aggregates and **escalation_rate** from mean **sentinel_hit** rates in `detail_eval_*temporal*.jsonl` (capped). The **committed** `cost_projections.json` reflects the **snapshot** inputs (**0.9**, **0.3**). **Re-run** `python -m scripts.run_phase4_analysis cost-analysis` after pipeline changes to refresh dollars and percentages consistently.

**Break-even (example):** With default CLI assumptions (overhead, latency, `cost_per_call=0.001`) and **r = 0.9**, **e = 0.3**, reported break-even suite size is **≈2.4** tests—an **order-of-magnitude** sanity check, not a deployment recommendation.

---

## 9. Synthesis: goals vs. evidence

| Goal | Verdict in this run | Evidence |
|------|---------------------|----------|
| Cheap, reproducible labels | Met | Phase 1 summaries: 19,170 total calls across 3 domains, $3.62 total cost |
| Rule baseline + stress cases | Met | Phase 2 aggregates, by-type tables, and 16-config sweep per domain |
| Learned > rules on hard domains when enough data | **Mostly met** | domain_b version-holdout + k-fold; domain_a k-fold significant (*d* 0.54–0.71) |
| Learned > rules on easy domain_c | **Mixed** | Trees competitive; logreg (holdout) **worse** than rules; k-fold XGB *p*=0.022 |
| Cross-domain "plug and play" | **Not met** in general | C→B: test AUROC **0.45–0.50** (at/near **0.5** random baseline); partial exceptions (A→B logreg, A→C trees) |
| Statistical rigor | **Partial** | Strong where *n* = 70; holdout *n* = 20 underpowered on domain_a |
| Cost story | **Illustrative** | Scaled projections; with **e = 0.3**, `r(1−e)` spans ~**20–58%** over holdout **r** in §5.4 (**0.28–0.83**) |
| Feature interpretability | Met | key_overlap dominant on a/c; demo_labels on b; structurally meaningful |

---

## 10. Real-world deployment scenario and future work

This section bridges the offline evaluation presented above to a **production CI/CD** setting, drawing on the online-learning design in `docs/PROPOSAL_SUMMARY.md` §8 and practical engineering considerations surfaced during development.

### 10.1 Deployment architecture (Promptfoo / LangSmith integration)

A realistic deployment embeds CARTS as a **CI gate** between the developer's prompt change and the eval-suite run:

```
Developer pushes prompt PR
        │
        ▼
 ┌──────────────┐
 │ Change        │  diff(P, P') → typed changes Δ
 │ Classifier    │
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │ Impact        │  (Δ, test features) → predicted impacted set
 │ Predictor     │  uses latest trained model (pkl / ONNX)
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │ Sentinel      │  random sample from non-predicted pool (10%, min 5)
 │ Sampler       │
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │ Selective     │  run S = predicted ∪ sentinel on P'
 │ Runner        │  compare results to cached P baseline
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │ Safety Check  │  if any sentinel regresses → escalate to full rerun
 └──────┬───────┘
        │
        ▼
  CI pass / fail + logged observations
```

In **Promptfoo** or **LangSmith**, this would integrate by **filtering which test IDs run** (equivalent to the `select_tests` / `select_tests_learned` pipeline in this repo)—for example a CI step that writes a test list or wrapper that calls the harness with `sentinel_fraction` / `magnitude_threshold` matching `scripts/run_evaluation.py` defaults unless overridden. There is **no** Promptfoo plugin in-tree yet; see `docs/PROPOSAL_SUMMARY.md` integration pathway.

### 10.2 Online learning via predict–measure–retrain

The offline models evaluated above (LightGBM, XGBoost, logistic regression) are trained on a **fixed** historical dataset. In production, every CI run produces **new** labeled observations—each test we execute yields a fresh `(change_features, test_features, impacted_or_not)` triple. This enables an **online learning loop**:

1. **Predict:** The current model selects tests for the incoming PR.
2. **Measure:** The runner executes the selected set. Observed pass/fail outcomes become new training rows.
3. **Retrain:** Periodically (e.g. nightly or every *k* PRs), the model is retrained on the accumulated dataset—prior ground truth **plus** new observations.
4. **Deploy:** The updated model replaces the old one for the next CI run.

**Why this works.** Unlike deep-learning systems that need massive retraining, tree-based models (LightGBM / XGBoost) retrain in **seconds** on the data volumes involved (thousands to tens of thousands of rows). The retrain step can be a scheduled job or a post-merge hook.

**Partial observability.** A critical difference from standard supervised learning is that CARTS **only observes outcomes for tests it selects**—skipped tests produce no signal. Without mitigation, this creates a **self-reinforcing bias**: the model never learns about sensitivities it has already decided to ignore. Three mechanisms address this:

- **Sentinel sampling** acts as an **exploration arm** (analogous to epsilon-greedy in bandits). Sentinel tests are drawn from the **non-predicted** pool (`sample_sentinels`), so each run can surface impacted tests the predictor omitted. Stratified **sentinel catch rate** in §4.4 spans **0.00–0.91** by change type (e.g. **0.00** when the predictor already covers all impacted tests or sentinels rarely overlap misses; **0.83–0.91** on some **workflow** / **compound** / **policy** strata in these domains)—consistent with `sentinel_hit` in `src/phase2/evaluator.py` (impacted ∩ sentinel ≠ ∅).

- **Periodic full reruns.** Running the full suite on every *n*-th PR (e.g. every 10th–20th) provides **uncensored** calibration data. At a 90–95% reduction in full-rerun frequency, the incremental cost is modest. This also serves as a **model health check**: if recall on the full-rerun PR drops below a threshold, the team is alerted.

- **Uncertainty-driven selection (active learning).** Instead of purely random sentinels, the sampler can prioritize tests where the model's predicted probability is closest to the decision threshold—maximizing the information gained per call. This is a direct extension of the `threshold` parameter already stored per model in `comparison_summary.json`.

### 10.3 Cold start and bootstrap

A new team adopting CARTS has **no historical** `(change, test, outcome)` data. The bootstrap path mirrors the project's own phasing:

1. **Rule-only mode (day 1).** Deploy the rule-based selector from Phase 2. No training data needed; recall+s ranges from **0.62** (domain_b) to **0.94** (domain_c) with **0.37–0.47** call reduction from this evaluation. The team starts saving immediately, albeit conservatively.

2. **Accumulate labels (weeks 1–4).** Every CI run with the rule selector generates labeled rows. Sentinels and occasional full reruns (e.g. weekly) expand coverage.

3. **Switch to learned model (week 4+).** Once **~500–1000** labeled rows accumulate (roughly 5–10 full reruns × suite size, or more if most runs are selective), train the first LightGBM/XGBoost model. Our k-fold results (§5.6) show that even with modest data, tree models achieve **0.92–0.97** recall+s with **0.58–0.81** call reduction.

4. **Continuous improvement.** The predict–measure–retrain loop from §10.2 takes over. The model improves as the team's change patterns evolve.

### 10.4 Escalation policy and fail-safe guarantees

CARTS is designed to **degrade gracefully** to full-rerun behavior, never to silently miss regressions:

- **Sentinel hit → full rerun.** In this codebase, **`sentinel_hit`** means a ground-truth **impacted** test (outcome change vs. baseline) appeared among the sentinel IDs (`src/phase2/evaluator.py`). A production analogue: if any sentinel-evaluated test **fails the gate** vs. the stored baseline for `P`, treat that as a model miss and **rerun the full suite** on `P'`. The cost of escalation then matches a full-rerun policy, so the worst case is the status quo.

- **Model staleness detection.** If the fraction of PRs triggering sentinel escalation exceeds a configurable threshold (e.g. **20%**), the system can automatically fall back to full-rerun mode and alert the team that the model needs retraining or that a new change category has appeared.

- **High-risk change override.** Operationally, teams can force a full suite when the **change classifier** reports certain `unit_type` values (see `docs/PROPOSAL_SUMMARY.md` taxonomy). Our Phase 2 rule selector **already** selects all tests for **`role`** versions (**1.0** recall, **0.0** call reduction in §4.4), matching the "role is high-risk" policy. **`model_settings`** and "≥3 sections" overrides are **design options**—not hard-coded escalation paths in the current selector beyond the typed rules.

### 10.5 Multi-tenant and cross-domain considerations

Our cross-domain evaluation (§5.5) showed that models trained on one domain generally **do not transfer** to another without retraining (e.g. domain_c → domain_b: predictor recall **0.01–0.02**). This has a direct deployment implication:

- **Each application needs its own model.** A single CARTS model cannot serve unrelated prompt applications. The cold-start path (§10.3) must be repeated per application.

- **Combined training helps within related domains.** Our k-fold combined results (§5.6) show that pooling data from related applications improves stability (e.g. combined XGBoost on domain_c: **1.00** recall+s, **0.78** call reduction vs. single-domain **0.97** / **0.80**). Organizations with multiple **similar** prompt applications (e.g. several JSON extraction bots for different schemas) can benefit from shared training.

- **Feature-space alignment.** Cross-domain transfer fails because the feature distributions (change types, test-tag vocabularies, impact patterns) diverge. A production system could use **domain-agnostic features** (e.g. embedding similarities, magnitude percentiles) to improve transferability, though this remains future work.

### 10.6 Scaling to large suites

The cost projections in §8 assume suites of 100–500 tests. At larger scales:

- **Selection overhead.** The implementation scores one prediction per test (learned path) or runs the rule pipeline per version; complexity scales with suite size but **does not** issue LLM calls for selection itself. Overhead is **not benchmarked** in this repository; it is expected to be **small relative to** running the selected eval LLM calls. The break-even suite size of **≈2.4 tests** from `break_even_suite_size` in `src/phase4/cost_model.py` (§8) compares **time** overhead vs. saved call latency, not Python profiling.

- **Sentinel sizing.** Defaults are **10% of the non-predicted pool** with **min 5** (`sample_sentinels`). For huge suites, that fraction can still be large; the code already caps at the pool size. A lower cap or **adaptive** rate (production extension) would mirror the PROPOSAL's ablation grid (`_SENTINEL_FRACTIONS` in `src/phase2/evaluator.py`).

- **Parallelism.** Selected tests and sentinels can be dispatched in parallel (batch API, as used in Phase 1). Call reduction translates directly to wall-clock reduction when the bottleneck is API rate limits or per-call latency.

### 10.7 Future work beyond this evaluation

| Direction | Description | Building on |
|-----------|-------------|-------------|
| **Online retraining loop** | Implement the predict–measure–retrain cycle in a CI plugin | §10.2 design; Phase 3 models |
| **Active sentinel sampling** | Replace random sentinels with uncertainty-driven selection | §10.2 exploration; threshold data |
| **Real changelog evaluation** | Validate on production prompt changelogs (not synthetic mutations) | Phase 1 generator as scaffolding |
| **Multi-model evaluation** | Extend to model-version changes (GPT-4o → GPT-4o-mini) | `model_settings` change type |
| **Promptfoo plugin** | Package CARTS as a test-filter integration for Promptfoo or similar CI | §10.1; `docs/PROPOSAL_SUMMARY.md` |
| **Flaky test handling** | Detect and quarantine non-deterministic tests (e.g. multi-sample voting) | PROPOSAL ablation idea; baseline uses `temperature` from `config/settings.yaml` / cache keying in `src/phase1/harness/` — no majority-vote loop in-tree |
| **Embedding-based features** | Use semantic similarity between change diffs and test inputs as predictor features | `sentence-transformers` already in stack |

---

## 11. Limitations

- **Version-holdout** evaluation uses **20** versions; **Wilcoxon** on domain_a holdout is **underpowered** despite visible mean gains.
- **Cross-domain** results are **synthetic** domain shifts; production tenants may differ.
- **High recall+s** with **tiny** predictor recall indicates **sentinel-dependent** safety, not reliable impact localization.
- **Phase 4 ablations** (tables T4/T8) require a **full** analysis run without `--skip-ablations`.
- **Online learning** (§10.2) is a **design**, not yet implemented or empirically validated in this project.
- **Cost projections** use assumed inputs (r=0.9, e=0.3); real-world savings depend on actual call reduction achieved, which ranges from 0.28 (logreg, domain_b holdout) to 0.83 (LGBM, domain_c holdout).

---

## 12. Artifacts and reproduction

### 12.1 Generated figures

| Figure | Description |
|--------|-------------|
| `confusion_breakdown_domain_*.png` | Per-domain confusion matrix breakdown |
| `cost_savings.png` | Cost savings across scenarios |
| `cross_domain_heatmap.png` | Cross-domain recall transfer matrix |
| `feature_importance_domain_*.png` | Top LightGBM features per domain |
| `magnitude_scatter.png` | Impact magnitude vs. selection metrics |
| `pareto_front.png` | Recall vs. call reduction Pareto front |
| `recall_by_change.png` | Recall stratified by change type |

### 12.2 Generated tables

| Table | Description |
|-------|-------------|
| `T1_main_results` | Consolidated metrics for all 48 experiments |
| `T2_learned_vs_rule` | Learned vs. rule with CIs, p-values, effect sizes |
| `T3_feature_importance` | Top 15 features per domain |
| `T5_cross_domain` | 3×3 cross-domain recall matrix |
| `T6_cost_projections` | Cost scenario table |
| `T7_by_change_type` | Per-change-type metrics for all experiments (327 rows) |

### 12.3 Key output files

| Phase | Key outputs |
|-------|----------------|
| 1 | `results/baseline/ground_truth_domain_*.jsonl`, `summary_domain_*.json` |
| 2 | `results/selection/metrics_domain_*.json`, `sweep_domain_*.json`, `detail_domain_*.jsonl` |
| 3 | `models/*.pkl`, `eval_summary.json`, `comparison_summary.json`, `kfold_summary.json`, `training_summary.json`, `detail_eval_*.jsonl` |
| 4 | `main_results.json`, `metric_summaries.json`, `statistical_tests.json`, `cost_projections.json`, `cost_curves.json`, `feature_importance.json`, `cross_domain_matrix.json`, `by_change_type.json`, `detail_record_counts.json`, `figures/*.png`, `tables/*.tex` |

### 12.4 Reproduction commands

```bash
python -m scripts.run_batch_baseline --both
python -m scripts.run_evaluation --all-domains --sweep
python -m scripts.run_phase3_train train-all
python -m scripts.run_phase3_eval evaluate-all
python -m scripts.run_phase3_eval kfold-evaluate-all
python -m scripts.run_phase3_eval compare
python -m scripts.run_phase4_analysis full
```

CLI details match `scripts/run_batch_baseline.py` (single command `run`, invoked as `python -m scripts.run_batch_baseline --both`), `scripts/run_evaluation.py` (`evaluate` is the only command; Typer may show options at top level), and `scripts/run_phase3_eval.py`. For **combined** k-fold only, use `python -m scripts.run_phase3_eval kfold-evaluate-combined --target-domain <domain>`.

Use `python -m scripts.run_phase4_analysis full` **without** `--skip-ablations` if you need **T4/T8** ablation tables (slow).

---

*Numeric values reflect the committed pipeline definitions and the artifacts checked into `results/` at documentation time.*
