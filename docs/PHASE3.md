# Phase 3: Learned Selector

**Goal:** Replace the rule-based impact predictor from Phase 2 with learned models (**LightGBM**, **XGBoost**, **logistic regression**) trained on Phase 1 ground truth at **(version, test)** granularity (feature pipeline in `src/phase3/dataset.py`). Evaluate with a **version holdout** (train v01–v50, test v51–v70), **5-fold GroupKFold** (single-domain and **combined** multi-domain), and **cross-domain** transfer — reusing the Phase 2 evaluator for final metrics.

---

## 1. Problem Statement

### 1.1 Why Rules Are Not Enough

Phase 2's rule-based predictor maps change types to test sensitivity categories via a hand-written affinity table. This works for direct, same-section impacts (a format change breaking a schema test) but fundamentally cannot capture **cross-section ripple effects** — the dominant failure mode observed in Phase 2 evaluation:

Illustrative failure modes (see [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) for current numbers): **policy** and **demonstration** rows often have low **predictor-only** recall under rules; **workflow** can be sentinel-dependent on domain_b. Aggregate rule recall (+sentinel) is moderate on domain_a/b and high on domain_c in the reference run.

The fundamental issue is that an LLM is not modular: static rules miss cross-section effects that show up in the labels.

### 1.2 What a Learned Model Can Do

The tens of thousands of labeled **(version, test)** outcomes from Phase 1 encode cross-section patterns that rules miss. A model trained on this data can learn statistical regularities like:

- "When the policy section is modified and the test uses a `keyword_presence` monitor, impact probability is 0.72"
- "When a demonstration is deleted and the test checks for `code_execution`, impact probability is 0.45"
- "When a format key is renamed and the test's `inferred_key_deps` include that key, impact probability is 0.95"

These are not hand-coded heuristics — they are patterns learned from observed outcomes.

### 1.3 Design Constraint: No Oracle Leakage

The learned model must use only features that would be available in a real CI/CD deployment:

- The old prompt text and the new prompt text
- The eval suite metadata (test IDs, monitor types, tags, `sensitive_to`)
- The trained model weights (learned offline from historical data)

The model must **never** see ground truth labels at prediction time. Ground truth is used only during training (offline) and evaluation (post-hoc comparison).

---

## 2. What Phase 3 Produces

By the end of this phase we have:

1. A **feature extraction pipeline** — transforms `ClassifiedChange` and `TaggedTest` objects into a fixed-width numeric feature vector per (change, test) pair.
2. **Ablation sensitivity profiles** — per-test section sensitivity measured by running each test with each prompt section ablated. Stored in `data/{domain}/sensitivity_profiles.json`.
3. A **training dataset** — pandas DataFrame of feature rows + `impacted` label per domain/split (see `build_dataset` in `dataset.py`).
4. **Trained models** — `.pkl` artifacts under `models/` for **lightgbm**, **xgboost**, **logreg** (split tag `temporal` for version holdout, plus cross-domain and k-fold naming conventions per `run_phase3_train.py`).
5. A **learned predictor** — drop-in replacement for the rule-based `predict_impacted()` with the same `PredictionResult` interface.
6. A **learned selector** — composes the learned predictor with the existing classifier, tagger, and sentinel sampler.
7. **Evaluation results** — full comparison of learned vs. rule-based selectors across version holdout, all-data, and cross-domain protocols.
8. **Analysis artifacts** — feature importance rankings, threshold sweep curves, and ablation studies.

---

## 3. Phase 3 Inputs (from Phase 1 & Phase 2)

| Artifact | Path | Used For |
|---|---|---|
| Ground truth (domain_a) | `results/baseline/ground_truth_domain_a.jsonl` | Training labels: `impacted` field |
| Ground truth (all domains) | `results/baseline/ground_truth_domain_*.jsonl` | Training labels: `impacted` |
| Version prompts | `data/{domain}/versions/v*.yaml` | `prompt_text` for classifier |
| Eval suites | `data/{domain}/eval_suite.yaml` | Test metadata |
| Base prompts | `data/{domain}/base_prompt.txt` | Assembled via `load_base_prompt` / loader |
| Change classifier | `src/phase2/change_classifier.py` | Produces `ClassifiedChange` objects for feature extraction |
| Test tagger | `src/phase2/test_tagger.py` | Produces `TaggedTest` objects for feature extraction |
| Evaluator | `src/phase2/evaluator.py` | Reused for final evaluation of learned selector |
| Phase 2 metrics | `results/selection/metrics_{domain}.json` | Baseline comparison: rule-based predictor performance |

---

## 4. Project Structure Additions

```
src/
  phase3/
    __init__.py
    features.py              # Feature extraction: (ClassifiedChange, TaggedTest) → feature vector
    dataset.py               # Build labeled dataset from ground truth + features
    trainer.py               # Train LightGBM and logistic regression models
    learned_predictor.py     # Drop-in replacement for rule-based predict_impacted()
    learned_selector.py      # Compose learned predictor + classifier + tagger + sentinel
scripts/
  run_phases.py              # wrapper: phase3-train / phase3-eval → scripts below
  run_ablation.py            # Run ablation sensitivity profiling (LLM calls)
  run_phase3_train.py        # CLI: train models with configurable splits
  run_phase3_eval.py         # CLI: evaluate learned selector via Phase 2 evaluator
models/                      # Trained artifacts (examples; full set from train-all)
  lightgbm_domain_a_temporal.pkl
  xgboost_domain_b_cross_bc.pkl
  logreg_domain_c_temporal.pkl
  pca_domain_*.pkl           # embedding reduction per domain
data/
  domain_a/
    sensitivity_profiles.json # Ablation-based per-test sensitivity vectors
  domain_b/
    sensitivity_profiles.json
results/phase3/
    eval_domain_*_*.json                     # Per-run aggregate metrics (+ by_change_type, …)
    eval_summary.json                         # Flat list of runs
    comparison_summary.json                   # Thresholds + rule baselines slice
    kfold_*.json, kfold_combined_*.json
    training_summary.json
    detail_eval_*.jsonl
tests/
  test_features.py
  test_dataset.py
  test_learned_predictor.py
```

---

## 5. Feature Engineering

Each training example is a `(change, test)` pair with a binary label (`impacted`). Since a version can have multiple changes (compound mutations), each change is paired separately with each test. The feature vector is the concatenation of four groups.

### 5.1 Change Features

Derived from the `ClassifiedChange` object produced by the Phase 2 change classifier. These describe *what changed* in the prompt.

| Feature | Type | Source |
|---|---|---|
| `unit_type_format`, `unit_type_demonstration`, `unit_type_demo_item`, `unit_type_policy`, `unit_type_workflow`, `unit_type_role` | Binary (one-hot) | `change.unit_type` |
| `change_type_inserted`, `change_type_deleted`, `change_type_modified`, `change_type_reordered` | Binary (one-hot) | `change.change_type` |
| `magnitude` | Float [0, 1] | `change.magnitude` |
| `num_affected_keys` | Int | `len(classified_change.affected_keys)` |
| `num_affected_demo_labels` | Int | `len(classified_change.affected_demo_labels)` |
| `is_compound` | Binary | Whether the version has >1 distinct `unit_type` across all its changes |
| `change_embedding` | Float[384] → PCA[16-32] | `sentence-transformers/all-MiniLM-L6-v2` encoding of `change.content_diff` |

**Total: 15 scalar keys + reduced embedding** (see `extract_change_features` in `src/phase3/features.py`)

### 5.2 Test Features

Derived from the `TaggedTest` object produced by the Phase 2 test tagger. These describe *what the test checks*.

| Feature | Type | Source |
|---|---|---|
| `monitor_type_schema`, `monitor_type_required_keys`, `monitor_type_keyword_presence`, `monitor_type_policy`, `monitor_type_format`, `monitor_type_code_execution` | Binary (one-hot) | `tagged_test.monitor_type` |
| `sensitivity_category_format`, `sensitivity_category_policy`, `sensitivity_category_demo`, `sensitivity_category_workflow`, `sensitivity_category_general` | Binary (one-hot) | `tagged_test.sensitivity_category` |
| `num_inferred_key_deps` | Int | `len(tagged_test.inferred_key_deps)` |
| `num_inferred_demo_deps` | Int | `len(tagged_test.inferred_demo_deps)` |
| `has_sensitive_to_format` | Binary | `"format" in tagged_test.sensitive_to` |
| `has_sensitive_to_demonstration` | Binary | `"demonstration" in tagged_test.sensitive_to` |
| `has_sensitive_to_policy` | Binary | `"policy" in tagged_test.sensitive_to` |
| `has_sensitive_to_workflow` | Binary | `"workflow" in tagged_test.sensitive_to` |
| `test_input_embedding` | Float[384] → PCA[16-32] | `all-MiniLM-L6-v2` encoding of `tagged_test.input_text` |

**Total: 15 scalar + reduced embedding**

### 5.3 Pairwise Features

Computed for each specific (change, test) pair. These capture the *relationship* between a change and a test.

| Feature | Type | Computation |
|---|---|---|
| `cosine_similarity` | Float [-1, 1] | Cosine similarity between `change_embedding` and `test_input_embedding` (before PCA) |
| `key_overlap` | Binary | `bool(set(affected_keys) & set(inferred_key_deps))` |
| `demo_overlap` | Binary | `bool(set(affected_demo_labels) & set(inferred_demo_deps))` |
| `section_match` | Binary | `change.unit_type` matches `sensitivity_category` mapping (e.g., format→format, policy→policy, demonstration→demo, demo_item→demo) |
| `rule_based_prediction` | Binary | Whether Phase 2's rule-based predictor would have selected this test for this change |

**Total: 5 scalar**

### 5.4 Ablation Sensitivity Profiles

Measured once per test by running the test with each prompt section individually replaced by a generic placeholder (e.g., `"[This section has been removed.]"`). If the test outcome flips (pass→fail or fail→pass), the test is sensitive to that section.

| Feature | Type | Measurement |
|---|---|---|
| `ablation_sensitive_role` | Binary | Outcome flips when `<role>` section is ablated |
| `ablation_sensitive_format` | Binary | Outcome flips when `<format>` section is ablated |
| `ablation_sensitive_demonstrations` | Binary | Outcome flips when `<demonstrations>` section is ablated |
| `ablation_sensitive_policy` | Binary | Outcome flips when `<policy>` section is ablated |
| `ablation_sensitive_workflow` | Binary | Outcome flips when `<workflow>` section is ablated |

**Total: 5 binary**

**Cost:** 5 sections × 100 tests = 500 LLM calls for domain_a, 5 × 80 = 400 for domain_b. At $0.001/call (gpt-4o-mini), total cost is ~$0.90. Run once, cached forever.

**Procedure:**
1. Load the base prompt and parse into sections.
2. For each section, create an ablated prompt by replacing that section's content with a placeholder.
3. Run each test on the ablated prompt.
4. Compare the outcome to the baseline outcome (from Phase 1 cache): if different, the test is sensitive to that section.
5. Store as `data/{domain}/sensitivity_profiles.json`.

### 5.5 Feature Summary

| Group | Scalar Features | Embedding Features | Total (after PCA) |
|---|---|---|---|
| Change | 15 | 384 → 16-32 | 31-47 |
| Test | 22 (incl. ablation flags when present) | 384 → 16-32 | varies |
| Pairwise | 5 | — | 5 |
| Ablation | 5 | — | 5 |
| **Total (typical)** | **42+** | **768 → PCA** | **see `features.py`** |

---

## 6. Model Architecture

### 6.1 LightGBM (Primary Model)

Gradient-boosted decision trees are well-suited for this task because:
- They handle mixed feature types (binary, integer, float) natively
- They are robust to feature scale differences
- They provide built-in feature importance
- They train in seconds on **10⁴–10⁵** rows per configuration

**Configuration:**

```
objective: binary
metric: binary_logloss
scale_pos_weight: ~9.0  (computed as n_negative / n_positive per training set)
n_estimators: 100-500   (tuned)
max_depth: 4-8           (tuned)
learning_rate: 0.01-0.1  (tuned)
min_child_samples: 5-20  (tuned)
feature_fraction: 0.8
bagging_fraction: 0.8
bagging_freq: 5
```

Hyperparameter tuning via Optuna with 50 trials, optimizing for recall at a fixed call-reduction floor of 30%.

### 6.2 XGBoost

Second tree ensemble (`xgboost`); same probability interface as LightGBM. Trained and evaluated in parallel with LightGBM (`run_phase3_train.py` / `run_phase3_eval.py`).

### 6.3 Logistic Regression (Interpretable Baseline)

Provides a linear baseline and interpretable feature coefficients. Useful for understanding which features drive predictions.

**Configuration:**

```
penalty: l2
C: tuned via 5-fold cross-validation on training set (grid: 0.01, 0.1, 1.0, 10.0)
class_weight: 'balanced'
max_iter: 1000
solver: lbfgs
```

All features are standardized (zero mean, unit variance) before training. Embeddings are reduced to 8-16 PCA components for the logistic regression model (fewer than LightGBM, since linear models are more sensitive to high dimensionality).

### 6.4 Embedding Handling

Raw `all-MiniLM-L6-v2` embeddings are 384-dimensional. Feeding them directly to tree models wastes splits on noisy dimensions.

**Strategy:**
1. Compute embeddings for all change `content_diff` texts and all test `input_text` texts.
2. Fit PCA on the training set's embeddings.
3. Reduce to 16-32 components (capturing ~90% variance; exact number determined by explained variance ratio).
4. Concatenate reduced embeddings with scalar features.
5. PCA transformer is saved alongside the model for inference.

### 6.5 Output

Both models output `P(impacted | features)` — a probability between 0 and 1. The selector converts this to a binary prediction using a threshold `τ`:

```
predicted_impacted = { test_id : P(impacted) >= τ }
```

Lower `τ` → more tests selected → higher recall, lower call reduction (conservative).
Higher `τ` → fewer tests selected → lower recall, higher call reduction (aggressive).

---

## 7. Evaluation Strategy

### 7.1 Version holdout (Primary — deployment-style)

**Rationale:** In a real deployment, the model trains on historical prompt-version data and is scored on **new** versions. The benchmark uses a **fixed version-range** holdout (v01–v50 train, v51–v70 test): it is **not** wall-clock temporal ordering, but it mimics “train on an earlier block, test on a later block.”

**Protocol:**
- **Train:** v01–v50
  - domain_a: 50 × 100 = **5,000** rows; domain_b: 50 × 80 = **4,000**; domain_c: 50 × 90 = **4,500**
- **Test:** v51–v70 (20 versions each)
  - domain_a: **2,000** rows; domain_b: **1,600**; domain_c: **1,800**

This is a deliberately hard test: the LLM-generated mutations are more realistic, more diverse, and stylistically different from the mechanical mutations used for training. The model must generalize from surgical string replacements to natural, multi-word rewrites.

**Threshold tuning:** Use 20% of the training versions (v41–v50) as a validation set for threshold selection, then report final metrics on the test set (v51–v70).

### 7.2 All-Data Training (Ceiling Performance)

**Rationale:** Shows the upper bound of model performance — how well it can fit this dataset with maximum information.

**Protocol:**
- **Train / eval:** All **70** versions per domain, or **combined** training on pooled domains (k-fold / combined modes in `run_phase3_eval`)
- **Evaluate:** Held-out folds or full-domain metrics per CLI

Overfitting risk is acknowledged. This is **not** the primary evaluation — it serves to answer: "Is the feature space rich enough to capture the impact patterns?" If all-data performance is low, the features need improvement. If it is high but version-holdout performance is low, the model is not generalizing.

**Threshold tuning:** 5-fold cross-validation by version (each fold holds out ~14 versions) to select threshold, then report metrics on the full data.

### 7.3 Cross-Domain Generalization

**Rationale:** Tests transfer across **three** scenario families. Full matrix: train on one domain, evaluate on **all 70 versions** of another (tags like `lightgbm_domain_b_cross_ba` = trained B, tested A).

**Examples:**

| Direction | Question |
|---|---|
| A→B, A→C | Does domain_a structure help on b/c? |
| B→A, B→C | Coding-assistant signal vs. other suites |
| C→A, C→B | Hardest directions in the reference run (see `EVALUATION_RESULTS.md`) |

**Combined k-fold:** Pool **all three** domains’ versions for training, still report metrics on a **named target** domain.

**Threshold tuning:** On the version-holdout validation slice for models trained under `--split temporal`; cross-domain uses thresholds from `comparison_summary.json` conventions.

### 7.4 Metrics

All metrics are computed using the Phase 2 evaluator framework, ensuring exact comparability with rule-based results.

| Metric | Formula | Description |
|---|---|---|
| **Recall (predictor only)** | `\|I ∩ predicted\| / \|I\|` | Fraction of truly impacted tests caught by the model alone |
| **Recall (with sentinel)** | `\|I ∩ selected\| / \|I\|` | Fraction caught by model + sentinel combined |
| **Call reduction** | `1 - \|selected\| / \|T\|` | Fraction of tests skipped |
| **False omission rate** | `\|I ∩ (T \ S)\| / \|T \ S\|` | Among **skipped** tests, fraction that were impacted (`src/phase2/evaluator.py`) |
| **Sentinel catch rate** | `(# versions where sentinel caught a miss) / (# versions with any miss)` | How often the safety net fires |
| **AUROC** | Area under receiver operating characteristic | Model discrimination quality (threshold-independent) |
| **AUPRC** | Area under precision-recall curve | Better metric under class imbalance |
| **Effective recall** | Version-level roll-up of Phase 2 semantics | If `sentinel_hit`, **1.0**; else recall (+sentinel) |
| **Effective call reduction** | Version-level roll-up | If `sentinel_hit`, **0**; else call reduction |

Phase 3 eval JSON mirrors Phase 2 (`mean_effective_recall`, `mean_effective_call_reduction` in aggregates; per-version fields in detail JSONL).

---

## 8. Threshold Tuning

The classification threshold `τ` is the key hyperparameter controlling the recall-reduction tradeoff.

**Procedure:**
1. After model training, compute predicted probabilities for all examples in the validation set.
2. Sweep `τ` from 0.01 to 0.99 in steps of 0.01.
3. At each threshold, compute recall (predictor only) and call reduction across all versions.
4. Select the operating point: the highest `τ` (maximum call reduction) where recall >= 0.95 on the validation set.
5. Report the full Pareto curve: recall vs. call reduction for all thresholds.

**Ablation table:** Report results at fixed thresholds (0.05, 0.10, 0.20, 0.30, 0.50) to show the tradeoff explicitly.

**Safety margin:** Since the threshold is tuned on training/validation data, apply a small safety buffer when deploying to test data — e.g., if the selected threshold is 0.25, use 0.20 for final evaluation to account for distribution shift.

---

## 9. Integration

### 9.1 Architecture

The learned model replaces only the impact predictor. All other pipeline components are reused unchanged.

```
Existing (reused):                New (Phase 3):
  classify_changes(P, P')   ──→   feature_extraction(changes, tagged_tests)
  tag_tests(test_cases)     ──→         │
                                        ▼
                                  learned_model.predict_proba(features)
                                        │
                                        ▼
                                  threshold_gate(proba, τ) → predicted_ids
                                        │
                                        ▼
  sample_sentinels()         ──→  selected = predicted ∪ sentinels
                                        │
                                        ▼
                                  SelectionResult
```

### 9.2 `src/phase3/learned_predictor.py`

Same interface as Phase 2's `predict_impacted()`:

```python
def predict_impacted(
    changes: list[ClassifiedChange],
    tagged_tests: list[TaggedTest],
    *,
    model_path: str | Path,
    pca_path: str | Path,
    threshold: float = 0.20,
    magnitude_threshold: float = 0.005,
) -> PredictionResult:
    """Learned impact predictor — drop-in replacement for rule-based version.

    1. Extract features for each (change, test) pair.
    2. Load trained model and PCA transformer.
    3. Predict P(impacted) for each pair.
    4. Apply threshold to produce predicted set.
    5. Return PredictionResult with predicted_ids and reasons.
    """
```

### 9.3 `src/phase3/learned_selector.py`

Thin wrapper around the existing `select_tests()` flow, replacing the predictor:

```python
def select_tests_learned(
    old_prompt: str,
    new_prompt: str,
    test_cases: list[TestCase],
    *,
    model_path: str | Path,
    pca_path: str | Path,
    threshold: float = 0.20,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int | None = None,
    sensitivity_profiles: dict | None = None,
) -> SelectionResult:
```

---

## 10. Implementation Order (Sprints)

### Sprint 1: Feature Engineering + Ablation Profiling

**Goal:** Build the feature extraction pipeline and generate per-test sensitivity profiles via LLM ablation runs.

**Tasks:**

- [ ] Create `src/phase3/__init__.py`
- [ ] Implement `src/phase3/features.py`
  - `extract_change_features(change: ClassifiedChange, is_compound: bool) -> dict`
  - `extract_test_features(test: TaggedTest, sensitivity: dict | None) -> dict`
  - `extract_pairwise_features(change: ClassifiedChange, test: TaggedTest, change_emb, test_emb, rule_pred: bool) -> dict`
  - `build_feature_row(change, test, ...) -> dict` (combines all groups)
  - Embedding computation via `sentence-transformers`
- [ ] Implement `scripts/run_ablation.py`
  - For each domain: load base prompt, parse sections, for each section create ablated prompt, run all tests, compare to baseline outcome
  - Use the existing eval harness (`src/phase1/harness/runner.py`) and cache
  - Output: `data/{domain}/sensitivity_profiles.json` — `{test_id: {role: bool, format: bool, demonstrations: bool, policy: bool, workflow: bool}}`
  - Progress bar via tqdm; estimated ~900 LLM calls total
- [ ] Implement `src/phase3/dataset.py`
  - `build_dataset(domain, *, split)` → loads ground truth, runs classifier + tagger on each version, extracts features, returns `(X: DataFrame, y: Series, metadata: DataFrame)`
  - Handle multi-change versions: for compound changes, create one row per (change, test) pair; label is `impacted` for the *version* (since impact is at version level, not change level)
  - Alternatively: aggregate change features per version (max magnitude, union of unit_types) for a version-level predictor — implement both and compare
- [ ] Write `tests/test_features.py`
  - Test: one-hot encoding produces correct columns
  - Test: pairwise features compute correctly for known inputs
  - Test: embedding dimension is 384
  - Test: feature row has expected number of columns

**Validation:** Feature extraction runs on **all versions for all domains** without errors; row count ≈ **70 × suite_size** per domain for `split=all`.

---

### Sprint 2: Model Training + Threshold Tuning

**Goal:** Train LightGBM and logistic regression models, tune thresholds, and evaluate on all three protocols.

**Tasks:**

- [ ] Implement `src/phase3/trainer.py`
  - `train_lightgbm(X_train, y_train, *, tune: bool = True) -> lgb.Booster`
  - `train_logreg(X_train, y_train) -> LogisticRegression`
  - `fit_pca(embeddings, *, n_components: int) -> PCA`
  - `tune_threshold(y_true, y_proba, *, min_recall: float = 0.95) -> float`
  - `save_model(model, pca, threshold, path)` and `load_model(path)`
  - Optuna integration for LightGBM hyperparameter search (50 trials)
- [ ] Implement `scripts/run_phase3_train.py` (CLI via typer)
  - `--domain domain_a|domain_b|combined`
  - `--split temporal|all|cross_ab|cross_ba`
  - `--model lightgbm|logreg|both`
  - Trains model, tunes threshold, saves to `models/`, prints metrics
- [ ] Run training for all configurations:
  - Per-domain version holdout (`--split temporal`): 4 models (2 domains × 2 model types)
  - Per-domain all-data: 4 models
  - Combined all-data: 2 models
  - Cross-domain: 4 models (A→B, B→A × 2 model types)
  - Total: 14 trained model configurations
- [ ] Threshold sweep and Pareto analysis
  - For each model: sweep τ from 0.01 to 0.99, record recall and call_reduction
  - Output: `results/phase3/threshold_sweep_{domain}.json`
- [ ] Feature importance analysis
  - LightGBM: `model.feature_importance(importance_type='gain')`
  - Logistic regression: `model.coef_` absolute values
  - Output: `results/phase3/feature_importance_{domain}.json`

**Validation:**
- All models train without errors
- Version-holdout LightGBM recall (predictor only) ≥ 0.85 on at least one domain
- Threshold sweep shows a clear recall-reduction tradeoff curve
- Feature importance ranks pairwise features (section_match, key_overlap, rule_based_prediction) among the top 10

---

### Sprint 3: Integration + Final Evaluation

**Goal:** Integrate the trained model into the selector pipeline and run the full evaluation comparison.

**Tasks:**

- [ ] Implement `src/phase3/learned_predictor.py`
  - Same `PredictionResult` output as Phase 2's predictor
  - Loads model from pickle, applies PCA, predicts, thresholds
  - Returns reasons dict with predicted probability per test
- [ ] Implement `src/phase3/learned_selector.py`
  - `select_tests_learned(old_prompt, new_prompt, test_cases, *, model_path, ...)` → `SelectionResult`
  - Composes: classify → tag → extract features → predict → sentinel → combine
- [ ] Implement `scripts/run_phase3_eval.py` (CLI via typer)
  - Runs learned selector on all versions using Phase 2's evaluator framework
  - Computes all metrics (recall, call_reduction, false_omission_rate, sentinel_catch_rate)
  - Aggregates by change_type, magnitude_bucket, mutation_category
  - Compares against Phase 2 rule-based results and baselines
  - Output: `results/phase3/metrics_{domain}_{model}_{split}.json`
- [ ] Run full evaluation for all configurations
- [ ] Build comparison summary
  - Side-by-side table: rule-based vs. LightGBM vs. logistic regression
  - Breakdown by change type, magnitude, and mutation category
  - Output: `results/phase3/comparison_summary.json`
- [ ] Ablation studies:
  - Feature group ablation: train with each group removed (no change features, no test features, no pairwise features, no ablation features) to measure contribution
  - Sentinel interaction: measure recall improvement from sentinel at different model thresholds
  - Threshold sensitivity: report metrics at τ = {0.05, 0.10, 0.20, 0.30, 0.50}
- [ ] Write `tests/test_learned_predictor.py`
  - Test: predictor returns PredictionResult with correct fields
  - Test: high-probability predictions are included, low-probability excluded
  - Test: threshold parameter affects selection size
  - Test: output is deterministic given same inputs

**Validation:** Results match expectations:
- Learned model outperforms rule-based on recall while maintaining comparable call reduction
- LightGBM outperforms logistic regression
- Version-holdout metrics are lower than all-data (expected — harder task)
- Cross-domain recall is lower than within-domain (expected — hardest task)

---

## 11. Success Criteria

| Criterion | Target | Notes |
|---|---|---|
| Recall (predictor only) on version holdout | ≥ 0.85 | Learned model catches 85%+ of impacted tests on unseen versions |
| Recall (with sentinel) on version holdout | ≥ 0.90 | Adding sentinels brings total recall to 90%+ |
| Call reduction on version holdout | ≥ 30% | We still skip at least 30% of tests |
| Improvement over rule-based | ≥ +15pp recall | Learned model adds ≥15 percentage points of recall over Phase 2 |
| Cross-domain recall | ≥ 0.75 | Patterns transfer across domains with reasonable quality |
| All-data recall (ceiling) | ≥ 0.95 | Feature space is rich enough when data is maximized |
| No oracle leakage | Verified | Model uses only features available in production deployment |
| Feature importance coherence | Validated | Top features are interpretable (section_match, key_overlap, cosine_sim, ablation sensitivity) |
| No errors | 0 | Training, prediction, and evaluation complete for all configurations without crashes |

---

## 12. Handling Compound Changes

A key design decision is how to handle versions with multiple changes (e.g., v01 has a format change + 8 demo_item changes + a policy change).

**Approach: Version-Level Aggregation**

Since the ground truth label (`impacted`) is at the version level (test `t_i` was/wasn't impacted by the overall version change, not by any specific sub-change), we aggregate change features per version:

1. **Per-change features:** Compute features for each individual change.
2. **Aggregation:** For each version, aggregate across changes:
   - `unit_type` one-hots: take the max (OR) across changes (1 if any change is of that type)
   - `change_type` one-hots: take the max (OR)
   - `magnitude`: take the max across changes
   - `num_affected_keys`: sum across changes
   - `num_affected_demo_labels`: sum across changes
   - `is_compound`: 1 if >1 distinct unit_type
   - `change_embedding`: take the embedding of the highest-magnitude change (or the mean)
3. **Pairwise features:** Compute for each (change, test) pair, then take the max across changes:
   - `key_overlap`: 1 if any change has key overlap with this test
   - `demo_overlap`: 1 if any change has demo overlap
   - `section_match`: 1 if any change matches the test's sensitivity category
   - `rule_based_prediction`: 1 if the rule-based predictor would select this test
   - `cosine_similarity`: max across changes

This produces one feature row per **(version, test)** pair, aligned with Phase 1 ground truth labels for that version.

---

## 13. What Phase 3 Feeds Into

- **Phase 4** (Evaluation & Paper) presents the learned selector as the main contribution, with the rule-based selector as the baseline. The evaluator framework, metrics, and comparison infrastructure are all reused.
- The learned model's feature importance analysis directly supports the paper's discussion of which change and test attributes drive impact prediction.
- The cross-domain evaluation provides evidence for generalization claims.
- The threshold sweep provides the recall-reduction Pareto front that is the main figure in the paper.
- The ablation studies provide evidence for the value of each feature group, particularly the ablation sensitivity profiles.
