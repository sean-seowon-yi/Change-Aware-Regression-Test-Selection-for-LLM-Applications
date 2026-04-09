# Phase 2: Change Classifier & Rule-Based Selector

**Goal:** Build the change classifier that diffs two prompts into typed changes, a rule-based impact predictor that maps those changes to predicted test subsets, a sentinel sampler for safety, and an evaluation framework that measures recall, call reduction, and false omission rate against the Phase 1 ground truth.

---

## 1. What Phase 2 Produces

By the end of this phase we have:

1. A **change classifier** — given two prompt versions `P` and `P'`, it parses them into labeled sections, diffs them, and outputs a list of typed `Change` objects with unit_type, change_type, magnitude, and content_diff.
2. A **rule-based test tagger** — enriches each test case with inferred sensitivity tags derived from its monitor type, monitor config, and existing metadata.
3. A **rule-based impact predictor** — an affinity table that maps change types to predicted test tags, producing a predicted impact set for any given change.
4. A **sentinel sampler** — draws a random sample from tests *not* in the predicted set, providing a safety net against model blind spots.
5. A **selector** — composes the impact predictor and sentinel sampler to produce the final selected set `S(Δ) = predicted ∪ sentinel`.
6. An **evaluator** — runs the selector against **every version** in each domain (e.g. **70 × 3** in the full benchmark), compares `S(Δ)` to the true impact set `I(Δ)`, and reports recall, call reduction, false omission rate (among **non-selected** tests), and sentinel catch rate.
7. **Evaluation results** — metrics broken down by domain, change type, magnitude bucket, and category, plus baselines (full rerun, random 50%, monitor-type heuristic) and optional **parameter sweep** JSON (`sweep_domain_*.json`).

Phase 3 replaces the rule-based impact predictor (item 3) with a learned model trained on the same ground truth. Everything else stays the same.

---

## 2. Phase 2 Inputs (from Phase 1)

| Artifact | Path | Content |
|---|---|---|
| Ground truth (per domain) | `results/baseline/ground_truth_domain_*.jsonl` | One line per (version, test): outcomes vs baseline, `impacted`, change metadata |
| Summary (per domain) | `results/baseline/summary_domain_*.json` | Aggregate stats, cost, wall time, `impact_summary` |
| Version metadata | `data/{domain}/versions/v*.yaml` | **70** per domain: `prompt_text`, `changes`, parent link |
| Eval suites | `data/{domain}/eval_suite.yaml` | **100 / 80 / 90** tests for domain_a / b / c |
| Base prompts | `data/{domain}/base_prompt.txt` | Assembled base prompt text |
| Prompt parser | `src/phase1/prompts/parser.py` | `parse_prompt()`, `diff_sections()` |
| Data models | `src/phase1/models.py` | Shared dataclasses |
| Edit distance | `src/phase1/prompts/mutator.py` | `_token_edit_distance()` |

---

## 3. Project Structure Additions

```
src/
  phase2/
    __init__.py
    change_classifier.py     # diff(P, P') → list[Change]
    test_tagger.py           # enrich tests with inferred sensitivity tags
    impact_predictor.py      # rule-based affinity table: changes → predicted test set
    sentinel.py              # random sample from non-predicted tests
    selector.py              # compose predictor + sentinel → SelectionResult
    evaluator.py             # evaluate selector against ground truth
scripts/
  run_evaluation.py          # Typer: Phase 2 eval vs ground truth (--all-domains, --sweep)
  run_selection.py           # Ad hoc: run rule selector on chosen versions
results/
  selection/
    metrics_domain_{a,b,c}.json
    detail_domain_*.jsonl
    sweep_domain_*.json        # sentinel × magnitude grid (when --sweep)
tests/
  test_change_classifier.py
  test_impact_predictor.py
  test_selector.py
```

---

## 4. Pipeline Overview

```
                    OFFLINE (once per suite)            PER-CHANGE (on each PR)
                    ________________________            ______________________________

                    ┌─────────────────────┐
                    │   Test Tagger        │
                    │  eval_suite.yaml     │──→  tagged_tests
                    │  + rule enrichment   │       (cached)
                    └─────────────────────┘
                                                  ┌──────────────────────┐
                                                  │  Change Classifier   │
                                            P ──→ │  parse + diff +      │──→  Δ = list[Change]
                                            P'──→ │  magnitude + typing  │
                                                  └──────────────────────┘
                                                              │
                                                              ▼
                                                  ┌──────────────────────┐
                                                  │  Impact Predictor    │
                                            Δ ──→ │  affinity table      │──→  predicted_ids
                                  tagged_tests ──→│  + key matching      │
                                                  └──────────────────────┘
                                                              │
                                                              ▼
                                                  ┌──────────────────────┐
                                                  │  Sentinel Sampler    │
                                   all_test_ids ──→│  max(5, ⌈10%⌉) of   │──→  sentinel_ids
                                 predicted_ids ──→│  non-predicted pool  │
                                                  └──────────────────────┘
                                                              │
                                                              ▼
                                                  ┌──────────────────────┐
                                                  │  Selector            │
                                                  │  S = predicted ∪     │──→  SelectionResult
                                                  │      sentinel        │
                                                  └──────────────────────┘
                                                              │
                                                              ▼
                                                  ┌──────────────────────┐
                                                  │  Runner              │
                                                  │  run S on P'         │──→  results
                                                  │  compare to P cache  │
                                                  └──────────────────────┘
                                                              │
                                                              ▼
                                                  ┌──────────────────────┐
                                                  │  Safety Check        │
                                                  │  if sentinel catches │──→  escalate to
                                                  │  regression → alarm  │     full rerun
                                                  └──────────────────────┘
```

In evaluation mode, we skip the Runner and Safety Check — instead we compare `S(Δ)` directly to the ground truth `I(Δ)` from Phase 1.

---

## 5. Component-by-Component Implementation

### 5.1 Change Classifier (`src/phase2/change_classifier.py`)

**Purpose:** Given two prompt texts, produce a list of typed `Change` objects that describe what was modified, at what granularity, and with what magnitude.

**Reuses from Phase 1:**
- `parse_prompt()` from `src/phase1/prompts/parser.py` — splits a prompt into `{role, format, demonstrations, policy, workflow}` sections
- `diff_sections()` from `src/phase1/prompts/parser.py` — classifies each section as inserted, deleted, modified, or unchanged
- `_token_edit_distance()` from `src/phase1/prompts/mutator.py` — normalized Levenshtein on whitespace tokens
- `_split_demos()` from `src/phase1/prompts/mutator.py` — splits demonstration text into individual examples

**New logic:**

1. **Section-level diffing** — For each section classified as `modified` by `diff_sections`, compute magnitude using `_token_edit_distance`. Generate a content_diff summary (first 100 chars of the changed text).

2. **Demonstration item-level diffing** — If the `demonstrations` section is modified:
   - Split both old and new into individual examples using `_split_demos()`
   - Match examples by header label (e.g., "billing", "refund") using fuzzy matching
   - Classify each example as: unchanged, modified, inserted, deleted, reordered
   - Emit one `Change` per affected example with `unit_type="demo_item"` or `unit_type="demonstration"` for whole-section changes (add/remove)

3. **Reorder detection** — If a section's items appear in a different order but the content is identical, emit `change_type="reordered"` instead of `"modified"`.

4. **Key-level detection for format changes** — When the format section is modified, scan the diff for renamed/added/removed JSON key names (regex for `"key_name":`). Record detected key changes in the `content_diff` field for downstream key-level matching.

```python
@dataclass
class ClassifiedChange:
    """Extended Change with additional classifier metadata."""
    change: Change
    affected_keys: list[str]        # JSON keys involved (for format changes)
    affected_demo_labels: list[str] # demo labels involved (for demo changes)

def classify_changes(
    old_prompt: str,
    new_prompt: str,
) -> list[ClassifiedChange]:
    """Diff two prompts and return typed, classified changes."""
    ...
```

**Edge cases:**
- If both prompts are identical → return empty list
- If a section exists in one but not the other → `inserted` or `deleted`
- If the prompt doesn't use XML tags → fall back to whole-prompt string diff with `unit_type="unknown"`

---

### 5.2 Test Tagger (`src/phase2/test_tagger.py`)

**Purpose:** Enrich each test case with inferred sensitivity information, combining the existing `tags` and `sensitive_to` fields with rules derived from the test's monitor type and config.

**Why this is needed:** The existing `sensitive_to` field in `eval_suite.yaml` was hand-curated and is correct but coarse. The tagger adds finer-grained signals:
- Which specific JSON keys a test checks for
- Whether a test is sensitive to demonstration ordering vs. content
- Whether a test probes safety/policy compliance

```python
@dataclass
class TaggedTest:
    test_id: str
    input_text: str
    monitor_type: str
    tags: list[str]
    sensitive_to: list[str]
    inferred_key_deps: list[str]     # JSON keys this test checks
    inferred_demo_deps: list[str]    # demo labels this test exercises
    sensitivity_category: str         # "format" | "policy" | "demo" | "workflow" | "role" | "general"
```

**Tagging rules:**

| Monitor Type | Inferred Sensitivity | Additional Extraction |
|---|---|---|
| `schema` | `format`, `schema_sensitive` | Extract required keys from `json_schema.required` |
| `required_keys` | `format`, `schema_sensitive` | Extract keys from `monitor_config.keys` |
| `keyword_presence` | Depends on keywords: if keywords are JSON key names → `format`; if keywords are domain terms → `demonstration`; if keywords are refusal phrases → `policy` | Extract keywords from `must_contain` and `must_not_contain` |
| `policy` | `policy`, `safety_sensitive` | — |
| `regex` | Varies by pattern; default `format` | — |
| `format` | `format`, `format_sensitive` | — |
| `code_execution` | `format`, `workflow`, `demonstration` | — |

**Key dependency extraction (domain_a specific):**
- For `schema` monitors: parse `json_schema.properties` to get the set of keys the test validates (e.g., `["action", "priority", "customer_id", "summary", "metadata"]`).
- For `required_keys` monitors: read `monitor_config.keys` directly.
- For `keyword_presence` monitors: if any keyword in `must_contain` matches a known JSON key name (action, priority, customer_id, etc.), add that key to `inferred_key_deps`.

**Demo dependency extraction:**
- If a test's `sensitive_to` includes `"demo:exN"`, extract the label from the corresponding example.
- If a test's `tags` include a domain label (e.g., "billing", "refund"), map it to the demonstration example that covers that domain.

---

### 5.3 Impact Predictor (`src/phase2/impact_predictor.py`)

**Purpose:** Given a list of changes `Δ` and a list of tagged tests, predict which tests are likely impacted. This is the core of the rule-based selector.

**Affinity table:**

| Change `unit_type` | Select tests where... |
|---|---|
| `format` | `sensitivity_category == "format"` OR `inferred_key_deps` overlaps with `affected_keys` from the change |
| `demonstration` | `sensitivity_category == "demo"` OR `inferred_demo_deps` overlaps with `affected_demo_labels` |
| `demo_item` | `inferred_demo_deps` overlaps with `affected_demo_labels` from the specific edited example |
| `policy` | `sensitivity_category == "policy"` OR `"safety_sensitive" in tags` |
| `workflow` | `sensitivity_category == "workflow"` OR `"workflow_sensitive" in tags` OR `"workflow" in sensitive_to` |
| `role` | **All tests** — role changes are high-risk, low-frequency; always run everything |
| `tool_description` | Tests with `"tool" in tags` (not applicable for current domains, future-proofing) |
| `model_settings` | **All tests** — model changes can affect anything |

**Magnitude gating:**
- If a change has `magnitude < 0.005`, skip it (treat as no-op). This handles trivial whitespace or punctuation changes.
- The threshold is configurable and can be tuned in Phase 4's ablation study.

**Key-level matching (format changes):**
- When the classifier detects that a format change involves specific keys (e.g., renamed `"action"` to `"task_type"`), the predictor selects not just all format-sensitive tests but specifically tests whose `inferred_key_deps` includes the affected key.
- If the classifier cannot identify specific keys (e.g., a structural rewrite), fall back to selecting all format-sensitive tests.

**Compound changes:**
- When `Δ` contains multiple changes, the predicted set is the **union** of all individual predictions. This ensures conservatism — no impacted test is missed because one change masked another.

```python
def predict_impacted(
    changes: list[ClassifiedChange],
    tagged_tests: list[TaggedTest],
    *,
    magnitude_threshold: float = 0.005,
) -> PredictionResult:
    """Predict which tests are impacted by the given changes.

    Returns PredictionResult with predicted test IDs and reasoning.
    """
    ...

@dataclass
class PredictionResult:
    predicted_ids: set[str]
    reasons: dict[str, list[str]]  # test_id → list of reasons it was selected
```

---

### 5.4 Sentinel Sampler (`src/phase2/sentinel.py`)

**Purpose:** From tests NOT in the predicted impact set, draw a random safety sample. If any sentinel test turns out to be impacted (in ground truth or in a live run), the impact predictor was wrong — escalate to full rerun.

```python
def sample_sentinels(
    all_test_ids: set[str],
    predicted_ids: set[str],
    *,
    fraction: float = 0.10,
    min_sentinels: int = 5,
    seed: int | None = None,
) -> set[str]:
    """Sample sentinel tests from the non-predicted pool.

    Returns at least min_sentinels tests, or fraction * |non-predicted|,
    whichever is larger. If all tests are already predicted, returns empty set.
    """
    ...
```

**Sentinel behavior in evaluation:**
- For each version pair, after computing `S(Δ)`, check if any sentinel test is in `I(Δ)`.
- If yes: the sentinel "caught" a missed regression. Record this as a sentinel escalation event.
- The **sentinel catch rate** = (# versions where sentinel caught a miss) / (# versions where the predictor missed at least one impacted test).

---

### 5.5 Selector (`src/phase2/selector.py`)

**Purpose:** The top-level composition that orchestrates classification, tagging, prediction, and sentinel sampling into a single `select_tests` call.

```python
@dataclass
class SelectionResult:
    selected_ids: set[str]       # predicted ∪ sentinel
    predicted_ids: set[str]      # tests selected by impact predictor
    sentinel_ids: set[str]       # tests selected as sentinels
    changes: list[ClassifiedChange]
    total_tests: int
    call_reduction: float        # 1 - |selected| / |total|
    metadata: dict               # additional info for logging

def select_tests(
    old_prompt: str,
    new_prompt: str,
    test_cases: list[TestCase],
    *,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int | None = None,
) -> SelectionResult:
    """Full selection pipeline: classify → tag → predict → sentinel → combine."""
    # 1. Classify changes
    changes = classify_changes(old_prompt, new_prompt)

    # 2. Tag tests (can be cached per suite)
    tagged = tag_tests(test_cases)

    # 3. Predict impacted tests
    prediction = predict_impacted(changes, tagged, magnitude_threshold=magnitude_threshold)

    # 4. Sample sentinels
    all_ids = {tc.test_id for tc in test_cases}
    sentinels = sample_sentinels(all_ids, prediction.predicted_ids,
                                  fraction=sentinel_fraction, seed=sentinel_seed)

    # 5. Combine
    selected = prediction.predicted_ids | sentinels

    return SelectionResult(
        selected_ids=selected,
        predicted_ids=prediction.predicted_ids,
        sentinel_ids=sentinels,
        changes=changes,
        total_tests=len(test_cases),
        call_reduction=1.0 - len(selected) / len(test_cases),
        metadata={"reasons": prediction.reasons},
    )
```

---

### 5.6 Evaluator (`src/phase2/evaluator.py`)

**Purpose:** Run the selector against every version pair from Phase 1, compare the selected set to the ground truth impact set, and compute all metrics.

**Evaluation loop (per domain):**

```
For each version v in data/{domain}/versions/:
    Load P_base from data/{domain}/base_prompt.txt
    Load P_v from v.prompt_text
    Load test_cases from data/{domain}/eval_suite.yaml
    Load I(Δ) from ground_truth_{domain}.jsonl (the true impacted test set for v)

    S(Δ) = select_tests(P_base, P_v, test_cases)

    Compute:
      recall          = |I(Δ) ∩ S.predicted_ids| / |I(Δ)|     if |I(Δ)| > 0 else 1.0
      call_reduction  = 1 - |S.selected_ids| / |T|
      false_omissions = I(Δ) \ S.selected_ids                  (missed by both predictor AND sentinel)
      sentinel_hit    = |I(Δ) ∩ S.sentinel_ids| > 0            (sentinel caught a miss)

    Record per-version detail
```

**Metrics definitions:**

| Metric | Formula | Description |
|---|---|---|
| **Recall (predictor only)** | `\|I ∩ predicted\| / \|I\|` | Fraction of truly impacted tests caught by the predictor alone |
| **Recall (with sentinel)** | `\|I ∩ selected\| / \|I\|` | Fraction caught by predictor + sentinel combined |
| **Call reduction** | `1 - \|selected\| / \|T\|` | Fraction of tests skipped |
| **False omission rate** | `\|I \ selected\| / \|I\|` | Fraction of impacted tests missed entirely |
| **Sentinel catch rate** | `(# versions where sentinel caught a miss) / (# versions with any miss)` | How often the sentinel safety net fires |
| **Escalation rate** | `(# versions where sentinel caught) / (total versions)` | How often we'd escalate to full rerun |

**Aggregation dimensions:**
- Per domain (domain_a, domain_b)
- Per change `unit_type` (format, demonstration, policy, workflow, role, compound)
- Per magnitude bucket (low: <0.05, medium: 0.05-0.20, high: >0.20)
- Per mutation category (mechanical, llm_rewrite, llm_semantic, inconsistency, new_type, sequential_chain)

**Baselines for comparison:**

| Baseline | Description | Expected Call Reduction |
|---|---|---|
| **Full rerun** | Select all tests. `S = T`. | 0% (the status quo) |
| **Random 50%** | Select a random 50% of tests. Change-agnostic. | 50% (but low recall) |
| **Monitor-type heuristic** | Select only tests whose monitor_type matches the changed section's "family" (e.g., format changes → schema + format monitors) | Variable |

**Output format:**

`results/selection/metrics_{domain}.json`:
```json
{
  "domain": "domain_a",
  "total_versions": 70,
  "total_tests": 100,
  "aggregate": {
    "mean_recall_predictor": 0.92,
    "mean_recall_with_sentinel": 0.95,
    "mean_call_reduction": 0.45,
    "mean_false_omission_rate": 0.05,
    "sentinel_catch_rate": 0.60,
    "escalation_rate": 0.10,
    "versions_with_perfect_recall": 55
  },
  "by_change_type": { ... },
  "by_magnitude_bucket": { ... },
  "by_category": { ... },
  "baselines": {
    "full_rerun": { "recall": 1.0, "call_reduction": 0.0 },
    "random_50": { "mean_recall": 0.50, "call_reduction": 0.50 },
    "monitor_heuristic": { "mean_recall": 0.70, "call_reduction": 0.55 }
  }
}
```

`results/selection/detail_{domain}.jsonl` (one line per version):
```json
{"version_id": "v01", "changes": [...], "predicted_ids": [...], "sentinel_ids": [...], "impact_ids": [...], "recall": 1.0, "call_reduction": 0.47, "false_omissions": [], "sentinel_hit": false}
```

---

## 6. Implementation Order (Sprints)

### Sprint 1: Change Classifier + Test Tagger

**Goal:** Given any two prompts, automatically produce the same typed `Change` objects that Phase 1's mutator hard-coded, and enrich tests with inferred sensitivity tags.

**Tasks:**

- [ ] Create `src/phase2/__init__.py`
- [ ] Implement `src/phase2/change_classifier.py`
  - Section-level diffing using `parse_prompt()` + `diff_sections()`
  - Demo item-level diffing using `_split_demos()`
  - Magnitude computation using `_token_edit_distance()`
  - Key detection for format changes (regex for `"key":` patterns)
  - Reorder detection for workflow and demo sections
- [ ] Implement `src/phase2/test_tagger.py`
  - Rule-based enrichment of existing tags
  - Key dependency extraction from monitor configs
  - Demo dependency mapping from test tags to example labels
- [ ] Write `tests/test_change_classifier.py`
  - Test: identical prompts → empty change list
  - Test: known mutation (e.g., v01 rename) → correct Change with right unit_type, change_type, magnitude
  - Test: demo removal produces demonstration:deleted change
  - Test: workflow reorder produces workflow:reordered change
- [ ] Spot-check: run classifier on 10 version pairs per domain, compare output to the Change metadata in the YAML files

**Validation:** The classifier's output matches the version YAML metadata (unit_type, change_type) for at least 90% of versions. Magnitude values should be within ±0.05 of the YAML values.

---

### Sprint 2: Impact Predictor + Sentinel + Selector

**Goal:** Given changes and tagged tests, predict which tests to run, add a sentinel safety net, and produce a complete `SelectionResult`.

**Tasks:**

- [ ] Implement `src/phase2/impact_predictor.py`
  - Affinity table mapping change types to test sensitivity categories
  - Key-level matching for format changes
  - Magnitude gating (skip near-zero changes)
  - Compound change handling (union of per-change predictions)
- [ ] Implement `src/phase2/sentinel.py`
  - Random sampling with configurable fraction and seed
  - Minimum sentinel count floor
- [ ] Implement `src/phase2/selector.py`
  - Compose classifier + tagger + predictor + sentinel
  - Return `SelectionResult` with full metadata
- [x] `scripts/run_evaluation.py` — primary batch eval vs ground truth; `run_selection.py` for ad hoc runs
- [ ] Write `tests/test_impact_predictor.py`
  - Test: format:modified change selects schema_sensitive tests
  - Test: policy:modified change selects safety_sensitive tests
  - Test: role:modified change selects all tests
  - Test: compound change (format + policy) selects union
  - Test: magnitude below threshold → no tests selected
- [ ] Write `tests/test_selector.py`
  - Integration test: full pipeline from prompts to SelectionResult

**Validation:** Run evaluation on all versions (e.g. `python -m scripts.run_evaluation --all-domains`). Verify:
- Every version produces a non-empty selected set
- call_reduction is > 0 for most versions (we're skipping something)
- For versions with 0 impacted tests, recall is trivially 1.0

---

### Sprint 3: Evaluation

**Goal:** Comprehensive evaluation of the rule-based selector against ground truth, with metric breakdowns and baseline comparisons.

**Tasks:**

- [ ] Implement `src/phase2/evaluator.py`
  - Load ground truth from Phase 1 JSONL files
  - Run selector on each version pair
  - Compute per-version metrics (recall, call_reduction, false_omissions, sentinel_hit)
  - Aggregate metrics by domain, change_type, magnitude_bucket, mutation category
  - Implement baseline selectors (random 50%, monitor-type heuristic) for comparison
  - Output metrics JSON + detail JSONL
- [ ] Run evaluation for all domains
- [ ] Analyze results:
  - Which change types have highest/lowest recall?
  - Which change types have highest/lowest call reduction?
  - Where does the rule-based predictor fail (false omissions)?
  - How often does the sentinel catch missed regressions?
  - How does the rule-based selector compare to baselines?
- [ ] Sweep sentinel fraction (0.05, 0.10, 0.15, 0.20) and magnitude threshold (0.0, 0.005, 0.01, 0.02) to characterize sensitivity

**Validation:** Results match expectations from Phase 1's ground truth analysis:
- Format changes should have high recall (format is the #1 impactor)
- Role/tone changes should have low false omissions (they have few impacts)
- Compound changes should have recall at least as good as individual changes

---

## 7. Success criteria (design targets vs. observed run)

The table below states **aspirational** targets from the original plan. The **observed** rule-based metrics on the three-domain benchmark (sentinel 10%, min 5, default magnitude threshold) are summarized in [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) — e.g. mean recall (+sentinel) is **not** ≥0.95 on every domain under rules alone; learned models improve hard domains.

| Criterion | Original target | Notes |
|---|---|---|
| Recall (predictor) | High on easy strata | Varies by `unit_type` / domain |
| Recall (+ sentinel) | Safety net | Reported in `metrics_domain_*.json` |
| Call reduction | Meaningful skip rate | Trade-off with recall |
| FOR | Low in skipped pool | Definition: impacted ∩ not_selected / not_selected |
| Evaluation completes | All versions | 70 × 3 without crashes |
| vs. random 50% | Beat on recall | See `baselines` in metrics JSON |

---

## 8. What Phase 2 Feeds Into

- **Phase 3** (Learned Selector) replaces the rule-based affinity table with **LightGBM, XGBoost, and logistic regression**. The `TaggedTest` features, classifier output, and evaluator are reused. Training rows are built from Phase 1 ground truth (`src/phase3/dataset.py`).
- **Phase 4** (Evaluation & Paper) reports Phase 2 metrics as the "rule-based baseline" alongside Phase 3's learned selector. The evaluation framework from this phase produces all the tables and figures for the paper.
