# Change-Aware Regression Test Selection for LLM Applications

## 1. Problem Statement

### 1.1 The cost problem

Teams building LLM applications run evaluation suites on every prompt, config, or model change. A typical suite has 100–500 test cases. Each test case requires an LLM call ($0.001–$0.02 per call depending on model). On every pull request that touches a prompt, the full suite reruns.

Most of these reruns are wasted. A change to the output format section does not affect safety-policy test cases. A deleted few-shot example does not affect schema-validation tests for fields that example never demonstrated. A reworded role section does not affect tests targeting workflow step ordering.

Yet current tools — Promptfoo, LangSmith, custom eval pipelines — rerun everything. They have no notion of *which tests are actually impacted by a specific change*.

### 1.2 The research question

> **Given a typed change to an LLM application's prompt, config, or model settings, can we safely select a subset of the eval suite that is sufficient to detect any regression introduced by that change — and skip the rest?**

"Safely" means: if the full suite would have caught a regression, the selected subset catches it too. The cost of a missed regression (false omission) is much higher than the cost of running extra tests (false inclusion), so the selector must be **conservative by design**.

### 1.3 Formal formulation

Let:
- `P` be the current prompt/config, `P'` the modified version
- `T = {t_1, ..., t_N}` be the full eval suite, each `t_i = (x_i, m_i)` with input `x_i` and monitor `m_i`
- `Δ(P, P') = {δ_1, ..., δ_k}` be the set of typed changes (each `δ_j` has a `unit_type`, `change_type`, and `content_diff`)
- `outcome(P, t_i) ∈ {pass, fail}` be the eval result for test `t_i` under prompt `P`
- `regressed(t_i) = (outcome(P, t_i) = pass) ∧ (outcome(P', t_i) = fail)` — test `t_i` regressed

Define the **impact set** as the tests whose outcomes actually change:

```
I(Δ) = { t_i ∈ T : outcome(P, t_i) ≠ outcome(P', t_i) }
```

The selector produces a **selected set** `S(Δ) ⊆ T` with `|S| << |T|`. The goal is:

```
minimize  |S(Δ)|
subject to  I(Δ) ⊆ S(Δ)   (safety: no impacted test is omitted)
```

Since `I(Δ)` is unknown before running tests, the selector must predict it from the change metadata `Δ` alone. We relax the hard safety constraint to a measurable **recall** target:

```
Recall = |I(Δ) ∩ S(Δ)| / |I(Δ)| ≥ τ     (e.g., τ = 0.95)
```

The system should also always include a **sentinel sample** — a small random draw from tests predicted to be unaffected — to detect when the impact model is wrong.

### 1.4 Why this is different from benchmark compression

Recent work on efficient LLM evaluation (Cer-Eval, SubLIME, EssenceBench, SCALES++) selects a **fixed small subset** of a benchmark that approximates full-benchmark model rankings. These are **change-agnostic** — the selected subset is the same regardless of what changed. They answer: "which 50 tests out of 500 represent this benchmark well?"

This project answers a different question: "given *this specific change* (e.g., the format section was rewritten), which tests are likely affected?" The selected set is different for every change. A format change selects schema tests; a safety-policy change selects refusal tests; a demo edit selects tests sensitive to few-shot examples. This is **regression test selection**, not benchmark compression.

### 1.5 Why now

- Ma et al. (2024) established that prompt regression is a real problem and that LLM regression testing needs fundamental rethinking — but did not build a selector.
- RETAIN (Dixit et al. 2024) provides interactive regression testing for model migrations — but focuses on error discovery across model changes, not selective rerun based on prompt changes.
- ReCatcher (2025) builds regression testing for code generation across model updates — a different domain and change type.
- Promptfoo supports `--filter-metadata` for manual test filtering, but has no automatic change-aware selection.
- The classical regression test selection literature (Rothermel & Harrold 1997, Yoo & Harman 2012) is mature for code changes, but the "change taxonomy" for LLM applications (prompt sections, few-shot examples, output schemas, tool descriptions, retrieval settings) has not been defined or exploited.

The gap is clear: **change-aware selective rerun for evolving LLM applications**.

---

## 2. Related Work & Positioning

| Area | Key Work | Relationship |
|---|---|---|
| **Classical regression test selection** | Rothermel & Harrold (1997): safe selection via control flow graphs. Yoo & Harman (2012): comprehensive survey. | Foundational algorithms. Not applicable to LLM apps — no control flow graph, no code dependency analysis. We adapt the *principles* (safety, conservatism, change-based selection) to a new artifact type. |
| **LLM regression testing** | Ma et al. (2024): "(Why) Is My Prompt Getting Worse?" identifies the problem. RETAIN (2024): interactive tool for model migration. ReCatcher (2025): code generation regression testing. | Established that LLM regression testing is needed. None of them build a **test selector** that predicts which tests are impacted by a given change. |
| **Efficient LLM evaluation** | Cer-Eval (2025): certifiable subset selection. SubLIME (ACL 2025 Best Theme Paper): rank-correlation subset. EssenceBench: benchmark compression. SCALES++: cognitive-scale embeddings. | Reduce eval cost by compressing benchmarks. Change-agnostic — same subset for every evaluation. Complementary: their compressed suite could be the input; our selector further reduces it per-change. |
| **Prompt component attribution** | Regression Framework (arXiv:2603.26830): statistical models of prompt component impact. | Identifies which component *types* matter for performance. We use change *instances* to predict which *tests* are affected. Their findings (e.g., "format constraints drive 42.7% of quality") can inform our impact model priors. |
| **Prompt management & CI/CD** | Promptfoo (CI/CD gates, `--filter-metadata`), LangSmith (versions, diffs, rollback). | Operational infrastructure. Promptfoo already supports manual test filtering — our selector automates it. Natural integration target. |

**Positioning claim:** This project builds the first change-aware regression test selector for LLM applications, where changes are typed prompt/config artifacts and the selector uses change metadata to predict impacted test slices, with conservative safety guarantees and a sentinel fallback.

---

## 3. System Design

### 3.1 Overview

```
Input:  P (old prompt/config), P' (new prompt/config), T (full eval suite)
Output: S ⊆ T (selected subset to run), plus sentinel sample

Pipeline:
  1. Change Classifier     diff(P, P') → Δ = [(unit_type, change_type, magnitude), ...]
  2. Test Tagger            T → { t_i : [tags] }  (offline, done once per suite)
  3. Impact Predictor       (Δ, tagged_tests) → predicted_impact_set
  4. Sentinel Sampler       T \ predicted_impact_set → sentinel_sample (random k%)
  5. Selector               S = predicted_impact_set ∪ sentinel_sample
  6. Runner                 run S on P', compare to cached P results
  7. Safety Check           if sentinel catches unexpected regressions → escalate to full rerun
```

### 3.2 Change Classifier

Diff `P` and `P'` at the section and item level. Output a list of **typed changes**.

**Change unit taxonomy for LLM applications:**

| Unit Type | Examples | Detection Method |
|---|---|---|
| `role` | Persona, behavior framing | Section heading / label matching + keyword detection |
| `format` | Output schema, serialization rules, required keys | Regex for JSON schema keywords, key names |
| `demonstration` | Few-shot examples (array of items) | Delimiter-based item extraction |
| `demo_item` | Individual example within a demonstration section | Item-level diff |
| `policy` | Safety rules, refusal instructions, compliance | Keywords: "must not", "refuse", "never", "always" |
| `workflow` | Step-by-step procedures, routing logic | Numbered steps, conditionals |
| `tool_description` | Tool/function definitions, parameter schemas | JSON function-call schemas |
| `retrieval_config` | Chunk size, top-k, similarity threshold | Config file key-value changes |
| `model_settings` | temperature, max_tokens, model version | Config file key-value changes |

**Change type classification:** `inserted`, `deleted`, `modified`, `reordered`.

**Magnitude:** token-level edit distance, normalized by section size. A 5-word tweak to a 500-word section is low magnitude; a full rewrite is high.

```python
@dataclass
class Change:
    unit_id: str
    unit_type: str       # role | format | demonstration | demo_item | policy | ...
    change_type: str     # inserted | deleted | modified | reordered
    magnitude: float     # 0.0 to 1.0
    content_diff: str    # for logging / debugging
```

### 3.3 Test Tagger

Each test case is tagged with the **monitor types** it exercises and the **prompt features** it depends on. This is done once per suite (offline) and updated only when tests are added.

**Tagging method (MVP: rule-based):**
- If the test's monitor is a JSON schema check → tag `schema_sensitive`
- If the test's expected output references specific keys → tag with those key names
- If the test's input is adversarial / safety-probing → tag `safety_sensitive`
- If the test's assertion checks for specific formatting → tag `format_sensitive`
- If the test was added as a regression for a known failure → tag with the failure's root cause section

**Tagging method (ablation-based sensitivity, Phase 3 prep):**
- Run each test with ablated prompts (remove one section at a time) and record which ablations flip the outcome. This directly measures section sensitivity per test.

```python
@dataclass
class TaggedTest:
    test_id: str
    input_text: str
    monitor_type: str          # schema | policy | freeform | unit_test | ...
    sensitive_to: list[str]    # ["format", "demonstration", "role"]
    tags: list[str]            # ["billing", "refund", "json_keys:action,priority"]
```

### 3.4 Impact Predictor

Maps changes to predicted impacted tests. The core is a **change-type → test-tag affinity matrix**.

**MVP: rule-based affinity table**

| Change Unit Type | Predicted Impact Tags |
|---|---|
| `format` | `schema_sensitive`, `format_sensitive`, tests referencing changed keys |
| `demonstration` / `demo_item` | Tests whose topic/domain overlaps with the changed example |
| `policy` | `safety_sensitive`, policy-specific tests |
| `workflow` | Tests checking step ordering, routing, multi-step outputs |
| `role` | All tests (role changes are high-risk, low-frequency — run everything) |
| `tool_description` | Tests involving tool calls |
| `model_settings` | All tests (model changes can affect anything) |

**Phase 2: learned affinity model**

Train a classifier: given `(change_features, test_features) → P(outcome_changes)`. Features include change type, magnitude, test monitor type, semantic similarity between change content and test input, historical co-regression patterns.

### 3.5 Sentinel Sampler

From the tests **not** in the predicted impact set, draw a random sample of size `max(5, ⌈0.10 × |T \ S_predicted|⌉)` (10% of the non-predicted pool, at least five tests). Implemented as `sample_sentinels` in `src/phase2/sentinel.py`. If any sentinel test regresses, the impact model was wrong — escalate to full rerun.

This is the safety net. It makes the system **fail-safe**: the worst case is running the full suite (same as today), not silently missing a regression.

### 3.6 Cost model

For a suite of N=300 tests with k=5 changed units:

| Strategy | Tests Run | LLM Calls | Cost (GPT-4o-mini) |
|---|---|---|---|
| Full rerun | 300 | 300 | $0.30 |
| Change-aware selection (50% reduction) | 150 + 15 sentinel = 165 | 165 | $0.17 |
| Change-aware selection (70% reduction) | 90 + 21 sentinel = 111 | 111 | $0.11 |
| Change-aware selection (format-only change) | 40 + 26 sentinel = 66 | 66 | $0.07 |

Over 100 PRs/month: full rerun costs $30; 70% reduction costs $11. Savings scale linearly with suite size and PR frequency.

For expensive models (GPT-4o at $0.016/call): full rerun = $4.80/PR = $480/month for 100 PRs. 70% reduction saves $336/month.

---

## 4. Implementation Plan

### Phase 1: Infrastructure & Baseline (Weeks 1–3)

**Goal:** Build the eval harness, generate the data, and establish the "full rerun" baseline.

**As implemented in this repository:** three benchmark domains (**domain_a**, **domain_b**, **domain_c**), **70** prompt versions each (v01–v70), ground truth under `results/baseline/`. Primary baseline runner: OpenAI Batch API (`python -m scripts.run_batch_baseline run`, or `python -m scripts.run_phases phase1`). See [`PHASE1.md`](PHASE1.md), [`README.md`](../README.md), and [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md).

**Deliverables:**

1. **Three domain applications with prompts and eval suites:**
   - **Domain A: JSON extraction bot.** Large system prompt with role, format (JSON schema), demonstrations, policy. Eval suite: **100** tests.
   - **Domain B: Coding assistant.** Prompt with role, format, demonstrations, workflow. Eval suite: **80** tests.
   - **Domain C: Third scenario family** (distinct base prompt and suite). Eval suite: **90** tests.

2. **Version history generator:** Script that produces prompt versions by applying controlled mutations (`scripts/generate_versions.py`, mutators in `src/phase1/prompts/`):
   - Format, demo, policy, workflow, role, compound edits (see `PHASE1.md` for taxonomy)
   - **70 versions per domain** in the shipped dataset (210 versions total)

3. **Full-rerun baseline:** Run the complete eval suite on the base prompt and on each version. Record pass/fail vs. the **prior-version baseline** for labeling (see Phase 1 doc). This yields per-version impact sets for Phases 2–3.

4. **Baseline metrics:** Total calls, total cost, wall-clock time (summarized per domain in `summary_domain_*.json`).

**Tech stack:**

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM calls | `litellm` (unified API interface) |
| Eval framework | Promptfoo (YAML configs) or custom lightweight harness |
| Caching | SQLite via `diskcache`, keyed by `sha256(prompt + input + model)` |
| CLI | `typer` (`scripts/*.py`) |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` (local); PCA-reduced in Phase 3 |
| Boosted trees | `lightgbm`, `xgboost` (Phase 3) |

---

### Phase 2: Change Classifier & Rule-Based Selector (Weeks 3–5)

**Goal:** Build the change classifier and the MVP rule-based impact predictor. Measure call reduction and safety (recall).

**Deliverables:**

1. **Prompt parser:** Regex-based section extractor (Markdown headings, XML tags, explicit labels, fenced code blocks). Demo item extractor (User/Assistant pairs, numbered examples, JSON arrays).

2. **Diff engine:** Section alignment via weighted similarity (edit distance + cosine + family match). Hungarian algorithm for matching. Item-level diff for demonstration sections.

3. **Rule-based test tagger:** Tag each test by monitor type and section sensitivity using keyword rules.

4. **Rule-based impact predictor:** The affinity table from Section 3.4.

5. **Sentinel sampler:** `max(5, ⌈10%⌉)` of the non-predicted test pool (`src/phase2/sentinel.py`).

6. **Selector + runner:** Select tests, run them, compare to baseline, escalate if sentinel catches regressions.

**Evaluation at this phase:**
- For each version in each domain (e.g. **70 × 3** in the full benchmark), compare `S(Δ)` to `I(Δ)`.
- Measure: **recall** (did we select all regressed tests?), **call reduction** (what % of tests did we skip?), **false omissions** (regressed tests we missed), **sentinel catch rate** (how often did the sentinel detect a missed regression?).

---

### Phase 3: Learned Selector (Weeks 5–7)

**Goal:** Replace rule-based impact prediction with learned models trained on Phase 1 ground truth.

**Deliverables:**

1. **Sensitivity profiling (optional offline):** Section ablations per test via `scripts/run_ablation.py`; vectors in `data/{domain}/sensitivity_profiles.json` when generated.

2. **Learned impact predictor:** **LightGBM**, **XGBoost**, and **logistic regression** (`src/phase3/`) over features:
   - Change features: unit_type (one-hot), change_type, magnitude, content embedding
   - Test features: monitor_type, sensitivity vector, test input embedding
   - Pairwise features: cosine similarity between change content and test input
   - Target: `P(outcome_changes | change, test)`
   - Training data: Phase 1 ground truth at **(version, test)** granularity — on the order of **tens of thousands** of labeled rows across three domains (see `src/phase3/dataset.py`).

3. **Threshold tuning:** Choose the classification threshold to target recall ≥ 0.95 on a held-out validation set. The threshold controls the conservatism: lower threshold = more tests selected = higher recall, lower reduction.

**Evaluation at this phase:**
- Same metrics as Phase 2, but now comparing rule-based vs. learned selector.
- Ablation: threshold sweep (0.3, 0.5, 0.7, 0.9) showing the recall–reduction tradeoff curve.
- Ablation: sentinel size (5%, 10%, 15%) showing safety–cost tradeoff.

---

### Phase 4: Evaluation & Paper (Weeks 7–9)

**Goal:** Comprehensive evaluation, ablation studies, write-up.

**Primary research questions:**

| ID | Question |
|---|---|
| RQ1 | How much call reduction does change-aware selection achieve while maintaining ≥95% regression detection recall? |
| RQ2 | Does the learned selector outperform the rule-based selector? |
| RQ3 | How effective is the sentinel sample at catching missed regressions? |
| RQ4 | How does selection quality vary by change type (format-only vs. compound changes)? |
| RQ5 | What is the cost savings in realistic CI/CD scenarios? |

**Metrics:**

| Metric | Definition |
|---|---|
| **Regression detection recall** | `\|I(Δ) ∩ S(Δ)\| / \|I(Δ)\|` — fraction of actual regressions caught |
| **Call reduction rate** | `1 - \|S(Δ)\| / \|T\|` — fraction of tests skipped |
| **False omission rate** | Among tests **not** selected: `\|I(Δ) ∩ (T \ S)\| / \|T \ S\|` — risk in the skipped pool (see `src/phase2/evaluator.py`) |
| **Sentinel escalation rate** | Fraction of changes where sentinel catches a missed regression and triggers full rerun |
| **Cost savings** | Dollar amount saved per PR and per month vs. full rerun |
| **Wall-clock reduction** | Time saved (parallel calls bounded by rate limits) |

**Baselines:**

| Baseline | Description |
|---|---|
| **Full rerun** | Run all tests. 100% recall, 0% reduction. The upper bound on cost and lower bound on risk. |
| **Random selection (50%)** | Run a random 50% of tests. Change-agnostic. Shows whether any selection is better than random. |
| **Monitor-type heuristic** | Run only tests whose monitor type matches the changed section family. No learned component. |
| **Cer-Eval (adapted)** | *Planned comparison;* not wired in the current codebase. Implemented baselines: full rerun, random 50%, monitor heuristic, Phase 2 rules. |
| **Manual `--filter-metadata`** | Simulate a human using Promptfoo's tag filtering to select tests manually. |

**Ablation studies:**

| Ablation | Variable | Values |
|---|---|---|
| Selector type | Rule-based vs. learned | — |
| Sentinel size | % of non-selected tests | 0%, 5%, 10%, 15%, 20% |
| Recall threshold | Classification threshold | 0.3, 0.5, 0.7, 0.9 |
| Change complexity | Single-section vs. compound | 1, 2, 3, 4+ changes |
| Suite size | Number of tests | 50, 100, 200, 300 |
| LLM non-determinism | N majority-vote samples | 1, 3, 5 |

**Data summary:**

| Item | Count | Source |
|---|---|---|
| Domain applications | 3 | Hand-built (`domain_a`, `domain_b`, `domain_c`) |
| Prompt versions per domain | 70 | Scripted mutations (v01–v70) |
| Versions with ground truth | 210 total | Full-rerun / batch baseline |
| Tests per domain | 100 / 80 / 90 | YAML eval suites |
| Labeled (version, test) outcomes | ~19k+ rows | From Phase 1 JSONL; expanded with features in Phase 3 |
| Sensitivity profiles | Optional | `sensitivity_profiles.json` per domain if ablation script run |

**No external data needed.** The ground truth is a byproduct of running the system. No manual annotation, no production partners, no synthetic circularity.

---

## 5. Deliverables

| Phase | Deliverable | Week |
|---|---|---|
| 1 | Three domain apps + eval suites + 70 versions/domain + full-rerun ground truth | 3 |
| 2 | Change classifier + rule-based selector + evaluator + sweep | 5 |
| 3 | Learned selectors (LGBM, XGB, logreg) + version holdout / k-fold / cross-domain eval | 7 |
| 4 | Unified analysis, statistics, figures, cost projections (`run_phase4_analysis`) | 9 |
| Bonus | Promptfoo plugin prototype | 10 |

---

## 6. Feasibility & Risks

**Why this is feasible:**
- MVP is rule-based: change type → test tag lookup. No model training required.
- Ground truth is free: full rerun generates labeled data automatically.
- Three domains with 70 versions each produce **tens of thousands** of labeled (version, test) outcomes — enough for tree and linear models.
- Batch baseline pricing depends on OpenAI Batch rates; see `EVALUATION_RESULTS.md` for observed run costs.
- The entire data generation pipeline can run in a few hours with parallel API calls.

**Risks and mitigations:**

| Risk | Severity | Mitigation |
|---|---|---|
| Impact model is too conservative → low call reduction | Low | This is the safe failure mode. Even 30% reduction is a useful result. Learned model can improve. |
| Impact model is too aggressive → misses regressions | High | Sentinel sample catches misses and triggers escalation to full rerun. System degrades gracefully to full-rerun behavior. |
| LLM non-determinism causes flaky test outcomes | Medium | temperature=0 default; N=3 majority vote when needed; mark flaky tests and exclude from recall calculation. |
| Synthetic mutations don't reflect real prompt changes | Medium | Use realistic mutation types (based on what teams actually change: formatting, demos, policies). Report mutation-type breakdown clearly. |
| Change taxonomy doesn't cover all LLM app artifacts | Low | Taxonomy is extensible. Cover the common cases (prompt sections, demos, schemas) first. |

---

## 7. Expected Contributions

1. **A change taxonomy for LLM applications.** A typed classification of changeable artifacts in LLM apps (prompt sections, demo items, schemas, tool descriptions, model settings) with associated impact profiles.

2. **A change-aware regression test selection algorithm.** A rule-based and learned selector that predicts impacted tests from change metadata, with conservative safety guarantees via sentinel sampling.

3. **Empirical evidence.** Measurement of call reduction, recall, FOR, and cost projections across **three** domains, version-holdout and k-fold splits, cross-domain transfer, and parameter sweeps — summarized in [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md).

4. **Integration pathway.** A concrete design for a Promptfoo plugin that automates selective rerun on prompt PRs in CI/CD pipelines.

---

## 8. Future Work

### 8.1 Online Learning via Predict–Measure–Retrain Loop

The learned selector in Phase 3 is trained offline on a fixed ground truth dataset generated by full reruns. In a production deployment, however, each CI run produces new labeled examples — every test we run yields a fresh `(change, test, impacted)` observation. This creates an opportunity for an **online learning loop** analogous to the cost model refinement in TVM's AutoTVM [Chen et al. 2018], where an XGBoost model is iteratively retrained as it accumulates real measurements:

1. A developer submits a prompt change.
2. The selector predicts impacted tests and runs them (plus sentinels).
3. The observed outcomes are fed back as new training examples.
4. The model is periodically retrained on the accumulated data, improving its predictions over time.

A key challenge is **partial observability**: unlike TVM, where every evaluated candidate produces a ground truth measurement, CARTS only observes outcomes for the tests it selects — skipped tests produce no signal. This can create a self-reinforcing bias where the model never learns about test sensitivities it has already decided to ignore. Three mechanisms mitigate this:

- **Sentinel sampling** acts as an exploration arm (analogous to epsilon-greedy), providing ground truth from the unselected pool and detecting model blind spots.
- **Periodic full reruns** (e.g., every 10th–20th PR) provide uncensored calibration data at modest cost — a 90–95% reduction in full-rerun frequency while maintaining model freshness.
- **Uncertainty-driven selection** (active learning) can prioritize tests where the model is least confident, maximizing the information gained from each CI run.

This online extension is complementary to the offline evaluation presented in this work and represents a natural path toward a self-improving, production-grade selector.
