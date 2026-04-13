# Change-Aware Regression Test Selection for LLM Applications

**Short name:** CARTS — change-aware regression test selection for prompts and LLM app configs.

Typed change analysis, impact prediction, and conservative test subset selection so teams can avoid rerunning full eval suites when only part of a prompt, config, or model surface changes.

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/PROPOSAL_SUMMARY.md`](docs/PROPOSAL_SUMMARY.md) | Problem, design, related work, phased plan |
| [`docs/PHASE1.md`](docs/PHASE1.md) | Harness, data layout, baseline / ground truth |
| [`docs/PHASE2.md`](docs/PHASE2.md) | Change classifier, rule-based selector, evaluator |
| [`docs/PHASE3.md`](docs/PHASE3.md) | Learned models (LightGBM, XGBoost, logistic regression) |
| [`docs/PHASE4.md`](docs/PHASE4.md) | Unified analysis, statistics, figures, cost projections |
| [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md) | Full numeric results vs `results/` (half-up rounding) |

## Repository layout (implementation)

- `config/settings.yaml` — model, cache, concurrency
- `data/domain_{a,b,c}/` — `base_prompt.txt`, `eval_suite.yaml`, `versions/v01.yaml` … `v70.yaml`
- `src/phase1/` — harness (`harness/`), prompts (`prompts/`), baseline (`baseline/`), `models.py`
- `src/phase2/` — classifier, tagger, rules, sentinel, selector, `evaluator.py`
- `src/phase3/` — features, dataset, training, learned predictor/selector
- `src/phase4/` — aggregation, statistics, plots, cost model, ablations
- `scripts/` — Typer CLIs for baseline, Phase 2 eval, Phase 3 train/eval, Phase 4 analysis; `run_phases.py` wraps the same invocations behind one entry point
- `models/*.pkl` — trained selectors (gitignored if not committed)
- `results/` — baseline, selection, phase3, phase4 outputs (often gitignored)

**Requirements:** Python 3.11+ (see `requirements.txt`). Set `OPENAI_API_KEY` (e.g. in `.env`).

## Unified CLI (`run_phases`)

`python -m scripts.run_phases` delegates to the same `python -m scripts.<module> …` commands below, so flags match the underlying Typer apps. Use `list` for a cheat sheet, `phase4 <step>` for individual Phase 4 steps (`collect`, `statistics`, `ablations`, `cost-analysis`, `figures`, `tables`, `full`).

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

`phase2` requires `--all-domains` or `--domain <name>`. `phase3-eval` does not allow `--compare` and `--kfold` together.

## Quick reproduction (after dependencies)

Either use `run_phases` (above) or call the phase scripts directly:

```bash
pip install -r requirements.txt
python -m scripts.run_batch_baseline run --both          # Phase 1 ground truth (Batch API)
python -m scripts.run_evaluation evaluate --all-domains --sweep   # Phase 2 rule selector + sweep
python -m scripts.run_phase3_train train-all
python -m scripts.run_phase3_eval evaluate-all
python -m scripts.run_phase3_eval kfold-evaluate-all
python -m scripts.run_phase3_eval compare
python -m scripts.run_phase4_analysis full
```

**Metrics note:** Phase 2–4 aggregates include **effective recall** and **effective call reduction** per version: when a sentinel would trigger a complement pass (full suite), those versions count as recall 1.0 and call reduction 0 for the “effective” roll-up. See `compute_effective_metrics` in `src/phase2/evaluator.py` and [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md) §1.

See each phase doc and `scripts/*.py` for options (`--help`). Phase 1 baseline uses the OpenAI Batch API (`run_batch_baseline.py`, subcommand `run`).

## Tests

```bash
python -m pytest tests/ -q
```

## GitHub

Target name: **change-aware-llm-test-selection** (`sean-seowon-yi/change-aware-llm-test-selection`). Rename on GitHub if needed, then:

`git remote set-url origin git@github.com:sean-seowon-yi/change-aware-llm-test-selection.git`
