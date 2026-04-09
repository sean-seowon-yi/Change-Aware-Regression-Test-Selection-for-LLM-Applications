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
- `scripts/` — Typer CLIs for baseline, Phase 2 eval, Phase 3 train/eval, Phase 4 analysis
- `models/*.pkl` — trained selectors (gitignored if not committed)
- `results/` — baseline, selection, phase3, phase4 outputs (often gitignored)

**Requirements:** Python 3.11+ (see `requirements.txt`). Set `OPENAI_API_KEY` (e.g. in `.env`).

## Quick reproduction (after dependencies)

```bash
pip install -r requirements.txt
python -m scripts.run_batch_baseline --both          # Phase 1 ground truth (Batch API)
python -m scripts.run_evaluation --all-domains --sweep   # Phase 2 rule selector + sweep
python -m scripts.run_phase3_train train-all
python -m scripts.run_phase3_eval evaluate-all
python -m scripts.run_phase3_eval kfold-evaluate-all
python -m scripts.run_phase3_eval compare
python -m scripts.run_phase4_analysis full
```

See each phase doc and `scripts/*.py` for options (`--help`). Phase 1 baseline uses the OpenAI Batch API (see `run_batch_baseline.py`).

## Tests

```bash
python -m pytest tests/ -q
```

## GitHub

Target name: **change-aware-llm-test-selection** (`sean-seowon-yi/change-aware-llm-test-selection`). Rename on GitHub if needed, then:

`git remote set-url origin git@github.com:sean-seowon-yi/change-aware-llm-test-selection.git`
