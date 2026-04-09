"""Statistical analysis for Phase 4 evaluation.

Provides paired significance tests (Wilcoxon signed-rank), bootstrap
confidence intervals, and effect-size computation for comparing selector
configurations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of *values*.

    Returns (ci_lower, ci_upper).
    """
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(arr), size=len(arr))
        boot_means[i] = arr[idx].mean()
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lo, hi


def metric_summary(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Mean, std, and bootstrap CI for a metric vector."""
    arr = np.asarray(values, dtype=np.float64)
    ci_lo, ci_hi = bootstrap_ci(arr, confidence=confidence,
                                 n_bootstrap=n_bootstrap, seed=seed)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "n": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def paired_comparison(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    *,
    alternative: str = "greater",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired Wilcoxon signed-rank test + bootstrap CI on the difference.

    *metric_a* and *metric_b* must be aligned per-version vectors.
    *alternative*: "greater" tests H1: mean(a) > mean(b).

    Returns dict with test statistic, p-value, effect size (Cohen's d),
    mean difference, and 95% bootstrap CI on the difference.
    """
    a = np.asarray(metric_a, dtype=np.float64)
    b = np.asarray(metric_b, dtype=np.float64)
    diff = a - b

    if np.all(diff == 0):
        return {
            "wilcoxon_stat": None,
            "p_value": 1.0,
            "effect_size_cohens_d": 0.0,
            "mean_diff": 0.0,
            "ci_95_lo": 0.0,
            "ci_95_hi": 0.0,
        }

    try:
        stat, p_value = wilcoxon(a, b, alternative=alternative)
    except ValueError:
        stat, p_value = None, 1.0

    std = float(diff.std(ddof=1)) if len(diff) > 1 else 1e-9
    effect_size = float(diff.mean()) / max(std, 1e-9)

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot_diffs[i] = diff[idx].mean()
    alpha = 0.05 / 2
    ci_lo, ci_hi = float(np.percentile(boot_diffs, 100 * alpha)), float(np.percentile(boot_diffs, 100 * (1 - alpha)))

    return {
        "wilcoxon_stat": float(stat) if stat is not None else None,
        "p_value": float(p_value),
        "effect_size_cohens_d": round(effect_size, 4),
        "mean_diff": round(float(diff.mean()), 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
    }


# ---------------------------------------------------------------------------
# Multi-configuration comparison driver
# ---------------------------------------------------------------------------

def _load_detail_metrics(path: Path) -> list[dict]:
    """Load per-version detail from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_all_comparisons(
    domains: list[str] | None = None,
) -> dict:
    """Run all pairwise comparisons from stored evaluation results.

    Looks for detail JSONL files under ``results/selection/`` (Phase 2)
    and ``results/phase3/`` (Phase 3) and compares per-version recalls.

    Returns a dict keyed by comparison name containing test statistics.
    """
    if domains is None:
        domains = ["domain_a", "domain_b", "domain_c"]

    project_root = Path(__file__).resolve().parents[2]
    selection_dir = project_root / "results" / "selection"
    phase3_dir = project_root / "results" / "phase3"

    comparisons: dict[str, dict] = {}

    def _find_detail(directory: Path, patterns: list[str]) -> Path | None:
        """Try multiple glob patterns and return the first match."""
        for pat in patterns:
            matches = sorted(directory.glob(pat))
            if matches:
                return matches[0]
        return None

    for domain in domains:
        rule_detail = selection_dir / f"detail_{domain}.jsonl"
        if not rule_detail.exists():
            logger.warning("Missing rule-based detail for %s", domain)
            continue
        rule_records = _load_detail_metrics(rule_detail)
        rule_by_vid = {r["version_id"]: r for r in rule_records}

        for model_type in ["lightgbm", "xgboost", "logreg"]:
            for split in ["temporal", "kfold"]:
                patterns = [
                    f"detail_*{model_type}_{domain}_{split}*.jsonl",
                    f"detail_{model_type}_{domain}_{split}*.jsonl",
                    f"detail_eval_{domain}_{model_type}_{domain}_{split}*.jsonl",
                    f"detail_kfold_{domain}_{model_type}*.jsonl",
                ]
                learned_detail = _find_detail(phase3_dir, patterns)
                if learned_detail is None:
                    continue
                learned_records = _load_detail_metrics(learned_detail)
                learned_by_vid = {r["version_id"]: r for r in learned_records}

                common_vids = sorted(set(rule_by_vid) & set(learned_by_vid))
                if len(common_vids) < 5:
                    continue

                rule_recalls = [rule_by_vid[v].get("recall_with_sentinel", rule_by_vid[v].get("recall_predictor", 0)) for v in common_vids]
                learned_recalls = [learned_by_vid[v].get("recall_with_sentinel", learned_by_vid[v].get("recall_predictor", 0)) for v in common_vids]

                comp_key = f"learned_{model_type}_{domain}_{split}_vs_rule"
                comparisons[comp_key] = paired_comparison(
                    learned_recalls, rule_recalls, alternative="greater",
                )
                comparisons[comp_key]["n_versions"] = len(common_vids)
                comparisons[comp_key]["domain"] = domain
                comparisons[comp_key]["model_type"] = model_type
                comparisons[comp_key]["split"] = split

        for ma, mb in [("lightgbm", "logreg"), ("lightgbm", "xgboost")]:
            for split in ["temporal", "kfold"]:
                detail_a = _find_detail(phase3_dir, [
                    f"detail_*{ma}_{domain}_{split}*.jsonl",
                    f"detail_kfold_{domain}_{ma}*.jsonl",
                ])
                detail_b = _find_detail(phase3_dir, [
                    f"detail_*{mb}_{domain}_{split}*.jsonl",
                    f"detail_kfold_{domain}_{mb}*.jsonl",
                ])
                if detail_a is None or detail_b is None:
                    continue
                recs_a = {r["version_id"]: r for r in _load_detail_metrics(detail_a)}
                recs_b = {r["version_id"]: r for r in _load_detail_metrics(detail_b)}
                common = sorted(set(recs_a) & set(recs_b))
                if len(common) < 5:
                    continue
                recalls_a = [recs_a[v].get("recall_with_sentinel", 0) for v in common]
                recalls_b = [recs_b[v].get("recall_with_sentinel", 0) for v in common]
                comp_key = f"{ma}_vs_{mb}_{domain}_{split}"
                comparisons[comp_key] = paired_comparison(recalls_a, recalls_b)
                comparisons[comp_key]["n_versions"] = len(common)
                comparisons[comp_key]["domain"] = domain

    return comparisons
