"""Report builder: collect all Phase 2/3 results into unified tables.

Scans ``results/selection/`` and ``results/phase3/`` for evaluation JSONs,
normalises them into a common schema, and produces the eight analysis tables
defined in PHASE4.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase4.analysis import (
    bootstrap_ci,
    metric_summary,
    paired_comparison,
    paired_recall_series,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _PROJECT_ROOT / "results"

_DOMAINS = ["domain_a", "domain_b", "domain_c"]
_MODEL_TYPES = ["lightgbm", "xgboost", "logreg"]


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_all_results() -> dict:
    """Scan result directories and build a unified result store.

    Returns a dict keyed by descriptive tag with aggregated metrics.
    Structure: ``{tag: {domain, model_type, split, aggregate: {...}, ...}}``
    """
    store: dict[str, dict] = {}

    selection_dir = _RESULTS_DIR / "selection"
    phase3_dir = _RESULTS_DIR / "phase3"

    for domain in _DOMAINS:
        rule_path = selection_dir / f"metrics_{domain}.json"
        data = _safe_load_json(rule_path)
        if data:
            tag = f"rule_based_{domain}"
            store[tag] = {
                "domain": domain,
                "model_type": "rule_based",
                "split": "all_versions",
                **data,
            }

    for path in sorted(phase3_dir.glob("*.json")) if phase3_dir.exists() else []:
        if path.name.startswith("detail_") or path.name.startswith("threshold_"):
            continue
        data = _safe_load_json(path)
        if data and isinstance(data, dict):
            store[path.stem] = data

    return store


def collect_detail_records() -> dict[str, list[dict]]:
    """Collect per-version detail JSONL records from Phase 2 and Phase 3.

    Returns ``{tag: [per-version dicts]}``
    """
    records: dict[str, list[dict]] = {}
    selection_dir = _RESULTS_DIR / "selection"
    phase3_dir = _RESULTS_DIR / "phase3"

    for domain in _DOMAINS:
        detail_path = selection_dir / f"detail_{domain}.jsonl"
        if detail_path.exists():
            with open(detail_path) as f:
                recs = [json.loads(line) for line in f if line.strip()]
            records[f"rule_based_{domain}"] = recs

    if phase3_dir.exists():
        for path in sorted(phase3_dir.glob("detail_*.jsonl")):
            raw_tag = path.stem.replace("detail_", "")
            tag = raw_tag.replace("eval_", "")
            with open(path) as f:
                recs = [json.loads(line) for line in f if line.strip()]
            records[tag] = recs

    return records


# ---------------------------------------------------------------------------
# T1: Main results table
# ---------------------------------------------------------------------------

def build_main_results_table(all_results: dict) -> pd.DataFrame:
    """Build a DataFrame with one row per (domain, model, split) config.

    Columns: domain, model_type, split, recall_predictor, recall_sentinel,
    call_reduction, FOR, auroc, sentinel_catch_rate, n_versions.
    """
    rows: list[dict] = []
    for tag, data in sorted(all_results.items()):
        if not isinstance(data, dict):
            continue
        agg = data.get("aggregate", {})
        if not agg or not isinstance(agg, dict):
            continue
        rows.append({
            "tag": tag,
            "domain": data.get("domain", ""),
            "model_type": data.get("model_type", ""),
            "split": data.get("evaluation_mode", data.get("split", "")),
            "recall_predictor": agg.get("mean_recall_predictor"),
            "recall_with_sentinel": agg.get("mean_recall_with_sentinel"),
            "call_reduction": agg.get("mean_call_reduction"),
            "effective_recall": agg.get("mean_effective_recall"),
            "effective_call_reduction": agg.get("mean_effective_call_reduction"),
            "false_omission_rate": agg.get("mean_false_omission_rate"),
            "sentinel_catch_rate": agg.get("sentinel_catch_rate"),
            "n_versions": agg.get("total_versions", agg.get("count")),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T2: Learned vs. rule-based comparison
# ---------------------------------------------------------------------------

def _find_detail_records(
    detail_records: dict[str, list[dict]],
    domain: str,
    model_type: str,
    split_tag: str,
) -> list[dict]:
    """Find detail records matching a (domain, model_type, split) combo.

    Handles multiple naming conventions from Phase 3 eval scripts:
    - ``{model_type}_{domain}_{split_tag}``
    - ``{domain}_{model_type}_{domain}_{split_tag}_*``
    - ``kfold_{domain}_{model_type}_*``
    """
    exact = f"{model_type}_{domain}_{split_tag}"
    if exact in detail_records:
        return detail_records[exact]

    for key, recs in detail_records.items():
        if model_type in key and domain in key and split_tag in key:
            return recs

    if split_tag == "kfold":
        for key, recs in detail_records.items():
            if "kfold" in key and model_type in key and domain in key:
                return recs

    return []


def build_learned_vs_rule_table(detail_records: dict) -> pd.DataFrame:
    """Head-to-head comparison of learned (LightGBM temporal) vs rule-based.

    For each domain, pairs per-version recalls and runs a significance test.
    """
    rows: list[dict] = []
    for domain in _DOMAINS:
        rule_key = f"rule_based_{domain}"
        rule_recs = detail_records.get(rule_key, [])
        if not rule_recs:
            continue
        rule_by_vid = {r["version_id"]: r for r in rule_recs}

        for model_type in _MODEL_TYPES:
            for split_tag in ["temporal", "kfold"]:
                learned_recs = _find_detail_records(
                    detail_records, domain, model_type, split_tag,
                )
                if not learned_recs:
                    continue
                learned_by_vid = {r["version_id"]: r for r in learned_recs}

                common = sorted(set(rule_by_vid) & set(learned_by_vid))
                if len(common) < 3:
                    continue

                rule_recalls, learned_recalls = paired_recall_series(
                    rule_by_vid, learned_by_vid, common,
                )

                comp = paired_comparison(learned_recalls, rule_recalls)
                rule_summary = metric_summary(rule_recalls)
                learned_summary = metric_summary(learned_recalls)

                rows.append({
                    "domain": domain,
                    "model_type": model_type,
                    "split": split_tag,
                    "n_versions": len(common),
                    "rule_recall_mean": rule_summary["mean"],
                    "rule_recall_ci": f"[{rule_summary['ci_95_lo']}, {rule_summary['ci_95_hi']}]",
                    "learned_recall_mean": learned_summary["mean"],
                    "learned_recall_ci": f"[{learned_summary['ci_95_lo']}, {learned_summary['ci_95_hi']}]",
                    "recall_diff": comp["mean_diff"],
                    "p_value": comp["p_value"],
                    "effect_size": comp["effect_size_cohens_d"],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T5: Cross-domain transfer matrix
# ---------------------------------------------------------------------------

def build_cross_domain_matrix(all_results: dict) -> pd.DataFrame:
    """3×3 matrix of cross-domain recall (rows=train, cols=test)."""
    matrix: dict[str, dict[str, float | None]] = {}
    for d in _DOMAINS:
        matrix[d] = {dd: None for dd in _DOMAINS}

    for tag, data in all_results.items():
        agg = data.get("aggregate", {})
        if not agg:
            continue
        if "cross" not in tag:
            continue
        recall = agg.get("mean_effective_recall")
        if recall is None:
            recall = agg.get("mean_recall_with_sentinel", agg.get("mean_recall_predictor"))
        parts = tag.split("_")
        for i, p in enumerate(parts):
            if p == "cross":
                pair = parts[i + 1] if i + 1 < len(parts) else ""
                if len(pair) == 2:
                    domain_map = {"a": "domain_a", "b": "domain_b", "c": "domain_c"}
                    src = domain_map.get(pair[0])
                    tgt = domain_map.get(pair[1])
                    if src and tgt and recall is not None:
                        matrix[src][tgt] = round(recall, 4)
                break

    rows = []
    for src in _DOMAINS:
        row = {"train_domain": src}
        for tgt in _DOMAINS:
            row[tgt] = matrix[src][tgt]
        rows.append(row)
    return pd.DataFrame(rows).set_index("train_domain")


# ---------------------------------------------------------------------------
# T7: By change type
# ---------------------------------------------------------------------------

def build_change_type_table(all_results: dict) -> pd.DataFrame:
    """Aggregate recall and call_reduction per change unit_type across configs."""
    rows: list[dict] = []

    for tag, data in all_results.items():
        if not isinstance(data, dict):
            continue
        by_ct = data.get("by_change_type", {})
        if not by_ct:
            continue
        domain = data.get("domain", "")
        model_type = data.get("model_type", "")
        split = data.get("evaluation_mode", data.get("split", ""))
        for change_type, metrics in by_ct.items():
            rows.append({
                "tag": tag,
                "domain": domain,
                "model_type": model_type,
                "split": split,
                "change_type": change_type,
                "count": metrics.get("count", 0),
                "recall_predictor": metrics.get("mean_recall_predictor"),
                "recall_with_sentinel": metrics.get("mean_recall_with_sentinel"),
                "call_reduction": metrics.get("mean_call_reduction"),
                "effective_recall": metrics.get("mean_effective_recall"),
                "effective_call_reduction": metrics.get("mean_effective_call_reduction"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

def export_json(data: dict | list | pd.DataFrame, path: Path) -> None:
    """Write data to JSON, handling DataFrames and numpy types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logger.info("Wrote %s", path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def export_latex_table(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Export a DataFrame as a LaTeX table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=kwargs.pop("index", True), float_format="%.4f", **kwargs)
    path.write_text(latex)
    logger.info("Wrote LaTeX table %s", path)
