"""Ablation studies for Phase 4.

Provides functions for:
  - Feature group ablation (train with each group removed)
  - Sentinel fraction sweep (re-evaluate at different sentinel sizes)
  - Magnitude threshold sweep (re-evaluate at different gating thresholds)
  - Change complexity breakdown (metrics by number of distinct change types)
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase2.evaluator import (
    GroundTruth,
    VersionMetrics,
    _version_to_category,
    aggregate_results,
    list_version_ids,
    load_base_prompt,
    load_ground_truth,
    load_test_cases,
    load_version_prompt,
)
from src.phase3.dataset import build_dataset, load_sensitivity_profiles
from src.phase3.learned_selector import select_tests_learned
from src.phase3.trainer import (
    _split_embeddings,
    apply_pca,
    compute_metrics,
    fit_pca,
    oof_predictions_lgb,
    oof_predictions_xgb,
    oof_predictions_logreg,
    prepare_features,
    save_model,
    train_lightgbm,
    train_logreg,
    train_xgboost,
    tune_threshold,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Column group definitions
# ---------------------------------------------------------------------------

def _column_groups(columns: list[str]) -> dict[str, list[str]]:
    """Classify columns into named feature groups."""
    groups: dict[str, list[str]] = {
        "change": [],
        "test": [],
        "pairwise": [],
        "embeddings": [],
        "ablation": [],
    }
    pairwise_names = {
        "cosine_similarity", "key_overlap", "demo_overlap",
        "section_match", "rule_based_prediction",
    }
    for c in columns:
        if c.startswith(("change_emb_", "test_emb_")):
            groups["embeddings"].append(c)
        elif c.startswith("ablation_"):
            groups["ablation"].append(c)
        elif c in pairwise_names:
            groups["pairwise"].append(c)
        elif c.startswith(("unit_type_", "change_type_", "num_affected")) or c in ("magnitude", "is_compound"):
            groups["change"].append(c)
        else:
            groups["test"].append(c)
    return groups


# ---------------------------------------------------------------------------
# Feature group ablation
# ---------------------------------------------------------------------------

def feature_group_ablation(
    domain: str,
    model_type: str = "lightgbm",
    *,
    pca_components: int = 24,
    n_trials: int = 30,
) -> list[dict]:
    """Train and evaluate with each feature group removed.

    Uses the temporal split (v01-v50 train, v51-v70 test).

    Returns a list of dicts, one per ablation configuration, containing
    the config name, recall, call_reduction, and deltas from the full model.
    """
    X_all, y_all, meta_all = build_dataset(domain, split="all")

    train_mask = meta_all["version_id"].apply(
        lambda v: int(re.search(r"\d+", v).group()) <= 50
    )
    test_mask = ~train_mask

    X_tr_full = X_all[train_mask].reset_index(drop=True)
    y_tr = y_all[train_mask].values
    meta_tr = meta_all[train_mask].reset_index(drop=True)
    X_te_full = X_all[test_mask].reset_index(drop=True)
    y_te = y_all[test_mask].values

    groups = _column_groups(list(X_tr_full.columns))

    ablation_configs = [
        ("full", []),
        ("no_change", groups["change"]),
        ("no_test", groups["test"]),
        ("no_pairwise", groups["pairwise"]),
        ("no_embeddings", groups["embeddings"]),
        ("no_ablation", groups["ablation"]),
        ("pairwise_and_emb_only", groups["change"] + groups["test"]),
    ]

    results: list[dict] = []
    full_recall: float | None = None

    for config_name, cols_to_drop in ablation_configs:
        keep_cols = [c for c in X_tr_full.columns if c not in cols_to_drop]
        if not keep_cols:
            logger.warning("Ablation %s drops all columns — skipping", config_name)
            continue

        X_tr = X_tr_full[keep_cols]
        X_te = X_te_full[keep_cols]

        X_sc_tr, c_emb_tr, t_emb_tr = _split_embeddings(X_tr)
        pca_n = min(pca_components, max(1, c_emb_tr.shape[1]), c_emb_tr.shape[0])
        change_pca, test_pca = fit_pca(c_emb_tr, t_emb_tr, n_components=pca_n)
        X_combined_tr, feature_names = prepare_features(X_tr, change_pca, test_pca)

        X_sc_te, c_emb_te, t_emb_te = _split_embeddings(X_te)
        X_combined_te = apply_pca(X_sc_te, c_emb_te, t_emb_te, change_pca, test_pca)

        groups_tr = meta_tr["version_id"].factorize()[0]

        if model_type == "lightgbm":
            model, params, n_boost = train_lightgbm(
                X_combined_tr, y_tr, feature_names,
                groups=groups_tr, tune=True, n_trials=n_trials,
            )
            probas = model.predict(X_combined_te)
            oof = oof_predictions_lgb(
                X_combined_tr, y_tr, groups_tr, feature_names, params, n_boost,
            )
        elif model_type == "xgboost":
            import xgboost as xgb
            model, params, n_boost = train_xgboost(
                X_combined_tr, y_tr, feature_names,
                groups=groups_tr, tune=True, n_trials=n_trials,
            )
            dtest = xgb.DMatrix(X_combined_te, feature_names=feature_names)
            probas = model.predict(dtest)
            oof = oof_predictions_xgb(
                X_combined_tr, y_tr, groups_tr, feature_names, params, n_boost,
            )
        else:
            model, scaler = train_logreg(X_combined_tr, y_tr, groups=groups_tr)
            X_te_scaled = scaler.transform(X_combined_te)
            probas = model.predict_proba(X_te_scaled)[:, 1]
            oof = oof_predictions_logreg(X_combined_tr, y_tr, groups_tr, model.C)

        threshold = tune_threshold(y_tr, oof, min_recall=0.95)
        metrics = compute_metrics(y_te, probas, threshold)
        recall = metrics["recall"]
        cr = metrics["call_reduction"]

        if full_recall is None:
            full_recall = recall
            full_cr = cr

        results.append({
            "config": config_name,
            "recall": recall,
            "call_reduction": cr,
            "threshold": round(threshold, 4),
            "auroc": metrics.get("auroc"),
            "auprc": metrics.get("auprc"),
            "delta_recall": round(recall - (full_recall or 0), 4),
            "delta_cr": round(cr - (full_cr if full_recall is not None else 0), 4),
        })
        logger.info(
            "Ablation %s: recall=%.4f cr=%.4f (Δrecall=%+.4f)",
            config_name, recall, cr, results[-1]["delta_recall"],
        )

    return results


# ---------------------------------------------------------------------------
# Sentinel fraction sweep
# ---------------------------------------------------------------------------

def sentinel_fraction_sweep(
    domain: str,
    model_tag: str,
    *,
    fractions: list[float] | None = None,
    version_filter: str = "temporal_test",
) -> list[dict]:
    """Re-evaluate a pre-trained model at different sentinel fractions.

    Uses the full selector pipeline (classify → tag → predict → sentinel).
    """
    if fractions is None:
        fractions = [0.0, 0.05, 0.10, 0.15, 0.20]

    model_path = _MODELS_DIR / f"{model_tag}.pkl"
    if not model_path.exists():
        logger.warning("Model not found: %s", model_path)
        return []

    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    gt = load_ground_truth(domain)
    sensitivity_profiles = load_sensitivity_profiles(domain)

    version_ids = list_version_ids(domain)
    if version_filter == "temporal_test":
        version_ids = [v for v in version_ids if int(re.search(r"\d+", v).group()) > 50]

    results: list[dict] = []
    for frac in fractions:
        recalls, reductions, sentinel_hits, misses_count = [], [], [], []
        for vid in version_ids:
            version_prompt = load_version_prompt(domain, vid)
            res = select_tests_learned(
                base_prompt, version_prompt, test_cases,
                model_path=model_path,
                sentinel_fraction=frac,
                sentinel_seed=42,
                sensitivity_profiles=sensitivity_profiles,
            )
            impacted = gt.impacted_by_version.get(vid, set())
            n_i = len(impacted)
            hit = impacted & res.selected_ids
            recall = len(hit) / n_i if n_i > 0 else 1.0
            recalls.append(recall)
            reductions.append(res.call_reduction)

            pred_hit = impacted & res.predicted_ids
            sent_hit = impacted & res.sentinel_ids
            if n_i > 0 and len(pred_hit) < n_i:
                sentinel_hits.append(len(sent_hit) > 0)
                misses_count.append(n_i - len(hit))

        catch_rate = (
            sum(sentinel_hits) / len(sentinel_hits)
            if sentinel_hits else None
        )
        n_escalated = sum(1 for r in recalls if r < 1.0 - 1e-9)
        escalation_rate = n_escalated / max(len(version_ids), 1)

        results.append({
            "sentinel_fraction": frac,
            "mean_recall": round(float(np.mean(recalls)), 4),
            "mean_call_reduction": round(float(np.mean(reductions)), 4),
            "sentinel_catch_rate": round(catch_rate, 4) if catch_rate is not None else None,
            "escalation_rate": round(escalation_rate, 4),
            "n_versions_with_misses": len(sentinel_hits),
            "total_misses": int(sum(misses_count)),
        })
    return results


# ---------------------------------------------------------------------------
# Magnitude threshold sweep
# ---------------------------------------------------------------------------

def magnitude_threshold_sweep(
    domain: str,
    model_tag: str,
    *,
    thresholds: list[float] | None = None,
    version_filter: str = "temporal_test",
) -> list[dict]:
    """Re-evaluate at different magnitude gating thresholds."""
    if thresholds is None:
        thresholds = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]

    model_path = _MODELS_DIR / f"{model_tag}.pkl"
    if not model_path.exists():
        logger.warning("Model not found: %s", model_path)
        return []

    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    gt = load_ground_truth(domain)
    sensitivity_profiles = load_sensitivity_profiles(domain)

    version_ids = list_version_ids(domain)
    if version_filter == "temporal_test":
        version_ids = [v for v in version_ids if int(re.search(r"\d+", v).group()) > 50]

    results: list[dict] = []
    for mag_thresh in thresholds:
        recalls, reductions, for_rates = [], [], []
        for vid in version_ids:
            version_prompt = load_version_prompt(domain, vid)
            res = select_tests_learned(
                base_prompt, version_prompt, test_cases,
                model_path=model_path,
                magnitude_threshold=mag_thresh,
                sentinel_fraction=0.10,
                sentinel_seed=42,
                sensitivity_profiles=sensitivity_profiles,
            )
            impacted = gt.impacted_by_version.get(vid, set())
            n_i = len(impacted)
            hit = impacted & res.selected_ids
            recall = len(hit) / n_i if n_i > 0 else 1.0
            missed = n_i - len(hit)
            not_selected = res.total_tests - len(res.selected_ids)
            for_rate = missed / not_selected if not_selected > 0 else 0.0
            recalls.append(recall)
            reductions.append(res.call_reduction)
            for_rates.append(for_rate)

        results.append({
            "magnitude_threshold": mag_thresh,
            "mean_recall": round(float(np.mean(recalls)), 4),
            "mean_call_reduction": round(float(np.mean(reductions)), 4),
            "mean_for": round(float(np.mean(for_rates)), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Change complexity breakdown
# ---------------------------------------------------------------------------

def change_complexity_breakdown(
    domain: str,
    model_tag: str,
    *,
    version_filter: str = "all",
) -> list[dict]:
    """Break down metrics by number of distinct change unit_types per version."""
    model_path = _MODELS_DIR / f"{model_tag}.pkl"
    if not model_path.exists():
        logger.warning("Model not found: %s", model_path)
        return []

    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    gt = load_ground_truth(domain)
    sensitivity_profiles = load_sensitivity_profiles(domain)

    version_ids = list_version_ids(domain)
    if version_filter == "temporal_test":
        version_ids = [v for v in version_ids if int(re.search(r"\d+", v).group()) > 50]

    buckets: dict[str, list[dict]] = {"1_type": [], "2_types": [], "3+_types": []}

    for vid in version_ids:
        version_prompt = load_version_prompt(domain, vid)
        res = select_tests_learned(
            base_prompt, version_prompt, test_cases,
            model_path=model_path,
            sentinel_fraction=0.10,
            sentinel_seed=42,
            sensitivity_profiles=sensitivity_profiles,
        )
        n_types = len({c.change.unit_type for c in res.changes})
        impacted = gt.impacted_by_version.get(vid, set())
        n_i = len(impacted)
        hit = impacted & res.selected_ids
        recall = len(hit) / n_i if n_i > 0 else 1.0

        entry = {"recall": recall, "call_reduction": res.call_reduction, "n_impacted": n_i}
        if n_types <= 1:
            buckets["1_type"].append(entry)
        elif n_types == 2:
            buckets["2_types"].append(entry)
        else:
            buckets["3+_types"].append(entry)

    results = []
    for bucket_name, entries in buckets.items():
        if not entries:
            results.append({"complexity": bucket_name, "count": 0})
            continue
        results.append({
            "complexity": bucket_name,
            "count": len(entries),
            "mean_recall": round(float(np.mean([e["recall"] for e in entries])), 4),
            "mean_call_reduction": round(float(np.mean([e["call_reduction"] for e in entries])), 4),
            "mean_n_impacted": round(float(np.mean([e["n_impacted"] for e in entries])), 1),
        })
    return results
