"""CLI for Phase 3 learned-selector evaluation.

Evaluation configurations (all use honest held-out metrics):
  - Temporal: pre-trained models tested on held-out v51-v70 (evaluate-all)
  - Cross-domain: pre-trained models tested on unseen domain (evaluate-all)
  - Single-domain k-fold CV: fresh models per fold (kfold-evaluate-all)
  - Combined k-fold CV: fresh models trained on target+other domain (kfold-evaluate-all)

In-sample evaluations (all-versions, combined without k-fold) have been
removed; k-fold CV fully replaces them.

Usage:
    python -m scripts.run_phase3_eval evaluate-all
    python -m scripts.run_phase3_eval kfold-evaluate-all
    python -m scripts.run_phase3_eval kfold-evaluate-combined --target-domain domain_a
    python -m scripts.run_phase3_eval compare
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from src.phase1.models import TestCase
from src.phase2.evaluator import (
    GroundTruth,
    VersionMetrics,
    _version_to_category,
    aggregate_results,
    compute_effective_metrics,
    effective_means_from_aggregate,
    list_version_ids,
    load_base_prompt,
    load_ground_truth,
    load_test_cases,
    load_version_prompt,
)
from src.phase2.selector import SelectionResult
from src.phase3.dataset import build_dataset, load_sensitivity_profiles
from src.phase3.learned_selector import select_tests_learned
from src.phase3.trainer import (
    _split_embeddings,
    fit_pca,
    load_model,
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

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / "models"
_RESULTS_DIR = _PROJECT_ROOT / "results" / "phase3"


def _echo_learned_aggregate(a: dict) -> None:
    typer.echo(f"  Recall (predictor): {a['mean_recall_predictor']:.3f}")
    typer.echo(f"  Recall (w/ sentinel): {a['mean_recall_with_sentinel']:.3f}")
    er, ecr = effective_means_from_aggregate(a)
    typer.echo(f"  Effective recall: {er:.3f}")
    typer.echo(f"  Effective call reduction: {ecr:.3f}")
    typer.echo(f"  Call reduction: {a['mean_call_reduction']:.3f}")
    typer.echo(f"  False omission rate: {a['mean_false_omission_rate']:.3f}")


# ---------------------------------------------------------------------------
# Per-version evaluation (mirrors Phase 2's evaluate_version)
# ---------------------------------------------------------------------------

def _evaluate_version_learned(
    domain: str,
    version_id: str,
    base_prompt: str,
    test_cases: list[TestCase],
    gt: GroundTruth,
    *,
    model_path: Path,
    threshold: float | None = None,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int | None = 42,
    sensitivity_profiles: dict | None = None,
) -> VersionMetrics:
    """Run the learned selector on one version and compare against ground truth."""
    version_prompt = load_version_prompt(domain, version_id)
    res: SelectionResult = select_tests_learned(
        base_prompt,
        version_prompt,
        test_cases,
        model_path=model_path,
        threshold=threshold,
        sentinel_fraction=sentinel_fraction,
        magnitude_threshold=magnitude_threshold,
        sentinel_seed=sentinel_seed,
        sensitivity_profiles=sensitivity_profiles,
    )

    impacted: set[str] = gt.impacted_by_version.get(version_id, set())
    n_impacted = len(impacted)

    predicted_hit = impacted & res.predicted_ids
    selected_hit = impacted & res.selected_ids
    missed = impacted - res.selected_ids
    sentinel_caught = len(impacted & res.sentinel_ids) > 0

    recall_pred = len(predicted_hit) / n_impacted if n_impacted > 0 else 1.0
    recall_sel = len(selected_hit) / n_impacted if n_impacted > 0 else 1.0
    not_selected = res.total_tests - len(res.selected_ids)
    for_rate = len(missed) / not_selected if not_selected > 0 else 0.0

    eff_recall, eff_call_reduction = compute_effective_metrics(
        sentinel_hit=sentinel_caught,
        recall_with_sentinel=recall_sel,
        call_reduction=res.call_reduction,
    )

    changes_info = [
        {
            "unit_type": c.change.unit_type,
            "change_type": c.change.change_type,
            "magnitude": c.change.magnitude,
            "affected_keys": c.affected_keys,
            "affected_demo_labels": c.affected_demo_labels,
        }
        for c in res.changes
    ]
    mag_max = max((c.change.magnitude for c in res.changes), default=0.0)

    return VersionMetrics(
        version_id=version_id,
        domain=domain,
        num_impacted=n_impacted,
        num_predicted=len(res.predicted_ids),
        num_sentinel=len(res.sentinel_ids),
        num_selected=len(res.selected_ids),
        total_tests=res.total_tests,
        recall_predictor=recall_pred,
        recall_with_sentinel=recall_sel,
        call_reduction=res.call_reduction,
        false_omission_rate=for_rate,
        false_omissions=sorted(missed),
        sentinel_hit=sentinel_caught,
        changes_detected=changes_info,
        change_types_gt=gt.change_types_by_version.get(version_id, []),
        magnitude_max=mag_max,
        mutation_category=_version_to_category(version_id),
        effective_recall=eff_recall,
        effective_call_reduction=eff_call_reduction,
    )


# ---------------------------------------------------------------------------
# Full evaluation driver
# ---------------------------------------------------------------------------

def _run_evaluation(
    domain: str,
    model_tag: str,
    *,
    version_filter: str = "all",
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int = 42,
) -> tuple[list[VersionMetrics], dict]:
    """Run the learned selector evaluation for one domain + model config."""
    model_path = _MODELS_DIR / f"{model_tag}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = load_model(model_path)
    threshold = bundle["threshold"]

    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    gt = load_ground_truth(domain)
    sensitivity_profiles = load_sensitivity_profiles(domain)

    version_ids = list_version_ids(domain)
    if version_filter == "temporal_test":
        version_ids = [v for v in version_ids if int(re.search(r"\d+", v).group()) > 50]
    elif version_filter == "temporal_train":
        version_ids = [v for v in version_ids if int(re.search(r"\d+", v).group()) <= 50]

    all_metrics: list[VersionMetrics] = []
    for vid in tqdm(version_ids, desc=f"[{model_tag}] Evaluating {domain}"):
        vm = _evaluate_version_learned(
            domain, vid, base_prompt, test_cases, gt,
            model_path=model_path,
            threshold=threshold,
            sentinel_fraction=sentinel_fraction,
            magnitude_threshold=magnitude_threshold,
            sentinel_seed=sentinel_seed,
            sensitivity_profiles=sensitivity_profiles,
        )
        all_metrics.append(vm)

    agg = aggregate_results(all_metrics, test_cases, gt)
    agg["model_tag"] = model_tag
    agg["threshold"] = threshold
    agg["version_filter"] = version_filter

    return all_metrics, agg


def _write_detail_jsonl(
    per_version: list[VersionMetrics],
    path: Path,
) -> None:
    """Write per-version detail metrics to a JSONL file for Phase 4 analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for m in per_version:
            row = {
                "version_id": m.version_id,
                "changes": m.changes_detected,
                "predicted_count": m.num_predicted,
                "sentinel_count": m.num_sentinel,
                "selected_count": m.num_selected,
                "impacted_count": m.num_impacted,
                "recall_predictor": round(m.recall_predictor, 4),
                "recall_with_sentinel": round(m.recall_with_sentinel, 4),
                "call_reduction": round(m.call_reduction, 4),
                "false_omission_rate": round(m.false_omission_rate, 4),
                "false_omissions": m.false_omissions,
                "sentinel_hit": m.sentinel_hit,
                "effective_recall": round(m.effective_recall, 4),
                "effective_call_reduction": round(m.effective_call_reduction, 4),
                "magnitude_max": round(m.magnitude_max, 4),
                "mutation_category": m.mutation_category,
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote detail JSONL: %s", path)


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

def _threshold_sensitivity(
    domain: str,
    model_tag: str,
    thresholds: list[float],
    version_filter: str = "all",
    magnitude_threshold: float = 0.005,
) -> list[dict]:
    """Evaluate at multiple fixed thresholds to show the tradeoff."""
    model_path = _MODELS_DIR / f"{model_tag}.pkl"
    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    gt = load_ground_truth(domain)
    sensitivity_profiles = load_sensitivity_profiles(domain)
    version_ids = list_version_ids(domain)
    if version_filter == "temporal_test":
        version_ids = [v for v in version_ids if int(re.search(r"\d+", v).group()) > 50]
    elif version_filter == "temporal_train":
        version_ids = [v for v in version_ids if int(re.search(r"\d+", v).group()) <= 50]

    if not version_ids:
        return []

    results = []
    for tau in thresholds:
        recalls, reductions = [], []
        eff_recalls, eff_reductions = [], []
        for vid in version_ids:
            version_prompt = load_version_prompt(domain, vid)
            res = select_tests_learned(
                base_prompt, version_prompt, test_cases,
                model_path=model_path,
                threshold=tau,
                sentinel_fraction=0.10,
                sentinel_seed=42,
                magnitude_threshold=magnitude_threshold,
                sensitivity_profiles=sensitivity_profiles,
            )
            impacted = gt.impacted_by_version.get(vid, set())
            n_i = len(impacted)
            recall = len(impacted & res.selected_ids) / n_i if n_i > 0 else 1.0
            recalls.append(recall)
            reductions.append(res.call_reduction)
            sentinel_hit = len(impacted & res.sentinel_ids) > 0
            er, ecr = compute_effective_metrics(
                sentinel_hit=sentinel_hit,
                recall_with_sentinel=recall,
                call_reduction=res.call_reduction,
            )
            eff_recalls.append(er)
            eff_reductions.append(ecr)

        n = len(recalls)
        results.append({
            "threshold": tau,
            "mean_recall_with_sentinel": round(sum(recalls) / n, 4),
            "mean_call_reduction": round(sum(reductions) / n, 4),
            "mean_effective_recall": round(sum(eff_recalls) / n, 4),
            "mean_effective_call_reduction": round(sum(eff_reductions) / n, 4),
        })
    return results


# ---------------------------------------------------------------------------
# K-fold CV evaluation for "all" split (Option B)
# ---------------------------------------------------------------------------

def _kfold_evaluate(
    domain: str,
    model_type: str,
    *,
    n_folds: int = 5,
    pca_components: int = 24,
    n_trials: int = 50,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int = 42,
) -> tuple[list[VersionMetrics], dict]:
    """K-fold cross-validation at the version level for the 'all' split.

    Splits all versions into *n_folds* groups.  For each fold, trains a fresh
    model on the remaining versions, tunes the threshold via OOF on the
    training portion, and evaluates the full selector pipeline on the held-out
    versions.  Aggregates metrics across all folds for an honest
    generalization estimate.
    """
    import xgboost as xgb_lib

    X_all, y_all_s, meta_all = build_dataset(domain, split="all")
    y_all = y_all_s.values

    version_ids_sorted = sorted(meta_all["version_id"].unique().tolist())
    n_versions = len(version_ids_sorted)
    actual_folds = min(n_folds, n_versions)

    vid_to_fold: dict[str, int] = {}
    for idx, vid in enumerate(version_ids_sorted):
        vid_to_fold[vid] = idx % actual_folds

    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    gt = load_ground_truth(domain)
    sensitivity_profiles = load_sensitivity_profiles(domain)

    all_fold_metrics: list[VersionMetrics] = []

    for fold_idx in range(actual_folds):
        typer.echo(f"\n--- Fold {fold_idx + 1}/{actual_folds} ---")

        test_vids = {v for v, f in vid_to_fold.items() if f == fold_idx}
        train_vids = {v for v, f in vid_to_fold.items() if f != fold_idx}

        train_mask = meta_all["version_id"].isin(train_vids).values
        X_tr = X_all[train_mask].reset_index(drop=True)
        y_tr = y_all[train_mask]
        meta_tr = meta_all[train_mask].reset_index(drop=True)

        X_sc_tr, c_emb_tr, t_emb_tr = _split_embeddings(X_tr)
        pca_n = min(pca_components, c_emb_tr.shape[0], c_emb_tr.shape[1])
        change_pca, test_pca = fit_pca(c_emb_tr, t_emb_tr, n_components=pca_n)

        X_combined_tr, feature_names = prepare_features(X_tr, change_pca, test_pca)
        groups_tr = meta_tr["version_id"].factorize()[0]

        scaler = None
        if model_type == "lightgbm":
            model, lgb_params, n_boost = train_lightgbm(
                X_combined_tr, y_tr, feature_names,
                groups=groups_tr, tune=True, n_trials=n_trials,
            )
            oof_proba = oof_predictions_lgb(
                X_combined_tr, y_tr, groups_tr, feature_names, lgb_params, n_boost,
            )
        elif model_type == "xgboost":
            model, xgb_params, n_boost = train_xgboost(
                X_combined_tr, y_tr, feature_names,
                groups=groups_tr, tune=True, n_trials=n_trials,
            )
            oof_proba = oof_predictions_xgb(
                X_combined_tr, y_tr, groups_tr, feature_names, xgb_params, n_boost,
            )
        else:
            model, scaler = train_logreg(X_combined_tr, y_tr, groups=groups_tr)
            oof_proba = oof_predictions_logreg(
                X_combined_tr, y_tr, groups_tr, model.C,
            )

        threshold = tune_threshold(y_tr, oof_proba, min_recall=0.95)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save_model(
                tmp_path,
                model=model,
                change_pca=change_pca,
                test_pca=test_pca,
                threshold=threshold,
                feature_names=feature_names,
                scaler=scaler,
                metadata={"fold": fold_idx, "domain": domain, "model_type": model_type},
            )

            fold_test_vids = sorted(test_vids)
            typer.echo(f"  Train versions: {len(train_vids)}, Test versions: {len(fold_test_vids)}")
            for vid in tqdm(fold_test_vids, desc=f"  Fold {fold_idx+1} eval"):
                vm = _evaluate_version_learned(
                    domain, vid, base_prompt, test_cases, gt,
                    model_path=tmp_path,
                    threshold=threshold,
                    sentinel_fraction=sentinel_fraction,
                    magnitude_threshold=magnitude_threshold,
                    sentinel_seed=sentinel_seed,
                    sensitivity_profiles=sensitivity_profiles,
                )
                all_fold_metrics.append(vm)
        finally:
            tmp_path.unlink(missing_ok=True)

    agg = aggregate_results(all_fold_metrics, test_cases, gt)
    agg["model_type"] = model_type
    agg["evaluation_mode"] = "kfold_cv"
    agg["n_folds"] = actual_folds
    agg["domain"] = domain

    detail_tag = f"kfold_{domain}_{model_type}_{actual_folds}fold"
    _write_detail_jsonl(all_fold_metrics, _RESULTS_DIR / f"detail_{detail_tag}.jsonl")

    return all_fold_metrics, agg


# ---------------------------------------------------------------------------
# K-fold CV evaluation for combined (both domains) training
# ---------------------------------------------------------------------------

def _kfold_evaluate_combined(
    target_domain: str,
    model_type: str,
    *,
    n_folds: int = 5,
    pca_components: int = 24,
    n_trials: int = 50,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int = 42,
) -> tuple[list[VersionMetrics], dict]:
    """K-fold CV with combined-domain training.

    Splits *target_domain*'s versions into *n_folds* groups.  For each fold
    the training set is (target domain train folds) + (ALL versions from the
    other domain).  Evaluation is on the held-out target-domain versions only.

    This answers: "Does pooling data from both domains improve performance
    on the target domain?" — with honest held-out metrics.
    """
    import xgboost as xgb_lib

    all_domains = ["domain_a", "domain_b", "domain_c"]
    other_domains = [d for d in all_domains if d != target_domain]

    X_target, y_target_s, meta_target = build_dataset(target_domain, split="all")
    y_target = y_target_s.values

    frames_X_other, frames_y_other, frames_m_other = [], [], []
    for od in other_domains:
        X_o, y_o, m_o = build_dataset(od, split="all")
        frames_X_other.append(X_o)
        frames_y_other.append(y_o.values)
        frames_m_other.append(m_o)
    X_other = pd.concat(frames_X_other, ignore_index=True)
    y_other = np.concatenate(frames_y_other)
    meta_other = pd.concat(frames_m_other, ignore_index=True)

    target_vids_sorted = sorted(meta_target["version_id"].unique().tolist())
    n_target_versions = len(target_vids_sorted)
    actual_folds = min(n_folds, n_target_versions)

    vid_to_fold: dict[str, int] = {}
    for idx, vid in enumerate(target_vids_sorted):
        vid_to_fold[vid] = idx % actual_folds

    base_prompt = load_base_prompt(target_domain)
    test_cases = load_test_cases(target_domain)
    gt = load_ground_truth(target_domain)
    sensitivity_profiles = load_sensitivity_profiles(target_domain)

    all_fold_metrics: list[VersionMetrics] = []

    for fold_idx in range(actual_folds):
        typer.echo(f"\n--- Combined fold {fold_idx + 1}/{actual_folds} (target={target_domain}) ---")

        test_vids = {v for v, f in vid_to_fold.items() if f == fold_idx}
        train_vids = {v for v, f in vid_to_fold.items() if f != fold_idx}

        train_mask = meta_target["version_id"].isin(train_vids).values
        X_target_tr = X_target[train_mask].reset_index(drop=True)
        y_target_tr = y_target[train_mask]
        meta_target_tr = meta_target[train_mask].reset_index(drop=True)

        X_tr = pd.concat([X_target_tr, X_other], ignore_index=True)
        y_tr = np.concatenate([y_target_tr, y_other])
        meta_tr = pd.concat([meta_target_tr, meta_other], ignore_index=True)

        X_sc_tr, c_emb_tr, t_emb_tr = _split_embeddings(X_tr)
        pca_n = min(pca_components, c_emb_tr.shape[0], c_emb_tr.shape[1])
        change_pca, test_pca = fit_pca(c_emb_tr, t_emb_tr, n_components=pca_n)

        X_combined_tr, feature_names = prepare_features(X_tr, change_pca, test_pca)
        composite_ids = meta_tr["domain"] + "_" + meta_tr["version_id"]
        groups_tr = composite_ids.factorize()[0]

        scaler = None
        if model_type == "lightgbm":
            model, lgb_params, n_boost = train_lightgbm(
                X_combined_tr, y_tr, feature_names,
                groups=groups_tr, tune=True, n_trials=n_trials,
            )
            oof_proba = oof_predictions_lgb(
                X_combined_tr, y_tr, groups_tr, feature_names, lgb_params, n_boost,
            )
        elif model_type == "xgboost":
            model, xgb_params, n_boost = train_xgboost(
                X_combined_tr, y_tr, feature_names,
                groups=groups_tr, tune=True, n_trials=n_trials,
            )
            oof_proba = oof_predictions_xgb(
                X_combined_tr, y_tr, groups_tr, feature_names, xgb_params, n_boost,
            )
        else:
            model, scaler = train_logreg(X_combined_tr, y_tr, groups=groups_tr)
            oof_proba = oof_predictions_logreg(
                X_combined_tr, y_tr, groups_tr, model.C,
            )

        threshold = tune_threshold(y_tr, oof_proba, min_recall=0.95)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save_model(
                tmp_path,
                model=model,
                change_pca=change_pca,
                test_pca=test_pca,
                threshold=threshold,
                feature_names=feature_names,
                scaler=scaler,
                metadata={
                    "fold": fold_idx,
                    "target_domain": target_domain,
                    "model_type": model_type,
                    "mode": "combined_kfold",
                },
            )

            fold_test_vids = sorted(test_vids)
            other_ver_count = len(meta_other["domain"].astype(str) + "_" + meta_other["version_id"].astype(str))
            typer.echo(
                f"  Train: {len(train_vids)} {target_domain} + "
                f"{other_ver_count} other-domain rows, "
                f"Test: {len(fold_test_vids)} {target_domain} versions"
            )
            for vid in tqdm(fold_test_vids, desc=f"  Combined fold {fold_idx+1} eval"):
                vm = _evaluate_version_learned(
                    target_domain, vid, base_prompt, test_cases, gt,
                    model_path=tmp_path,
                    threshold=threshold,
                    sentinel_fraction=sentinel_fraction,
                    magnitude_threshold=magnitude_threshold,
                    sentinel_seed=sentinel_seed,
                    sensitivity_profiles=sensitivity_profiles,
                )
                all_fold_metrics.append(vm)
        finally:
            tmp_path.unlink(missing_ok=True)

    agg = aggregate_results(all_fold_metrics, test_cases, gt)
    agg["model_type"] = model_type
    agg["evaluation_mode"] = "kfold_cv_combined"
    agg["n_folds"] = actual_folds
    agg["target_domain"] = target_domain

    detail_tag = f"kfold_combined_{target_domain}_{model_type}_{actual_folds}fold"
    _write_detail_jsonl(all_fold_metrics, _RESULTS_DIR / f"detail_{detail_tag}.jsonl")

    return all_fold_metrics, agg


# ---------------------------------------------------------------------------
# Comparison builder
# ---------------------------------------------------------------------------

def _build_comparison(domains: list[str] = None) -> dict:
    """Build side-by-side comparison across all available result files."""
    if domains is None:
        domains = ["domain_a", "domain_b", "domain_c"]

    phase2_dir = _PROJECT_ROOT / "results" / "selection"

    comparison: dict = {}
    for domain in domains:
        comparison[domain] = {}

        p2_path = phase2_dir / f"metrics_{domain}.json"
        if p2_path.exists():
            with open(p2_path) as f:
                p2 = json.load(f)
            p2a = p2.get("aggregate", {})
            er2, ecr2 = effective_means_from_aggregate(p2a)
            comparison[domain]["rule_based"] = {
                "recall_predictor": p2a.get("mean_recall_predictor"),
                "recall_with_sentinel": p2a.get("mean_recall_with_sentinel"),
                "call_reduction": p2a.get("mean_call_reduction"),
                "effective_recall": er2,
                "effective_call_reduction": ecr2,
            }

        for p3_file in sorted(_RESULTS_DIR.glob(f"eval_{domain}_*.json")):
            with open(p3_file) as f:
                p3 = json.load(f)
            tag = p3.get("model_tag", p3_file.stem)
            p3a = p3.get("aggregate", {})
            er3, ecr3 = effective_means_from_aggregate(p3a)
            comparison[domain][tag] = {
                "recall_predictor": p3a.get("mean_recall_predictor"),
                "recall_with_sentinel": p3a.get("mean_recall_with_sentinel"),
                "call_reduction": p3a.get("mean_call_reduction"),
                "effective_recall": er3,
                "effective_call_reduction": ecr3,
                "threshold": p3.get("threshold"),
                "version_filter": p3.get("version_filter"),
            }

    return comparison


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def evaluate(
    domain: str = typer.Option(..., help="domain_a | domain_b | domain_c"),
    model_tag: str = typer.Option(..., help="Model tag, e.g. lightgbm_domain_a_temporal"),
    version_filter: str = typer.Option("all", help="all | temporal_train | temporal_test"),
) -> None:
    """Evaluate a single learned model on one domain."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_version, agg = _run_evaluation(domain, model_tag, version_filter=version_filter)

    out_tag = f"eval_{domain}_{model_tag}"
    if version_filter != "all":
        out_tag += f"_{version_filter}"

    agg_path = _RESULTS_DIR / f"{out_tag}.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    _write_detail_jsonl(per_version, _RESULTS_DIR / f"detail_{out_tag}.jsonl")
    typer.echo(f"Wrote {agg_path}")

    a = agg["aggregate"]
    typer.echo(f"\n{model_tag} on {domain} ({version_filter}):")
    _echo_learned_aggregate(a)


@app.command()
def evaluate_all() -> None:
    """Evaluate all trained models on their matching domains and splits."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        # Temporal: train v01-v50, test v51-v70
        ("domain_a", "lightgbm_domain_a_temporal", "temporal_test"),
        ("domain_a", "xgboost_domain_a_temporal", "temporal_test"),
        ("domain_a", "logreg_domain_a_temporal", "temporal_test"),
        ("domain_b", "lightgbm_domain_b_temporal", "temporal_test"),
        ("domain_b", "xgboost_domain_b_temporal", "temporal_test"),
        ("domain_b", "logreg_domain_b_temporal", "temporal_test"),
        ("domain_c", "lightgbm_domain_c_temporal", "temporal_test"),
        ("domain_c", "xgboost_domain_c_temporal", "temporal_test"),
        ("domain_c", "logreg_domain_c_temporal", "temporal_test"),
        # Cross-domain: train on one domain, test on another
        ("domain_b", "lightgbm_domain_a_cross_ab", "all"),
        ("domain_b", "xgboost_domain_a_cross_ab", "all"),
        ("domain_b", "logreg_domain_a_cross_ab", "all"),
        ("domain_a", "lightgbm_domain_b_cross_ba", "all"),
        ("domain_a", "xgboost_domain_b_cross_ba", "all"),
        ("domain_a", "logreg_domain_b_cross_ba", "all"),
        ("domain_c", "lightgbm_domain_a_cross_ac", "all"),
        ("domain_c", "xgboost_domain_a_cross_ac", "all"),
        ("domain_c", "logreg_domain_a_cross_ac", "all"),
        ("domain_a", "lightgbm_domain_c_cross_ca", "all"),
        ("domain_a", "xgboost_domain_c_cross_ca", "all"),
        ("domain_a", "logreg_domain_c_cross_ca", "all"),
        ("domain_c", "lightgbm_domain_b_cross_bc", "all"),
        ("domain_c", "xgboost_domain_b_cross_bc", "all"),
        ("domain_c", "logreg_domain_b_cross_bc", "all"),
        ("domain_b", "lightgbm_domain_c_cross_cb", "all"),
        ("domain_b", "xgboost_domain_c_cross_cb", "all"),
        ("domain_b", "logreg_domain_c_cross_cb", "all"),
        # NOTE: all-versions and combined are evaluated via kfold-evaluate-all
    ]

    summary_rows = []
    for domain, model_tag, vf in configs:
        model_path = _MODELS_DIR / f"{model_tag}.pkl"
        if not model_path.exists():
            typer.echo(f"  Skipping {model_tag} (not found)")
            continue

        try:
            per_version, agg = _run_evaluation(domain, model_tag, version_filter=vf)
        except Exception as e:
            typer.echo(f"  ERROR {model_tag} on {domain}: {e}")
            continue

        out_tag = f"eval_{domain}_{model_tag}"
        if vf != "all":
            out_tag += f"_{vf}"
        agg_path = _RESULTS_DIR / f"{out_tag}.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        _write_detail_jsonl(per_version, _RESULTS_DIR / f"detail_{out_tag}.jsonl")

        a = agg["aggregate"]
        er, ecr = effective_means_from_aggregate(a)
        row = {
            "domain": domain,
            "model_tag": model_tag,
            "version_filter": vf,
            "recall_predictor": a["mean_recall_predictor"],
            "recall_with_sentinel": a["mean_recall_with_sentinel"],
            "call_reduction": a["mean_call_reduction"],
            "mean_effective_recall": er,
            "mean_effective_call_reduction": ecr,
            "false_omission_rate": a["mean_false_omission_rate"],
        }
        summary_rows.append(row)
        typer.echo(f"  {model_tag:40s} {domain:10s} {vf:15s} | "
                   f"rec={a['mean_recall_predictor']:.3f}  "
                   f"rec+s={a['mean_recall_with_sentinel']:.3f}  "
                   f"cr={a['mean_call_reduction']:.3f}  "
                   f"eff_r={er:.3f}  eff_cr={ecr:.3f}")

    summary_path = _RESULTS_DIR / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    typer.echo(f"\nWrote {summary_path}")


@app.command()
def compare() -> None:
    """Build and print comparison of rule-based vs learned models."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comp = _build_comparison()

    comp_path = _RESULTS_DIR / "comparison_summary.json"
    with open(comp_path, "w") as f:
        json.dump(comp, f, indent=2)
    typer.echo(f"Wrote {comp_path}")

    for domain, entries in comp.items():
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  {domain}")
        typer.echo(f"{'='*60}")
        for tag, m in entries.items():
            rec = m.get("recall_predictor", "N/A")
            rec_s = m.get("recall_with_sentinel", "N/A")
            cr = m.get("call_reduction", "N/A")
            eff_r = m.get("effective_recall", "N/A")
            eff_cr = m.get("effective_call_reduction", "N/A")
            rec_str = f"{rec:.3f}" if isinstance(rec, float) else rec
            rec_s_str = f"{rec_s:.3f}" if isinstance(rec_s, float) else rec_s
            cr_str = f"{cr:.3f}" if isinstance(cr, float) else cr
            eff_r_str = f"{eff_r:.3f}" if isinstance(eff_r, float) else eff_r
            eff_cr_str = f"{eff_cr:.3f}" if isinstance(eff_cr, float) else eff_cr
            typer.echo(
                f"  {tag:40s} | rec={rec_str}  rec+s={rec_s_str}  cr={cr_str}  "
                f"eff_r={eff_r_str}  eff_cr={eff_cr_str}",
            )


@app.command()
def threshold_sensitivity(
    domain: str = typer.Option("domain_b", help="Domain to evaluate"),
    model_tag: str = typer.Option("lightgbm_domain_b_temporal", help="Model tag"),
    version_filter: str = typer.Option("all", help="all | temporal_train | temporal_test"),
) -> None:
    """Run threshold sensitivity analysis at fixed thresholds."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

    typer.echo(f"Threshold sensitivity for {model_tag} on {domain} ({version_filter})")
    results = _threshold_sensitivity(domain, model_tag, thresholds, version_filter=version_filter)

    for r in results:
        typer.echo(
            f"  τ={r['threshold']:.2f}: rec+s={r['mean_recall_with_sentinel']:.3f}  "
            f"cr={r['mean_call_reduction']:.3f}  "
            f"eff_rec={r['mean_effective_recall']:.3f}  eff_cr={r['mean_effective_call_reduction']:.3f}"
        )

    out_path = _RESULTS_DIR / f"threshold_sensitivity_{domain}_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    typer.echo(f"Wrote {out_path}")


@app.command()
def kfold_evaluate(
    domain: str = typer.Option("domain_a", help="domain_a | domain_b | domain_c"),
    model_type: str = typer.Option("lightgbm", help="lightgbm | xgboost | logreg"),
    n_folds: int = typer.Option(5, help="Number of folds"),
    pca_components: int = typer.Option(24, help="PCA components for embeddings"),
    n_trials: int = typer.Option(50, help="Optuna trials"),
) -> None:
    """Run k-fold cross-validation evaluation on the 'all' split."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_version, agg = _kfold_evaluate(
        domain, model_type,
        n_folds=n_folds,
        pca_components=pca_components,
        n_trials=n_trials,
    )

    out_tag = f"kfold_{domain}_{model_type}_{n_folds}fold"
    agg_path = _RESULTS_DIR / f"{out_tag}.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    typer.echo(f"\nWrote {agg_path}")

    a = agg["aggregate"]
    typer.echo(f"\n{model_type} on {domain} ({n_folds}-fold CV):")
    _echo_learned_aggregate(a)


@app.command()
def kfold_evaluate_combined(
    target_domain: str = typer.Option("domain_a", help="domain_a | domain_b | domain_c"),
    model_type: str = typer.Option("lightgbm", help="lightgbm | xgboost | logreg"),
    n_folds: int = typer.Option(5, help="Number of folds"),
    pca_components: int = typer.Option(24, help="PCA components for embeddings"),
    n_trials: int = typer.Option(50, help="Optuna trials"),
) -> None:
    """Run k-fold CV with combined-domain training on a single target domain."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_version, agg = _kfold_evaluate_combined(
        target_domain, model_type,
        n_folds=n_folds,
        pca_components=pca_components,
        n_trials=n_trials,
    )

    out_tag = f"kfold_combined_{target_domain}_{model_type}_{n_folds}fold"
    agg_path = _RESULTS_DIR / f"{out_tag}.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    typer.echo(f"\nWrote {agg_path}")

    a = agg["aggregate"]
    typer.echo(f"\nCombined k-fold: {model_type} -> {target_domain} ({n_folds}-fold CV):")
    _echo_learned_aggregate(a)


@app.command()
def kfold_evaluate_all(
    n_folds: int = typer.Option(5, help="Number of folds"),
    pca_components: int = typer.Option(24, help="PCA components"),
    n_trials: int = typer.Option(50, help="Optuna trials"),
) -> None:
    """Run k-fold CV for all configurations: single-domain and combined."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # --- Single-domain k-fold ---
    single_configs = [
        ("domain_a", "lightgbm"),
        ("domain_a", "xgboost"),
        ("domain_a", "logreg"),
        ("domain_b", "lightgbm"),
        ("domain_b", "xgboost"),
        ("domain_b", "logreg"),
        ("domain_c", "lightgbm"),
        ("domain_c", "xgboost"),
        ("domain_c", "logreg"),
    ]

    for domain, mt in single_configs:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"K-fold CV (single-domain): {mt} on {domain}")
        typer.echo(f"{'='*60}")

        try:
            per_version, agg = _kfold_evaluate(
                domain, mt,
                n_folds=n_folds,
                pca_components=pca_components,
                n_trials=n_trials,
            )
        except Exception as e:
            typer.echo(f"  ERROR: {e}")
            continue

        out_tag = f"kfold_{domain}_{mt}_{n_folds}fold"
        agg_path = _RESULTS_DIR / f"{out_tag}.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)

        a = agg["aggregate"]
        er, ecr = effective_means_from_aggregate(a)
        row = {
            "domain": domain,
            "model_type": mt,
            "mode": "single_domain",
            "n_folds": n_folds,
            "recall_predictor": a["mean_recall_predictor"],
            "recall_with_sentinel": a["mean_recall_with_sentinel"],
            "call_reduction": a["mean_call_reduction"],
            "mean_effective_recall": er,
            "mean_effective_call_reduction": ecr,
            "false_omission_rate": a["mean_false_omission_rate"],
        }
        summary_rows.append(row)
        typer.echo(f"  rec={a['mean_recall_predictor']:.3f}  "
                   f"rec+s={a['mean_recall_with_sentinel']:.3f}  "
                   f"cr={a['mean_call_reduction']:.3f}  "
                   f"eff_r={er:.3f}  eff_cr={ecr:.3f}")

    # --- Combined k-fold ---
    combined_configs = [
        ("domain_a", "lightgbm"),
        ("domain_a", "xgboost"),
        ("domain_a", "logreg"),
        ("domain_b", "lightgbm"),
        ("domain_b", "xgboost"),
        ("domain_b", "logreg"),
        ("domain_c", "lightgbm"),
        ("domain_c", "xgboost"),
        ("domain_c", "logreg"),
    ]

    for target_domain, mt in combined_configs:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"K-fold CV (combined): {mt} -> {target_domain}")
        typer.echo(f"{'='*60}")

        try:
            per_version, agg = _kfold_evaluate_combined(
                target_domain, mt,
                n_folds=n_folds,
                pca_components=pca_components,
                n_trials=n_trials,
            )
        except Exception as e:
            typer.echo(f"  ERROR: {e}")
            continue

        out_tag = f"kfold_combined_{target_domain}_{mt}_{n_folds}fold"
        agg_path = _RESULTS_DIR / f"{out_tag}.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)

        a = agg["aggregate"]
        er, ecr = effective_means_from_aggregate(a)
        row = {
            "domain": target_domain,
            "model_type": mt,
            "mode": "combined",
            "n_folds": n_folds,
            "recall_predictor": a["mean_recall_predictor"],
            "recall_with_sentinel": a["mean_recall_with_sentinel"],
            "call_reduction": a["mean_call_reduction"],
            "mean_effective_recall": er,
            "mean_effective_call_reduction": ecr,
            "false_omission_rate": a["mean_false_omission_rate"],
        }
        summary_rows.append(row)
        typer.echo(f"  rec={a['mean_recall_predictor']:.3f}  "
                   f"rec+s={a['mean_recall_with_sentinel']:.3f}  "
                   f"cr={a['mean_call_reduction']:.3f}  "
                   f"eff_r={er:.3f}  eff_cr={ecr:.3f}")

    summary_path = _RESULTS_DIR / "kfold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    typer.echo(f"\nWrote {summary_path}")


if __name__ == "__main__":
    app()
