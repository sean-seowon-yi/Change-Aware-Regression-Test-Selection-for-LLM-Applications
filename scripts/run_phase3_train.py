"""CLI for Phase 3 model training.

Trains LightGBM, XGBoost, and/or logistic regression models on the feature
dataset, tunes thresholds, runs threshold sweeps, extracts feature
importance, and saves all artifacts.

Usage examples:
    # Train all three models on domain_a with temporal split
    python -m scripts.run_phase3_train --domain domain_a --split temporal --model all

    # Train all configurations (including XGBoost)
    python -m scripts.run_phase3_train --all-configs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from src.phase3.dataset import build_dataset
from src.phase3.trainer import (
    compute_metrics,
    feature_importance_lgb,
    feature_importance_logreg,
    feature_importance_xgb,
    fit_pca,
    oof_predictions_lgb,
    oof_predictions_logreg,
    oof_predictions_xgb,
    prepare_features,
    save_model,
    threshold_sweep,
    train_lightgbm,
    train_logreg,
    train_xgboost,
    tune_threshold,
    _split_embeddings,
)

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / "models"
_RESULTS_DIR = _PROJECT_ROOT / "results" / "phase3"


# ---------------------------------------------------------------------------
# Dataset loading for different split configurations
# ---------------------------------------------------------------------------

def _load_split(
    domain: str,
    split: str,
) -> tuple[
    pd.DataFrame, np.ndarray, pd.DataFrame,
    pd.DataFrame | None, np.ndarray | None, pd.DataFrame | None,
]:
    """Load train and (optionally) test data for a given configuration.

    Returns (X_train, y_train, meta_train, X_test, y_test, meta_test).
    X_test/y_test/meta_test are None for "all" split.
    """
    if split == "temporal":
        X_tr, y_tr, m_tr = build_dataset(domain, split="temporal_train")
        X_te, y_te, m_te = build_dataset(domain, split="temporal_test")
        return X_tr, y_tr.values, m_tr, X_te, y_te.values, m_te

    if split == "all":
        X, y, m = build_dataset(domain, split="all")
        return X, y.values, m, None, None, None

    cross_map = {
        "cross_ab": ("domain_a", "domain_b"),
        "cross_ba": ("domain_b", "domain_a"),
        "cross_ac": ("domain_a", "domain_c"),
        "cross_ca": ("domain_c", "domain_a"),
        "cross_bc": ("domain_b", "domain_c"),
        "cross_cb": ("domain_c", "domain_b"),
    }
    if split in cross_map:
        train_dom, test_dom = cross_map[split]
        X_tr, y_tr, m_tr = build_dataset(train_dom, split="all")
        X_te, y_te, m_te = build_dataset(test_dom, split="all")
        return X_tr, y_tr.values, m_tr, X_te, y_te.values, m_te

    raise ValueError(f"Unknown split: {split!r}")


def _load_combined_split(
    split: str,
) -> tuple[
    pd.DataFrame, np.ndarray, pd.DataFrame,
    pd.DataFrame | None, np.ndarray | None, pd.DataFrame | None,
]:
    """Load combined (all domains) datasets.

    The *split* parameter is forwarded to ``build_dataset`` for each domain.
    For combined training, "all" is the expected split (k-fold handles
    train/test partitioning). Temporal splits are also supported — each
    domain's data will be split identically (e.g. v01-v50 train, v51-v70 test).
    """
    _SPLIT_MAP = {
        "all": "all",
        "temporal": "temporal_train",
        "temporal_train": "temporal_train",
        "temporal_test": "temporal_test",
    }
    frames_X, frames_y, frames_m = [], [], []
    per_domain_split = _SPLIT_MAP.get(split, "all")
    for dom in ["domain_a", "domain_b", "domain_c"]:
        X, y, m = build_dataset(dom, split=per_domain_split)
        frames_X.append(X)
        frames_y.append(y.values)
        frames_m.append(m)
    X_tr = pd.concat(frames_X, ignore_index=True)
    y_tr = np.concatenate(frames_y)
    m_tr = pd.concat(frames_m, ignore_index=True)
    return X_tr, y_tr, m_tr, None, None, None


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def _extract_version_groups(meta: pd.DataFrame) -> np.ndarray:
    """Convert version_id column to integer group labels for GroupKFold.

    Uses composite domain+version_id when 'domain' column is present to avoid
    collisions between identical version IDs from different domains.
    """
    if "domain" in meta.columns:
        composite = meta["domain"] + "_" + meta["version_id"]
    else:
        composite = meta["version_id"]
    return composite.factorize()[0]


def _run_single(
    domain: str,
    split: str,
    model_type: str,
    *,
    pca_components: int = 24,
    n_trials: int = 50,
) -> dict:
    """Train one model, tune threshold via OOF, save artifacts, return metrics."""
    tag = f"{model_type}_{domain}_{split}"
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Training: {tag}")
    typer.echo(f"{'='*60}")

    if domain == "combined":
        X_tr, y_tr, m_tr, X_te, y_te, m_te = _load_combined_split(split)
    else:
        X_tr, y_tr, m_tr, X_te, y_te, m_te = _load_split(domain, split)

    X_sc_tr, c_emb_tr, t_emb_tr = _split_embeddings(X_tr)

    pca_n = min(pca_components, c_emb_tr.shape[0], c_emb_tr.shape[1])
    change_pca, test_pca = fit_pca(c_emb_tr, t_emb_tr, n_components=pca_n)

    X_combined_tr, feature_names = prepare_features(X_tr, change_pca, test_pca)
    groups_tr = _extract_version_groups(m_tr)

    import xgboost as xgb

    scaler = None
    if model_type == "lightgbm":
        model, lgb_params, n_boost = train_lightgbm(
            X_combined_tr, y_tr, feature_names,
            groups=groups_tr, tune=True, n_trials=n_trials,
        )
        y_proba_tr = model.predict(X_combined_tr)

        typer.echo("  Computing OOF predictions for threshold tuning...")
        oof_proba = oof_predictions_lgb(
            X_combined_tr, y_tr, groups_tr, feature_names, lgb_params, n_boost,
        )
    elif model_type == "xgboost":
        model, xgb_params, n_boost = train_xgboost(
            X_combined_tr, y_tr, feature_names,
            groups=groups_tr, tune=True, n_trials=n_trials,
        )
        dtrain = xgb.DMatrix(X_combined_tr, feature_names=feature_names)
        y_proba_tr = model.predict(dtrain)

        typer.echo("  Computing OOF predictions for threshold tuning...")
        oof_proba = oof_predictions_xgb(
            X_combined_tr, y_tr, groups_tr, feature_names, xgb_params, n_boost,
        )
    else:
        model, scaler = train_logreg(X_combined_tr, y_tr, groups=groups_tr)
        y_proba_tr = model.predict_proba(scaler.transform(X_combined_tr))[:, 1]

        typer.echo("  Computing OOF predictions for threshold tuning...")
        oof_proba = oof_predictions_logreg(
            X_combined_tr, y_tr, groups_tr, model.C,
        )

    threshold = tune_threshold(y_tr, oof_proba, min_recall=0.95)

    model_path = _MODELS_DIR / f"{tag}.pkl"
    save_model(
        model_path,
        model=model,
        change_pca=change_pca,
        test_pca=test_pca,
        threshold=threshold,
        feature_names=feature_names,
        scaler=scaler,
        metadata={"domain": domain, "split": split, "model_type": model_type},
    )

    result: dict = {"domain": domain, "split": split, "model_type": model_type}

    train_metrics = compute_metrics(y_tr, y_proba_tr, threshold)
    result["train_metrics"] = train_metrics
    typer.echo(f"  Train: recall={train_metrics['recall']:.3f}  "
               f"call_red={train_metrics['call_reduction']:.3f}  "
               f"auroc={train_metrics.get('auroc', 'N/A')}")

    oof_metrics = compute_metrics(y_tr, oof_proba, threshold)
    result["oof_metrics"] = oof_metrics
    typer.echo(f"  OOF:   recall={oof_metrics['recall']:.3f}  "
               f"call_red={oof_metrics['call_reduction']:.3f}  "
               f"auroc={oof_metrics.get('auroc', 'N/A')}")

    if X_te is not None and y_te is not None:
        X_combined_te, _ = prepare_features(X_te, change_pca, test_pca)

        if model_type == "lightgbm":
            y_proba_te = model.predict(X_combined_te)
        elif model_type == "xgboost":
            dtest = xgb.DMatrix(X_combined_te, feature_names=feature_names)
            y_proba_te = model.predict(dtest)
        else:
            y_proba_te = model.predict_proba(scaler.transform(X_combined_te))[:, 1]

        test_metrics = compute_metrics(y_te, y_proba_te, threshold)
        result["test_metrics"] = test_metrics
        typer.echo(f"  Test:  recall={test_metrics['recall']:.3f}  "
                   f"call_red={test_metrics['call_reduction']:.3f}  "
                   f"auroc={test_metrics.get('auroc', 'N/A')}")

        sweep = threshold_sweep(y_te, y_proba_te, len(y_te))
        result["threshold_sweep"] = sweep
    else:
        sweep = threshold_sweep(y_tr, oof_proba, len(y_tr))
        result["threshold_sweep"] = sweep

    if model_type == "lightgbm":
        result["feature_importance"] = feature_importance_lgb(model, feature_names)
    elif model_type == "xgboost":
        result["feature_importance"] = feature_importance_xgb(model, feature_names)
    else:
        result["feature_importance"] = feature_importance_logreg(model, feature_names)

    return result


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def train(
    domain: str = typer.Option("domain_a", help="domain_a | domain_b | domain_c | combined"),
    split: str = typer.Option("temporal", help="temporal | all | cross_ab | cross_ba | cross_ac | cross_ca | cross_bc | cross_cb"),
    model: str = typer.Option("all", help="lightgbm | xgboost | logreg | all"),
    pca_components: int = typer.Option(24, help="PCA components for embeddings"),
    n_trials: int = typer.Option(50, help="Optuna trials for LightGBM/XGBoost"),
) -> None:
    """Train model(s) for a single configuration."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if model == "all":
        model_types = ["lightgbm", "xgboost", "logreg"]
    elif model == "both":
        model_types = ["lightgbm", "logreg"]
    else:
        model_types = [model]

    all_results = []

    for mt in model_types:
        result = _run_single(domain, split, mt, pca_components=pca_components, n_trials=n_trials)
        all_results.append(result)

        out_path = _RESULTS_DIR / f"metrics_{domain}_{mt}_{split}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        typer.echo(f"  Wrote {out_path}")

    typer.echo("\nDone.")


@app.command()
def train_all(
    pca_components: int = typer.Option(24, help="PCA components for embeddings"),
    n_trials: int = typer.Option(50, help="Optuna trials for LightGBM/XGBoost"),
) -> None:
    """Train pre-trained models for temporal and cross-domain evaluation."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("domain_a", "temporal"),
        ("domain_b", "temporal"),
        ("domain_c", "temporal"),
        ("domain_a", "cross_ab"),
        ("domain_a", "cross_ac"),
        ("domain_b", "cross_ba"),
        ("domain_b", "cross_bc"),
        ("domain_c", "cross_ca"),
        ("domain_c", "cross_cb"),
        # NOTE: all-versions and combined models are trained on-the-fly
        # during kfold-evaluate-all; no pre-trained models needed.
    ]

    all_results = []
    for domain, split in configs:
        for mt in ["lightgbm", "xgboost", "logreg"]:
            result = _run_single(domain, split, mt, pca_components=pca_components, n_trials=n_trials)
            all_results.append(result)

            out_path = _RESULTS_DIR / f"metrics_{domain}_{mt}_{split}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

    summary_path = _RESULTS_DIR / "training_summary.json"
    summary = []
    for r in all_results:
        entry = {
            "domain": r["domain"],
            "split": r["split"],
            "model_type": r["model_type"],
            "train_recall": r["train_metrics"]["recall"],
            "train_call_reduction": r["train_metrics"]["call_reduction"],
            "train_auroc": r["train_metrics"].get("auroc"),
        }
        if "test_metrics" in r:
            entry["test_recall"] = r["test_metrics"]["recall"]
            entry["test_call_reduction"] = r["test_metrics"]["call_reduction"]
            entry["test_auroc"] = r["test_metrics"].get("auroc")
        summary.append(entry)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"\nWrote summary to {summary_path}")

    typer.echo("\n" + "="*70)
    typer.echo("TRAINING SUMMARY")
    typer.echo("="*70)
    for entry in summary:
        line = (f"  {entry['model_type']:8s} {entry['domain']:10s} {entry['split']:10s} | "
                f"train_rec={entry['train_recall']:.3f}  train_cr={entry['train_call_reduction']:.3f}")
        if "test_recall" in entry:
            line += f"  | test_rec={entry['test_recall']:.3f}  test_cr={entry['test_call_reduction']:.3f}"
        typer.echo(line)

    typer.echo("\nAll done.")


if __name__ == "__main__":
    app()
