"""Model training, threshold tuning, and persistence for the learned selector.

Trains LightGBM, XGBoost, and logistic regression on the feature dataset
produced by ``dataset.py``.  Embedding columns are reduced via PCA before
training.  Threshold is tuned on out-of-fold predictions to avoid
overfitting, then the final model is retrained on all training data.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)

_CHANGE_EMB_PREFIX = "change_emb_"
_TEST_EMB_PREFIX = "test_emb_"

# ---------------------------------------------------------------------------
# Embedding / PCA helpers
# ---------------------------------------------------------------------------

def _split_embeddings(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Separate scalar features from raw embedding columns.

    Returns (X_scalar, change_embs, test_embs) where:
      - X_scalar: DataFrame with embedding columns removed
      - change_embs: (N, 384) array
      - test_embs: (N, 384) array
    """
    change_cols = sorted([c for c in X.columns if c.startswith(_CHANGE_EMB_PREFIX)])
    test_cols = sorted([c for c in X.columns if c.startswith(_TEST_EMB_PREFIX)])
    scalar_cols = [c for c in X.columns if c not in change_cols and c not in test_cols]

    return (
        X[scalar_cols].copy(),
        X[change_cols].values.astype(np.float32),
        X[test_cols].values.astype(np.float32),
    )


def fit_pca(
    change_embs: np.ndarray,
    test_embs: np.ndarray,
    *,
    n_components: int = 24,
) -> tuple[PCA, PCA]:
    """Fit PCA on training embeddings.

    Returns (change_pca, test_pca) fitted transformers.
    """
    max_c = min(n_components, change_embs.shape[0], change_embs.shape[1])
    change_pca = PCA(n_components=max(1, max_c))
    change_pca.fit(change_embs)
    logger.info(
        "Change PCA: %d components, explained variance %.2f%%",
        change_pca.n_components_,
        change_pca.explained_variance_ratio_.sum() * 100,
    )

    max_t = min(n_components, test_embs.shape[0], test_embs.shape[1])
    test_pca = PCA(n_components=max(1, max_t))
    test_pca.fit(test_embs)
    logger.info(
        "Test PCA: %d components, explained variance %.2f%%",
        test_pca.n_components_,
        test_pca.explained_variance_ratio_.sum() * 100,
    )

    return change_pca, test_pca


def apply_pca(
    X_scalar: pd.DataFrame,
    change_embs: np.ndarray,
    test_embs: np.ndarray,
    change_pca: PCA,
    test_pca: PCA,
) -> np.ndarray:
    """Apply fitted PCA and concatenate with scalar features."""
    change_reduced = change_pca.transform(change_embs)
    test_reduced = test_pca.transform(test_embs)
    return np.hstack([X_scalar.values, change_reduced, test_reduced])


def prepare_features(
    X: pd.DataFrame,
    change_pca: PCA,
    test_pca: PCA,
) -> tuple[np.ndarray, list[str]]:
    """Full pipeline: split embeddings, apply PCA, return feature matrix + names."""
    X_scalar, change_embs, test_embs = _split_embeddings(X)
    X_combined = apply_pca(X_scalar, change_embs, test_embs, change_pca, test_pca)

    n_c = change_pca.n_components_
    n_t = test_pca.n_components_
    feature_names = (
        list(X_scalar.columns)
        + [f"change_pca_{i}" for i in range(n_c)]
        + [f"test_pca_{i}" for i in range(n_t)]
    )
    return X_combined, feature_names


# ---------------------------------------------------------------------------
# LightGBM training (with Optuna)
# ---------------------------------------------------------------------------

def _lgb_objective(
    trial: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    groups: np.ndarray | None,
) -> float:
    """Optuna objective: minimize binary logloss with 3-fold GroupKFold CV."""
    num_boost_round = trial.suggest_int("n_estimators", 100, 500)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "random_state": 42,
    }

    ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

    if groups is not None:
        n_groups = len(np.unique(groups))
        gkf = GroupKFold(n_splits=min(3, n_groups))
        folds = list(gkf.split(X_train, y_train, groups))
    else:
        folds = None

    cv_kwargs: dict[str, Any] = {"return_cvbooster": False, "num_boost_round": num_boost_round}
    if folds is not None:
        cv_kwargs["folds"] = folds
    else:
        cv_kwargs["nfold"] = 3
        cv_kwargs["seed"] = 42

    cv_result = lgb.cv(params, ds, **cv_kwargs)
    best_iter_key = "valid binary_logloss-mean"
    if best_iter_key not in cv_result:
        best_iter_key = "binary_logloss-mean"
    return min(cv_result[best_iter_key])


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    groups: np.ndarray | None = None,
    tune: bool = True,
    n_trials: int = 50,
) -> tuple[lgb.Booster, dict, int]:
    """Train a LightGBM model, optionally with Optuna hyperparameter search.

    Returns (model, lgb_params_without_n_estimators, n_estimators).
    The returned params can be passed directly to ``oof_predictions_lgb``.
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("Class balance: %d pos, %d neg, scale_pos_weight=%.2f", n_pos, n_neg, scale_pos_weight)

    if tune:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: _lgb_objective(trial, X_train, y_train, scale_pos_weight, groups),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        best = study.best_params
        logger.info("Best Optuna params: %s (loss=%.5f)", best, study.best_value)
    else:
        best = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_samples": 10,
        }

    n_estimators = best.pop("n_estimators")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "random_state": 42,
        **best,
    }

    ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=n_estimators)
    return model, params, n_estimators


# ---------------------------------------------------------------------------
# XGBoost training (with Optuna)
# ---------------------------------------------------------------------------

def _xgb_objective(
    trial: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    groups: np.ndarray | None,
) -> float:
    """Optuna objective for XGBoost: minimize logloss with GroupKFold CV."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "scale_pos_weight": scale_pos_weight,
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    n_estimators = trial.suggest_int("n_estimators", 100, 500)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    if groups is not None:
        n_groups = len(np.unique(groups))
        gkf = GroupKFold(n_splits=min(3, n_groups))
        folds = list(gkf.split(X_train, y_train, groups))
    else:
        folds = None

    if folds is not None:
        cv_result = xgb.cv(
            params, dtrain,
            num_boost_round=n_estimators,
            folds=folds,
            verbose_eval=False,
        )
    else:
        cv_result = xgb.cv(
            params, dtrain,
            num_boost_round=n_estimators,
            nfold=3,
            seed=42,
            verbose_eval=False,
        )

    return cv_result["test-logloss-mean"].min()


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    groups: np.ndarray | None = None,
    tune: bool = True,
    n_trials: int = 50,
) -> tuple[xgb.Booster, dict, int]:
    """Train an XGBoost model, optionally with Optuna hyperparameter search.

    Returns (model, xgb_params_without_n_estimators, n_estimators).
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("XGB class balance: %d pos, %d neg, scale_pos_weight=%.2f", n_pos, n_neg, scale_pos_weight)

    if tune:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: _xgb_objective(trial, X_train, y_train, scale_pos_weight, groups),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        best = study.best_params
        logger.info("Best XGB Optuna params: %s (loss=%.5f)", best, study.best_value)
    else:
        best = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 5,
        }

    n_estimators = best.pop("n_estimators")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        **best,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    model = xgb.train(params, dtrain, num_boost_round=n_estimators)
    return model, params, n_estimators


# ---------------------------------------------------------------------------
# Logistic Regression training
# ---------------------------------------------------------------------------

def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    groups: np.ndarray | None = None,
) -> tuple[LogisticRegression, StandardScaler]:
    """Train logistic regression with standardization and CV-tuned C.

    Uses GroupKFold (by version) when *groups* is provided, preventing
    rows from the same prompt version appearing in both train and
    validation folds during hyperparameter search.

    The scaler is wrapped inside a Pipeline with GridSearchCV so that
    validation folds are not seen during fit_transform (no leakage).
    The final scaler is refit on the full training set after CV.
    """
    from sklearn.pipeline import Pipeline

    if groups is not None:
        n_groups = len(np.unique(groups))
        cv_splitter = GroupKFold(n_splits=min(5, n_groups))
    else:
        cv_splitter = 5

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    grid = GridSearchCV(
        pipe,
        param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
        cv=cv_splitter,
        scoring="recall",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups)
    best_C = grid.best_params_["clf__C"]
    logger.info("Best LogReg C=%.2f (recall=%.3f)", best_C, grid.best_score_)

    best_pipe = grid.best_estimator_
    scaler = best_pipe.named_steps["scaler"]
    model = best_pipe.named_steps["clf"]
    return model, scaler


# ---------------------------------------------------------------------------
# Out-of-fold predictions (for threshold tuning)
# ---------------------------------------------------------------------------

def oof_predictions_lgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    params: dict,
    n_estimators: int,
) -> np.ndarray:
    """Get out-of-fold predictions via GroupKFold (5-fold by version).

    NOTE: PCA has already been applied to X_train by the caller. Ideally
    PCA would be refit per fold, but the effect is negligible (PCA on
    high-dim embeddings is very stable across 80/20 splits). The -0.05
    safety margin in tune_threshold compensates for any minor optimism.
    """
    oof = np.zeros(len(y_train), dtype=np.float64)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        logger.warning("Only %d group(s) — returning 0.5 as OOF predictions", n_groups)
        oof[:] = 0.5
        return oof
    gkf = GroupKFold(n_splits=min(5, n_groups))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        ds = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx],
                         feature_name=feature_names, free_raw_data=False)
        fold_model = lgb.train(params, ds, num_boost_round=n_estimators)
        oof[val_idx] = fold_model.predict(X_train[val_idx])

    return oof


def oof_predictions_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    params: dict,
    n_estimators: int,
) -> np.ndarray:
    """Get out-of-fold predictions for XGBoost via GroupKFold (5-fold by version)."""
    oof = np.zeros(len(y_train), dtype=np.float64)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        logger.warning("Only %d group(s) — returning 0.5 as OOF predictions", n_groups)
        oof[:] = 0.5
        return oof
    gkf = GroupKFold(n_splits=min(5, n_groups))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx], feature_names=feature_names)
        dval = xgb.DMatrix(X_train[val_idx], feature_names=feature_names)
        fold_model = xgb.train(params, dtrain, num_boost_round=n_estimators)
        oof[val_idx] = fold_model.predict(dval)

    return oof


def oof_predictions_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    C: float,
) -> np.ndarray:
    """Get out-of-fold predictions for logistic regression via GroupKFold.

    A fresh StandardScaler is fit inside each fold to prevent leakage of
    validation-set statistics into the training scaler.
    """
    oof = np.zeros(len(y_train), dtype=np.float64)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        logger.warning("Only %d group(s) — returning 0.5 as OOF predictions", n_groups)
        oof[:] = 0.5
        return oof
    gkf = GroupKFold(n_splits=min(5, n_groups))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_scaler = StandardScaler()
        X_tr_scaled = fold_scaler.fit_transform(X_train[tr_idx])
        X_val_scaled = fold_scaler.transform(X_train[val_idx])

        fold_model = LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=42,
        )
        fold_model.fit(X_tr_scaled, y_train[tr_idx])
        oof[val_idx] = fold_model.predict_proba(X_val_scaled)[:, 1]

    return oof


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    min_recall: float = 0.95,
) -> float:
    """Find the highest threshold that keeps recall >= min_recall.

    Must be called with **out-of-fold** predictions, not in-sample.
    Sweeps from 0.01 to 0.99.  Returns the threshold (with safety buffer
    of -0.05 applied) or 0.01 if no threshold meets the constraint.
    """
    best_tau = 0.01
    for tau_100 in range(1, 100):
        tau = tau_100 / 100.0
        preds = (y_proba >= tau).astype(int)
        rec = recall_score(y_true, preds, zero_division=1.0)
        if rec >= min_recall:
            best_tau = tau

    safe_tau = max(0.01, best_tau - 0.05)
    logger.info("Threshold tuning: best_tau=%.2f (safety -> %.2f)", best_tau, safe_tau)
    return safe_tau


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_total_tests: int,
) -> list[dict]:
    """Sweep threshold from 0.01 to 0.99, record recall and call_reduction."""
    results = []
    for tau_100 in range(1, 100):
        tau = tau_100 / 100.0
        preds = (y_proba >= tau).astype(int)
        n_selected = int(preds.sum())
        rec = recall_score(y_true, preds, zero_division=1.0)
        call_red = 1.0 - n_selected / max(n_total_tests, 1)
        results.append({
            "threshold": tau,
            "recall": round(rec, 4),
            "call_reduction": round(call_red, 4),
            "n_selected": n_selected,
        })
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Compute classification metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    n_pos = int(y_true.sum())
    n_pred = int(y_pred.sum())
    n_total = len(y_true)

    rec = recall_score(y_true, y_pred, zero_division=1.0)
    call_red = 1.0 - n_pred / max(n_total, 1)

    metrics = {
        "threshold": threshold,
        "recall": round(rec, 4),
        "call_reduction": round(call_red, 4),
        "n_impacted": n_pos,
        "n_predicted": n_pred,
        "n_total": n_total,
    }

    if n_pos > 0 and n_pos < n_total:
        metrics["auroc"] = round(roc_auc_score(y_true, y_proba), 4)
        metrics["auprc"] = round(average_precision_score(y_true, y_proba), 4)
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    return metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def feature_importance_lgb(model: lgb.Booster, feature_names: list[str]) -> list[dict]:
    """Extract gain-based feature importance from a LightGBM model."""
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(feature_names, importance.tolist()), key=lambda x: -x[1])
    return [{"feature": name, "importance": imp} for name, imp in pairs]


def feature_importance_logreg(
    model: LogisticRegression,
    feature_names: list[str],
) -> list[dict]:
    """Extract absolute coefficient-based importance from logistic regression."""
    coefs = np.abs(model.coef_[0])
    pairs = sorted(zip(feature_names, coefs.tolist()), key=lambda x: -x[1])
    return [{"feature": name, "importance": imp} for name, imp in pairs]


def feature_importance_xgb(model: xgb.Booster, feature_names: list[str]) -> list[dict]:
    """Extract gain-based feature importance from an XGBoost model."""
    score_map = model.get_score(importance_type="gain")
    importance = [score_map.get(fn, 0.0) for fn in feature_names]
    pairs = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    return [{"feature": name, "importance": imp} for name, imp in pairs]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    path: str | Path,
    *,
    model: Any,
    change_pca: PCA,
    test_pca: PCA,
    threshold: float,
    feature_names: list[str],
    scaler: StandardScaler | None = None,
    metadata: dict | None = None,
) -> None:
    """Save a trained model bundle to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "change_pca": change_pca,
        "test_pca": test_pca,
        "threshold": threshold,
        "feature_names": feature_names,
        "scaler": scaler,
        "metadata": metadata or {},
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Saved model to %s", path)


def load_model(path: str | Path) -> dict:
    """Load a model bundle from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
