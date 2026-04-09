"""Tests for src/phase3/trainer.py"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.phase3.trainer import (
    _split_embeddings,
    apply_pca,
    compute_metrics,
    feature_importance_lgb,
    feature_importance_logreg,
    fit_pca,
    load_model,
    oof_predictions_lgb,
    oof_predictions_logreg,
    prepare_features,
    save_model,
    threshold_sweep,
    train_lightgbm,
    train_logreg,
    tune_threshold,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_X():
    """Create a sample feature DataFrame with scalar + embedding columns."""
    rng = np.random.RandomState(42)
    n = 200
    data = {}
    for i in range(5):
        data[f"scalar_{i}"] = rng.randn(n)
    for i in range(384):
        data[f"change_emb_{i}"] = rng.randn(n).astype(np.float32)
    for i in range(384):
        data[f"test_emb_{i}"] = rng.randn(n).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture
def sample_y():
    rng = np.random.RandomState(42)
    y = np.zeros(200, dtype=int)
    y[rng.choice(200, 30, replace=False)] = 1
    return y


@pytest.fixture
def sample_groups():
    return np.repeat(np.arange(10), 20)


# ---------------------------------------------------------------------------
# _split_embeddings
# ---------------------------------------------------------------------------

class TestSplitEmbeddings:
    def test_basic_split(self, sample_X):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        assert X_sc.shape == (200, 5)
        assert c_emb.shape == (200, 384)
        assert t_emb.shape == (200, 384)

    def test_no_embedding_cols(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        X_sc, c_emb, t_emb = _split_embeddings(df)
        assert X_sc.shape == (2, 2)
        assert c_emb.shape == (2, 0)
        assert t_emb.shape == (2, 0)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class TestPCA:
    def test_fit_pca(self, sample_X):
        _, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=8)
        assert c_pca.n_components_ == 8
        assert t_pca.n_components_ == 8

    def test_apply_pca(self, sample_X):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=8)
        result = apply_pca(X_sc, c_emb, t_emb, c_pca, t_pca)
        assert result.shape == (200, 5 + 8 + 8)

    def test_prepare_features(self, sample_X):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        assert X_combined.shape == (200, 5 + 4 + 4)
        assert len(names) == 5 + 4 + 4
        assert names[:5] == [f"scalar_{i}" for i in range(5)]

    def test_n_components_capped_by_samples(self):
        rng = np.random.RandomState(0)
        small = rng.randn(5, 384).astype(np.float32)
        c_pca, t_pca = fit_pca(small, small, n_components=50)
        assert c_pca.n_components_ <= 5


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class TestTrainLightGBM:
    def test_train_no_tune(self, sample_X, sample_y):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        model, params, n_est = train_lightgbm(X_combined, sample_y, names, tune=False)
        proba = model.predict(X_combined)
        assert proba.shape == (200,)
        assert 0.0 <= proba.min() and proba.max() <= 1.0
        assert isinstance(params, dict)
        assert "objective" in params
        assert "n_estimators" not in params
        assert n_est > 0

    def test_train_with_tune(self, sample_X, sample_y, sample_groups):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        model, params, n_est = train_lightgbm(
            X_combined, sample_y, names, groups=sample_groups, tune=True, n_trials=3,
        )
        proba = model.predict(X_combined)
        assert proba.shape == (200,)
        assert isinstance(params, dict)

    def test_returned_params_usable_for_oof(self, sample_X, sample_y, sample_groups):
        """Verify the params returned by train_lightgbm work with oof_predictions_lgb."""
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        model, params, n_est = train_lightgbm(X_combined, sample_y, names, tune=False)
        oof = oof_predictions_lgb(X_combined, sample_y, sample_groups, names, params, n_est)
        assert oof.shape == (200,)
        assert not np.any(np.isnan(oof))


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class TestTrainLogReg:
    def test_train(self, sample_X, sample_y, sample_groups):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        model, scaler = train_logreg(X_combined, sample_y, groups=sample_groups)
        proba = model.predict_proba(scaler.transform(X_combined))[:, 1]
        assert proba.shape == (200,)
        assert 0.0 <= proba.min() and proba.max() <= 1.0


# ---------------------------------------------------------------------------
# OOF Predictions
# ---------------------------------------------------------------------------

class TestOOFPredictions:
    def test_lgb_oof(self, sample_X, sample_y, sample_groups):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "verbosity": -1, "scale_pos_weight": 5.0,
            "max_depth": 4, "learning_rate": 0.1,
            "min_child_samples": 5, "random_state": 42,
        }
        oof = oof_predictions_lgb(X_combined, sample_y, sample_groups, names, params, 50)
        assert oof.shape == (200,)
        assert not np.any(np.isnan(oof))

    def test_logreg_oof(self, sample_X, sample_y, sample_groups):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, _ = prepare_features(sample_X, c_pca, t_pca)
        oof = oof_predictions_logreg(X_combined, sample_y, sample_groups, 1.0)
        assert oof.shape == (200,)
        assert not np.any(np.isnan(oof))


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

class TestThresholdTuning:
    def test_high_recall_target(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05, 0.01, 0.02, 0.03, 0.04])
        tau = tune_threshold(y_true, y_proba, min_recall=0.95)
        preds = (y_proba >= tau).astype(int)
        from sklearn.metrics import recall_score
        assert recall_score(y_true, preds) >= 0.66

    def test_returns_low_threshold_when_hard(self):
        y_true = np.array([1, 0, 0, 0, 0])
        y_proba = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        tau = tune_threshold(y_true, y_proba, min_recall=0.95)
        assert tau <= 0.50

    def test_perfect_model(self):
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.99, 0.98, 0.01, 0.02, 0.01, 0.02, 0.01, 0.01])
        tau = tune_threshold(y_true, y_proba, min_recall=0.95)
        assert tau >= 0.50


class TestThresholdSweep:
    def test_sweep_length(self):
        y_true = np.array([1, 0, 0, 0])
        y_proba = np.array([0.9, 0.1, 0.2, 0.3])
        results = threshold_sweep(y_true, y_proba, 4)
        assert len(results) == 99
        assert results[0]["threshold"] == 0.01
        assert results[-1]["threshold"] == 0.99

    def test_monotonic_recall(self):
        rng = np.random.RandomState(0)
        y_true = (rng.rand(100) > 0.8).astype(int)
        y_proba = rng.rand(100)
        results = threshold_sweep(y_true, y_proba, 100)
        recalls = [r["recall"] for r in results]
        for i in range(len(recalls) - 1):
            assert recalls[i] >= recalls[i + 1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_basic(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
        m = compute_metrics(y_true, y_proba, 0.5)
        assert m["recall"] == 1.0
        assert m["call_reduction"] == 0.6
        assert m["n_predicted"] == 2
        assert m["auroc"] is not None
        assert m["auprc"] is not None

    def test_all_positive(self):
        y_true = np.array([1, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.7])
        m = compute_metrics(y_true, y_proba, 0.5)
        assert m["recall"] == 1.0
        assert m["auroc"] is None


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_lgb_sorted(self, sample_X, sample_y):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        model, _, _ = train_lightgbm(X_combined, sample_y, names, tune=False)
        fi = feature_importance_lgb(model, names)
        assert len(fi) == len(names)
        imps = [f["importance"] for f in fi]
        assert imps == sorted(imps, reverse=True)

    def test_logreg_sorted(self, sample_X, sample_y):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        model, scaler = train_logreg(X_combined, sample_y)
        fi = feature_importance_logreg(model, names)
        assert len(fi) == len(names)
        imps = [f["importance"] for f in fi]
        assert imps == sorted(imps, reverse=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, sample_X, sample_y):
        X_sc, c_emb, t_emb = _split_embeddings(sample_X)
        c_pca, t_pca = fit_pca(c_emb, t_emb, n_components=4)
        X_combined, names = prepare_features(sample_X, c_pca, t_pca)
        model, _, _ = train_lightgbm(X_combined, sample_y, names, tune=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            save_model(
                path,
                model=model,
                change_pca=c_pca,
                test_pca=t_pca,
                threshold=0.25,
                feature_names=names,
                metadata={"test": True},
            )
            bundle = load_model(path)

        assert bundle["threshold"] == 0.25
        assert bundle["feature_names"] == names
        assert bundle["metadata"]["test"] is True
        assert bundle["change_pca"].n_components_ == 4
        proba = bundle["model"].predict(X_combined)
        np.testing.assert_array_equal(proba, model.predict(X_combined))
