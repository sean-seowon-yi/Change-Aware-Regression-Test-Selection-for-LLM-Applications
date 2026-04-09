"""Learned impact predictor — drop-in replacement for the rule-based version.

Loads a trained model bundle (LightGBM or LogReg), extracts the same features
used during training, and returns a ``PredictionResult`` with predicted test
IDs and per-test probabilities.

No ground truth or oracle information is used — only the prompt texts, test
metadata, and the pre-trained model artifacts.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase2.change_classifier import ClassifiedChange
from src.phase2.impact_predictor import PredictionResult, predict_impacted as rule_predict
from src.phase2.test_tagger import TaggedTest
from src.phase3.features import (
    aggregate_change_features,
    aggregate_pairwise_features,
    build_feature_row,
    embed_texts,
    extract_change_features,
    extract_pairwise_features,
    extract_test_features,
    pick_representative_change_embedding,
)
from src.phase3.trainer import load_model, prepare_features

import lightgbm as lgb
import xgboost as xgb


@functools.lru_cache(maxsize=8)
def _cached_load_model(model_path: str) -> dict:
    """Load model bundle with LRU cache to avoid re-reading from disk."""
    return load_model(model_path)


def predict_impacted(
    changes: list[ClassifiedChange],
    tagged_tests: list[TaggedTest],
    *,
    model_path: str | Path,
    threshold: float | None = None,
    magnitude_threshold: float = 0.005,
    sensitivity_profiles: dict | None = None,
) -> PredictionResult:
    """Learned impact predictor.

    Mirrors the interface of ``src.phase2.impact_predictor.predict_impacted``
    but uses a trained model instead of affinity rules.

    Args:
        changes: Classified changes between old and new prompt.
        tagged_tests: Enriched test cases from the test tagger.
        model_path: Path to a saved model bundle (``.pkl``).
        threshold: Classification threshold (lower -> higher recall).
            If ``None``, the tuned threshold stored in the model bundle
            is used; if the bundle has none, falls back to 0.20.
        magnitude_threshold: Skip all predictions if max change magnitude
            is below this value (return empty set).
        sensitivity_profiles: Optional ablation profiles keyed by test_id.

    Returns:
        PredictionResult with ``predicted_ids`` and ``reasons`` dict
        containing per-test predicted probabilities.
    """
    if not changes:
        return PredictionResult()

    max_mag = max(c.change.magnitude for c in changes)
    if max_mag < magnitude_threshold:
        return PredictionResult()

    bundle = _cached_load_model(str(model_path))
    model = bundle["model"]
    change_pca = bundle["change_pca"]
    test_pca = bundle["test_pca"]
    feature_names = bundle["feature_names"]
    scaler = bundle.get("scaler")
    if threshold is None:
        threshold = bundle.get("threshold", 0.20)

    is_compound = len({c.change.unit_type for c in changes}) > 1

    change_diff_texts = [c.change.content_diff for c in changes]
    change_embeddings = embed_texts(change_diff_texts)

    rule_pred = rule_predict(changes, tagged_tests)

    per_change_feats = [extract_change_features(c, is_compound) for c in changes]
    agg_change = aggregate_change_features(per_change_feats)
    rep_change_emb = pick_representative_change_embedding(changes, change_embeddings)

    test_texts = [t.input_text for t in tagged_tests]
    test_embeddings = embed_texts(test_texts)

    rows: list[dict] = []
    test_ids: list[str] = []

    for ti, t in enumerate(tagged_tests):
        per_change_pw = []
        for ci, c in enumerate(changes):
            pw = extract_pairwise_features(
                c, t,
                change_embeddings[ci], test_embeddings[ti],
                rule_pred=t.test_id in rule_pred.predicted_ids,
            )
            per_change_pw.append(pw)
        agg_pw = aggregate_pairwise_features(per_change_pw)

        sensitivity = (sensitivity_profiles or {}).get(t.test_id)
        t_feats = extract_test_features(t, sensitivity)

        scalar_row = build_feature_row(agg_change, t_feats, agg_pw)

        for dim in range(384):
            scalar_row[f"change_emb_{dim}"] = float(rep_change_emb[dim])
        for dim in range(384):
            scalar_row[f"test_emb_{dim}"] = float(test_embeddings[ti][dim])

        rows.append(scalar_row)
        test_ids.append(t.test_id)

    X = pd.DataFrame(rows)
    X_combined, _ = prepare_features(X, change_pca, test_pca)

    if isinstance(model, lgb.Booster):
        probas = model.predict(X_combined)
    elif isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X_combined, feature_names=feature_names)
        probas = model.predict(dmat)
    else:
        X_scaled = scaler.transform(X_combined) if scaler is not None else X_combined
        probas = model.predict_proba(X_scaled)[:, 1]

    predicted_ids: set[str] = set()
    reasons: dict[str, list[str]] = {}

    for i, tid in enumerate(test_ids):
        p = float(probas[i])
        if p >= threshold:
            predicted_ids.add(tid)
            reasons[tid] = [f"learned_model P(impacted)={p:.3f} >= {threshold}"]

    return PredictionResult(predicted_ids=predicted_ids, reasons=reasons)
