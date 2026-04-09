"""Unit tests for src/phase3/features.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.phase1.models import Change
from src.phase2.change_classifier import ClassifiedChange
from src.phase2.test_tagger import TaggedTest
from src.phase3.features import (
    ABLATION_SECTIONS,
    CHANGE_TYPES,
    MONITOR_TYPES,
    SENSITIVITY_CATS,
    SENSITIVE_TO_SECTIONS,
    UNIT_TYPES,
    aggregate_change_features,
    aggregate_pairwise_features,
    build_feature_row,
    embed_texts,
    extract_change_features,
    extract_pairwise_features,
    extract_test_features,
    pick_representative_change_embedding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tagged(
    test_id: str = "t_001",
    category: str = "format",
    *,
    monitor_type: str = "schema",
    tags: list[str] | None = None,
    sensitive_to: list[str] | None = None,
    key_deps: list[str] | None = None,
    demo_deps: list[str] | None = None,
) -> TaggedTest:
    return TaggedTest(
        test_id=test_id,
        input_text="test input text",
        monitor_type=monitor_type,
        tags=tags or [],
        sensitive_to=sensitive_to or [],
        inferred_key_deps=key_deps or [],
        inferred_demo_deps=demo_deps or [],
        sensitivity_category=category,
    )


def _make_change(
    unit_type: str = "format",
    change_type: str = "modified",
    magnitude: float = 0.10,
    affected_keys: list[str] | None = None,
    affected_demo_labels: list[str] | None = None,
) -> ClassifiedChange:
    return ClassifiedChange(
        change=Change(
            unit_id=f"{unit_type}_section",
            unit_type=unit_type,
            change_type=change_type,
            magnitude=magnitude,
            content_diff="some diff text",
        ),
        affected_keys=affected_keys or [],
        affected_demo_labels=affected_demo_labels or [],
    )


# ---------------------------------------------------------------------------
# extract_change_features
# ---------------------------------------------------------------------------

class TestExtractChangeFeatures:

    def test_one_hot_unit_type(self):
        feats = extract_change_features(_make_change("policy"), is_compound=False)
        for name in UNIT_TYPES:
            expected = 1 if name == "policy" else 0
            assert feats[f"unit_type_{name}"] == expected

    def test_one_hot_change_type(self):
        feats = extract_change_features(
            _make_change(change_type="deleted"), is_compound=False,
        )
        for name in CHANGE_TYPES:
            expected = 1 if name == "deleted" else 0
            assert feats[f"change_type_{name}"] == expected

    def test_magnitude(self):
        feats = extract_change_features(_make_change(magnitude=0.42), is_compound=False)
        assert feats["magnitude"] == pytest.approx(0.42)

    def test_affected_counts(self):
        c = _make_change(affected_keys=["a", "b"], affected_demo_labels=["d1"])
        feats = extract_change_features(c, is_compound=False)
        assert feats["num_affected_keys"] == 2
        assert feats["num_affected_demo_labels"] == 1

    def test_is_compound_flag(self):
        feats_no = extract_change_features(_make_change(), is_compound=False)
        feats_yes = extract_change_features(_make_change(), is_compound=True)
        assert feats_no["is_compound"] == 0
        assert feats_yes["is_compound"] == 1

    def test_total_column_count(self):
        feats = extract_change_features(_make_change(), is_compound=False)
        assert len(feats) == 15


# ---------------------------------------------------------------------------
# extract_test_features
# ---------------------------------------------------------------------------

class TestExtractTestFeatures:

    def test_one_hot_monitor_type(self):
        t = _make_tagged(monitor_type="keyword_presence")
        feats = extract_test_features(t)
        for name in MONITOR_TYPES:
            expected = 1 if name == "keyword_presence" else 0
            assert feats[f"monitor_type_{name}"] == expected

    def test_one_hot_sensitivity_category(self):
        t = _make_tagged(category="demo")
        feats = extract_test_features(t)
        for name in SENSITIVITY_CATS:
            expected = 1 if name == "demo" else 0
            assert feats[f"sensitivity_category_{name}"] == expected

    def test_dep_counts(self):
        t = _make_tagged(key_deps=["action", "priority"], demo_deps=["Example 1"])
        feats = extract_test_features(t)
        assert feats["num_inferred_key_deps"] == 2
        assert feats["num_inferred_demo_deps"] == 1

    def test_sensitive_to_flags(self):
        t = _make_tagged(sensitive_to=["format", "policy"])
        feats = extract_test_features(t)
        assert feats["has_sensitive_to_format"] == 1
        assert feats["has_sensitive_to_policy"] == 1
        assert feats["has_sensitive_to_demonstration"] == 0
        assert feats["has_sensitive_to_workflow"] == 0

    def test_ablation_without_profile(self):
        feats = extract_test_features(_make_tagged(), sensitivity=None)
        for sec in ABLATION_SECTIONS:
            assert feats[f"ablation_sensitive_{sec}"] == -1

    def test_ablation_with_profile(self):
        profile = {"role": False, "format": True, "demonstrations": False,
                    "policy": True, "workflow": False}
        feats = extract_test_features(_make_tagged(), sensitivity=profile)
        assert feats["ablation_sensitive_role"] == 0
        assert feats["ablation_sensitive_format"] == 1
        assert feats["ablation_sensitive_demonstrations"] == 0
        assert feats["ablation_sensitive_policy"] == 1
        assert feats["ablation_sensitive_workflow"] == 0

    def test_total_column_count_without_ablation(self):
        feats = extract_test_features(_make_tagged(), sensitivity=None)
        # 6 monitor + 5 sensitivity_cat + 2 dep counts + 4 sensitive_to + 5 ablation = 22
        assert len(feats) == 22

    def test_total_column_count_with_ablation(self):
        profile = {sec: False for sec in ABLATION_SECTIONS}
        feats = extract_test_features(_make_tagged(), sensitivity=profile)
        assert len(feats) == 22


# ---------------------------------------------------------------------------
# extract_pairwise_features
# ---------------------------------------------------------------------------

class TestExtractPairwiseFeatures:

    def test_cosine_similarity(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        feats = extract_pairwise_features(
            _make_change(), _make_tagged(), a, b, rule_pred=False,
        )
        assert feats["cosine_similarity"] == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        feats = extract_pairwise_features(
            _make_change(), _make_tagged(), a, b, rule_pred=False,
        )
        assert feats["cosine_similarity"] == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_zero_vector(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        feats = extract_pairwise_features(
            _make_change(), _make_tagged(), a, b, rule_pred=False,
        )
        assert feats["cosine_similarity"] == 0.0

    def test_key_overlap(self):
        c = _make_change(affected_keys=["action", "priority"])
        t = _make_tagged(key_deps=["action", "summary"])
        emb = np.zeros(3, dtype=np.float32)
        feats = extract_pairwise_features(c, t, emb, emb, rule_pred=False)
        assert feats["key_overlap"] == 1

    def test_no_key_overlap(self):
        c = _make_change(affected_keys=["action"])
        t = _make_tagged(key_deps=["summary"])
        emb = np.zeros(3, dtype=np.float32)
        feats = extract_pairwise_features(c, t, emb, emb, rule_pred=False)
        assert feats["key_overlap"] == 0

    def test_demo_overlap(self):
        c = _make_change(affected_demo_labels=["Example 1"])
        t = _make_tagged(demo_deps=["Example 1", "Example 2"])
        emb = np.zeros(3, dtype=np.float32)
        feats = extract_pairwise_features(c, t, emb, emb, rule_pred=False)
        assert feats["demo_overlap"] == 1

    def test_section_match_format(self):
        c = _make_change("format")
        t = _make_tagged(category="format")
        emb = np.zeros(3, dtype=np.float32)
        feats = extract_pairwise_features(c, t, emb, emb, rule_pred=False)
        assert feats["section_match"] == 1

    def test_section_match_demo(self):
        c = _make_change("demo_item")
        t = _make_tagged(category="demo")
        emb = np.zeros(3, dtype=np.float32)
        feats = extract_pairwise_features(c, t, emb, emb, rule_pred=False)
        assert feats["section_match"] == 1

    def test_section_no_match(self):
        c = _make_change("format")
        t = _make_tagged(category="policy")
        emb = np.zeros(3, dtype=np.float32)
        feats = extract_pairwise_features(c, t, emb, emb, rule_pred=False)
        assert feats["section_match"] == 0

    def test_rule_based_prediction(self):
        emb = np.zeros(3, dtype=np.float32)
        feats_yes = extract_pairwise_features(
            _make_change(), _make_tagged(), emb, emb, rule_pred=True,
        )
        feats_no = extract_pairwise_features(
            _make_change(), _make_tagged(), emb, emb, rule_pred=False,
        )
        assert feats_yes["rule_based_prediction"] == 1
        assert feats_no["rule_based_prediction"] == 0

    def test_total_column_count(self):
        emb = np.zeros(3, dtype=np.float32)
        feats = extract_pairwise_features(
            _make_change(), _make_tagged(), emb, emb, rule_pred=False,
        )
        assert len(feats) == 5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregateChangeFeatures:

    def test_single_change(self):
        feats = [extract_change_features(_make_change("format"), is_compound=False)]
        agg = aggregate_change_features(feats)
        assert agg["unit_type_format"] == 1
        assert agg["unit_type_policy"] == 0
        assert agg["magnitude"] == pytest.approx(0.10)

    def test_compound_changes(self):
        f1 = extract_change_features(
            _make_change("format", magnitude=0.20, affected_keys=["a"]),
            is_compound=True,
        )
        f2 = extract_change_features(
            _make_change("policy", magnitude=0.50, affected_keys=["b", "c"]),
            is_compound=True,
        )
        agg = aggregate_change_features([f1, f2])
        assert agg["unit_type_format"] == 1
        assert agg["unit_type_policy"] == 1
        assert agg["magnitude"] == pytest.approx(0.50)
        assert agg["num_affected_keys"] == 3  # 1 + 2
        assert agg["is_compound"] == 1

    def test_empty_list(self):
        agg = aggregate_change_features([])
        assert agg["magnitude"] == 0.0
        assert agg["is_compound"] == 0


class TestAggregatePairwiseFeatures:

    def test_max_across_changes(self):
        pw1 = {"cosine_similarity": 0.3, "key_overlap": 0, "demo_overlap": 0,
               "section_match": 0, "rule_based_prediction": 0}
        pw2 = {"cosine_similarity": 0.8, "key_overlap": 1, "demo_overlap": 0,
               "section_match": 1, "rule_based_prediction": 1}
        agg = aggregate_pairwise_features([pw1, pw2])
        assert agg["cosine_similarity"] == pytest.approx(0.8)
        assert agg["key_overlap"] == 1
        assert agg["section_match"] == 1

    def test_empty_list(self):
        agg = aggregate_pairwise_features([])
        assert agg["cosine_similarity"] == 0


# ---------------------------------------------------------------------------
# build_feature_row
# ---------------------------------------------------------------------------

class TestBuildFeatureRow:

    def test_merges_all_groups(self):
        change = {"magnitude": 0.5, "unit_type_format": 1}
        test = {"monitor_type_schema": 1}
        pair = {"cosine_similarity": 0.7}
        row = build_feature_row(change, test, pair)
        assert row["magnitude"] == 0.5
        assert row["monitor_type_schema"] == 1
        assert row["cosine_similarity"] == 0.7

    def test_expected_column_count(self):
        c = extract_change_features(_make_change(), is_compound=False)
        agg_c = aggregate_change_features([c])
        t = extract_test_features(_make_tagged(), sensitivity={sec: False for sec in ABLATION_SECTIONS})
        emb = np.zeros(3, dtype=np.float32)
        pw = extract_pairwise_features(
            _make_change(), _make_tagged(), emb, emb, rule_pred=False,
        )
        agg_pw = aggregate_pairwise_features([pw])
        row = build_feature_row(agg_c, t, agg_pw)
        # 15 change + 22 test + 5 pairwise = 42 scalar features
        assert len(row) == 15 + 22 + 5


# ---------------------------------------------------------------------------
# pick_representative_change_embedding
# ---------------------------------------------------------------------------

class TestPickRepresentativeChangeEmbedding:

    def test_highest_magnitude(self):
        c1 = _make_change(magnitude=0.1)
        c2 = _make_change(magnitude=0.9)
        embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rep = pick_representative_change_embedding([c1, c2], embs)
        np.testing.assert_array_equal(rep, [0.0, 1.0])

    def test_empty_changes(self):
        rep = pick_representative_change_embedding([], np.empty((0, 384), dtype=np.float32))
        assert rep.shape == (384,)
        assert np.all(rep == 0.0)


# ---------------------------------------------------------------------------
# embed_texts (integration — uses real model)
# ---------------------------------------------------------------------------

class TestEmbedTexts:

    @pytest.mark.slow
    def test_embedding_shape(self):
        """Verify the real model produces 384-dim embeddings."""
        result = embed_texts(["Hello world", "Another sentence"])
        assert result.shape == (2, 384)
        assert result.dtype == np.float32

    def test_empty_input(self):
        result = embed_texts([])
        assert result.shape == (0, 384)
