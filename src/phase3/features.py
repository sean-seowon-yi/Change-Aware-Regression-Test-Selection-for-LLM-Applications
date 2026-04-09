"""Feature extraction for the learned selector.

Transforms ClassifiedChange and TaggedTest objects into fixed-width numeric
feature vectors.  Four feature groups are produced:

  1. Change features  – what changed in the prompt (14 scalars)
  2. Test features    – what the test checks (17 scalars + 5 ablation = 22)
  3. Pairwise features – relationship between a change and a test (5 scalars)
  4. Embeddings       – raw 384-dim vectors (PCA applied later in training)

Version-level aggregation collapses per-change features when a version
contains multiple changes, yielding one row per (version, test) pair.
"""

from __future__ import annotations

import functools
from typing import Sequence

import numpy as np

from src.phase2.change_classifier import ClassifiedChange
from src.phase2.test_tagger import TaggedTest

# ---------------------------------------------------------------------------
# Vocabulary constants (order determines one-hot column positions)
# ---------------------------------------------------------------------------

UNIT_TYPES: list[str] = [
    "format", "demonstration", "demonstrations", "demo_item", "policy", "workflow", "role",
]
CHANGE_TYPES: list[str] = ["inserted", "deleted", "modified", "reordered"]
MONITOR_TYPES: list[str] = [
    "schema", "required_keys", "keyword_presence", "policy", "format",
    "code_execution",
]
SENSITIVITY_CATS: list[str] = ["format", "policy", "demo", "workflow", "general"]
SENSITIVE_TO_SECTIONS: list[str] = ["format", "demonstration", "policy", "workflow"]
ABLATION_SECTIONS: list[str] = ["role", "format", "demonstrations", "policy", "workflow"]

_UNIT_TO_SENSITIVITY: dict[str, str] = {
    "format": "format",
    "policy": "policy",
    "demonstration": "demo",
    "demonstrations": "demo",
    "demo_item": "demo",
    "workflow": "workflow",
    "role": "general",
}

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_embedder():
    """Lazy-loaded singleton for the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Batch-encode texts into (N, 384) float32 embeddings."""
    model = get_embedder()
    if len(texts) == 0:
        return np.empty((0, 384), dtype=np.float32)
    return model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Per-change scalar features (15 keys)
# ---------------------------------------------------------------------------

def extract_change_features(
    change: ClassifiedChange,
    is_compound: bool,
) -> dict:
    """Extract scalar features for a single ClassifiedChange.

    Returns a dict with 15 keys:
      7 unit_type one-hots + 4 change_type one-hots +
      magnitude + num_affected_keys + num_affected_demo_labels + is_compound
    """
    feats: dict = {}
    ut = change.change.unit_type
    for name in UNIT_TYPES:
        feats[f"unit_type_{name}"] = int(ut == name)

    ct = change.change.change_type
    for name in CHANGE_TYPES:
        feats[f"change_type_{name}"] = int(ct == name)

    feats["magnitude"] = change.change.magnitude
    feats["num_affected_keys"] = len(change.affected_keys)
    feats["num_affected_demo_labels"] = len(change.affected_demo_labels)
    feats["is_compound"] = int(is_compound)
    return feats


# ---------------------------------------------------------------------------
# Per-test scalar features (17 base + 5 ablation = 22 features)
# ---------------------------------------------------------------------------

def extract_test_features(
    test: TaggedTest,
    sensitivity: dict | None = None,
) -> dict:
    """Extract scalar features for a single TaggedTest.

    Args:
        test: Enriched test from the test tagger.
        sensitivity: Optional ablation profile, e.g.
            {"role": True, "format": False, ...}.  If None the 5 ablation
            columns are filled with -1 (missing indicator).
    """
    feats: dict = {}
    for name in MONITOR_TYPES:
        feats[f"monitor_type_{name}"] = int(test.monitor_type == name)

    for name in SENSITIVITY_CATS:
        feats[f"sensitivity_category_{name}"] = int(test.sensitivity_category == name)

    feats["num_inferred_key_deps"] = len(test.inferred_key_deps)
    feats["num_inferred_demo_deps"] = len(test.inferred_demo_deps)

    for sec in SENSITIVE_TO_SECTIONS:
        feats[f"has_sensitive_to_{sec}"] = int(sec in test.sensitive_to)

    for sec in ABLATION_SECTIONS:
        if sensitivity is not None:
            feats[f"ablation_sensitive_{sec}"] = int(sensitivity.get(sec, False))
        else:
            feats[f"ablation_sensitive_{sec}"] = -1

    return feats


# ---------------------------------------------------------------------------
# Pairwise features (5 features)
# ---------------------------------------------------------------------------

def extract_pairwise_features(
    change: ClassifiedChange,
    test: TaggedTest,
    change_emb: np.ndarray,
    test_emb: np.ndarray,
    rule_pred: bool,
) -> dict:
    """Compute relationship features for a (change, test) pair.

    Args:
        change: A single classified change.
        test: A single tagged test.
        change_emb: Raw 384-dim embedding of the change's content_diff.
        test_emb: Raw 384-dim embedding of the test's input_text.
        rule_pred: Whether the Phase 2 rule-based predictor selected this
            test for this change's version.
    """
    section_match = (
        _UNIT_TO_SENSITIVITY.get(change.change.unit_type, "general")
        == test.sensitivity_category
    )
    return {
        "cosine_similarity": _cosine_sim(change_emb, test_emb),
        "key_overlap": int(bool(set(change.affected_keys) & set(test.inferred_key_deps))),
        "demo_overlap": int(bool(set(change.affected_demo_labels) & set(test.inferred_demo_deps))),
        "section_match": int(section_match),
        "rule_based_prediction": int(rule_pred),
    }


# ---------------------------------------------------------------------------
# Version-level aggregation
# ---------------------------------------------------------------------------

_ONE_HOT_CHANGE_COLS = (
    [f"unit_type_{n}" for n in UNIT_TYPES]
    + [f"change_type_{n}" for n in CHANGE_TYPES]
)
_SUM_CHANGE_COLS = ["num_affected_keys", "num_affected_demo_labels"]
_MAX_CHANGE_COLS = ["magnitude"]


def aggregate_change_features(per_change_feats: list[dict]) -> dict:
    """Aggregate per-change scalar features to a single version-level dict.

    Aggregation rules (from PHASE3.md §12):
      - one-hot columns: max (OR) across changes
      - magnitude: max
      - num_affected_keys / num_affected_demo_labels: sum
      - is_compound: set to 1 if >1 distinct unit_type
    """
    if not per_change_feats:
        agg = {col: 0 for col in _ONE_HOT_CHANGE_COLS}
        for col in _SUM_CHANGE_COLS:
            agg[col] = 0
        agg["magnitude"] = 0.0
        agg["is_compound"] = 0
        return agg

    agg: dict = {}
    for col in _ONE_HOT_CHANGE_COLS:
        agg[col] = max(f[col] for f in per_change_feats)
    for col in _SUM_CHANGE_COLS:
        agg[col] = sum(f[col] for f in per_change_feats)
    agg["magnitude"] = max(f["magnitude"] for f in per_change_feats)

    distinct_units = {
        ut for f in per_change_feats for ut in UNIT_TYPES if f.get(f"unit_type_{ut}", 0)
    }
    agg["is_compound"] = int(len(distinct_units) > 1)
    return agg


_ALL_PAIRWISE_COLS = [
    "cosine_similarity", "key_overlap", "demo_overlap",
    "section_match", "rule_based_prediction",
]


def aggregate_pairwise_features(per_change_pairwise: list[dict]) -> dict:
    """Aggregate per-change pairwise features to version level via max."""
    if not per_change_pairwise:
        return {col: 0 for col in _ALL_PAIRWISE_COLS}
    return {
        col: max(f[col] for f in per_change_pairwise)
        for col in _ALL_PAIRWISE_COLS
    }


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def build_feature_row(
    agg_change: dict,
    test_feats: dict,
    agg_pairwise: dict,
) -> dict:
    """Merge the three feature groups into a single flat dict."""
    row: dict = {}
    row.update(agg_change)
    row.update(test_feats)
    row.update(agg_pairwise)
    return row


# ---------------------------------------------------------------------------
# Embedding aggregation (highest-magnitude change)
# ---------------------------------------------------------------------------

def pick_representative_change_embedding(
    changes: list[ClassifiedChange],
    change_embeddings: np.ndarray,
) -> np.ndarray:
    """Select the embedding of the highest-magnitude change in a version.

    If there are no changes returns a zero vector.
    """
    if len(changes) == 0:
        return np.zeros(384, dtype=np.float32)
    magnitudes = [c.change.magnitude for c in changes]
    idx = int(np.argmax(magnitudes))
    return change_embeddings[idx]
