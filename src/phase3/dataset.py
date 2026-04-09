"""Build the labelled training dataset for the learned selector.

Loads Phase 1 ground truth, runs the Phase 2 classifier/tagger on each
prompt version, extracts features, and produces a (version, test)-level
DataFrame ready for model training.

Ground truth labels are used *only* as the target variable ``y``.
All feature columns are derived from observable prompt text and test metadata
(no oracle leakage).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.phase1.prompts.parser import parse_prompt
from src.phase2.change_classifier import classify_changes
from src.phase2.evaluator import (
    GroundTruth,
    list_version_ids,
    load_base_prompt,
    load_ground_truth,
    load_test_cases,
    load_version_prompt,
)
from src.phase2.impact_predictor import predict_impacted
from src.phase2.test_tagger import tag_tests
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Sensitivity profile loader
# ---------------------------------------------------------------------------

def load_sensitivity_profiles(domain: str) -> dict[str, dict[str, bool]]:
    """Load ablation sensitivity profiles from data/{domain}/sensitivity_profiles.json.

    Returns an empty dict if the file does not exist (ablation not yet run).
    """
    path = _DATA_DIR / domain / "sensitivity_profiles.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _version_number(vid: str) -> int:
    m = re.search(r"\d+", vid)
    return int(m.group()) if m else 0


def _apply_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter rows by the requested split."""
    if split == "all":
        return df
    nums = df["_version_id"].map(_version_number)
    if split == "temporal_train":
        return df[nums <= 50].reset_index(drop=True)
    if split == "temporal_test":
        return df[nums > 50].reset_index(drop=True)
    raise ValueError(f"Unknown split: {split!r}")


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    domain: str,
    *,
    split: str = "all",
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the (X, y, metadata) triple for one domain.

    Args:
        domain: ``"domain_a"`` or ``"domain_b"``.
        split: ``"all"`` | ``"temporal_train"`` (v01-v50) |
               ``"temporal_test"`` (v51-v70).
        show_progress: Show tqdm progress bar.

    Returns:
        X: Feature DataFrame (scalar + embedding columns).
        y: Binary Series (1 = impacted).
        metadata: DataFrame with ``version_id``, ``test_id``, ``domain``.
    """
    gt: GroundTruth = load_ground_truth(domain)
    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    sensitivity_profiles = load_sensitivity_profiles(domain)

    base_sections = parse_prompt(base_prompt)
    demos_text = base_sections.get("demonstrations", "")
    tagged_tests = tag_tests(test_cases, demos_text)

    test_texts = [t.input_text for t in tagged_tests]
    test_embeddings = embed_texts(test_texts)
    test_emb_by_id = {
        t.test_id: test_embeddings[i] for i, t in enumerate(tagged_tests)
    }

    version_ids = list_version_ids(domain)
    rows: list[dict] = []

    iterator = tqdm(version_ids, desc=f"[{domain}] Building features") if show_progress else version_ids
    for vid in iterator:
        if vid not in gt.impacted_by_version:
            continue

        version_prompt = load_version_prompt(domain, vid)
        changes = classify_changes(base_prompt, version_prompt)

        is_compound = len({c.change.unit_type for c in changes}) > 1

        if changes:
            change_diff_texts = [c.change.content_diff for c in changes]
            change_embeddings = embed_texts(change_diff_texts)
        else:
            change_embeddings = np.empty((0, 384), dtype=np.float32)

        rule_pred = predict_impacted(changes, tagged_tests)

        per_change_feats = [
            extract_change_features(c, is_compound) for c in changes
        ]
        agg_change = aggregate_change_features(per_change_feats)

        rep_change_emb = pick_representative_change_embedding(changes, change_embeddings)

        for t in tagged_tests:
            per_change_pw = []
            for ci, c in enumerate(changes):
                pw = extract_pairwise_features(
                    c, t,
                    change_embeddings[ci], test_emb_by_id[t.test_id],
                    rule_pred=t.test_id in rule_pred.predicted_ids,
                )
                per_change_pw.append(pw)
            agg_pw = aggregate_pairwise_features(per_change_pw)

            sensitivity = sensitivity_profiles.get(t.test_id)
            t_feats = extract_test_features(t, sensitivity)

            scalar_row = build_feature_row(agg_change, t_feats, agg_pw)

            for dim in range(384):
                scalar_row[f"change_emb_{dim}"] = float(rep_change_emb[dim])
            test_emb = test_emb_by_id[t.test_id]
            for dim in range(384):
                scalar_row[f"test_emb_{dim}"] = float(test_emb[dim])

            impacted_set = gt.impacted_by_version.get(vid, set())
            scalar_row["_label"] = int(t.test_id in impacted_set)
            scalar_row["_version_id"] = vid
            scalar_row["_test_id"] = t.test_id
            scalar_row["_domain"] = domain

            rows.append(scalar_row)

    df = pd.DataFrame(rows)
    df = _apply_split(df, split)

    meta_cols = ["_version_id", "_test_id", "_domain"]
    label_col = "_label"

    metadata = df[meta_cols].rename(
        columns={"_version_id": "version_id", "_test_id": "test_id", "_domain": "domain"}
    ).reset_index(drop=True)

    y = df[label_col].reset_index(drop=True).astype(int)
    X = df.drop(columns=meta_cols + [label_col]).reset_index(drop=True)

    return X, y, metadata
