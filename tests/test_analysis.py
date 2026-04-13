"""Tests for src/phase4/analysis.py."""

from __future__ import annotations

import pytest

from src.phase4.analysis import paired_recall_series


def test_paired_recall_series_uses_effective_when_complete() -> None:
    rows_a = {
        "v01": {"effective_recall": 1.0, "recall_with_sentinel": 0.8},
        "v02": {"effective_recall": 0.95, "recall_with_sentinel": 0.9},
    }
    rows_b = {
        "v01": {"effective_recall": 1.0, "recall_with_sentinel": 0.85},
        "v02": {"effective_recall": 0.98, "recall_with_sentinel": 0.88},
    }
    a, b = paired_recall_series(rows_a, rows_b, ["v01", "v02"])
    assert a == [1.0, 0.95]
    assert b == [1.0, 0.98]


def test_paired_recall_series_falls_back_if_one_side_missing_effective() -> None:
    rows_a = {
        "v01": {"effective_recall": 1.0, "recall_with_sentinel": 0.8},
    }
    rows_b = {
        "v01": {"recall_with_sentinel": 0.85},
    }
    a, b = paired_recall_series(rows_a, rows_b, ["v01"])
    assert a == pytest.approx([0.8])
    assert b == pytest.approx([0.85])


def test_paired_recall_series_empty_versions() -> None:
    rows_a = {"v01": {"recall_with_sentinel": 0.8}}
    rows_b = {"v01": {"recall_with_sentinel": 0.9}}
    a, b = paired_recall_series(rows_a, rows_b, [])
    assert a == [] and b == []


def test_paired_recall_series_fallback_uses_recall_predictor() -> None:
    rows_a = {"v01": {"recall_predictor": 0.7}}
    rows_b = {"v01": {"recall_predictor": 0.75}}
    a, b = paired_recall_series(rows_a, rows_b, ["v01"])
    assert a == pytest.approx([0.7])
    assert b == pytest.approx([0.75])
