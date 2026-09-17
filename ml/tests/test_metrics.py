"""Tests for the metric and split logic. No torch needed, these must stay fast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import recall_at_k, tpr_at_fpr, verification  # noqa: E402
from src.splits import split_groups  # noqa: E402


def test_recall_perfect():
    scores = np.eye(4)
    out = recall_at_k(scores, np.arange(4), ks=(1, 5))
    assert out["recall_at_1"] == 1.0
    assert out["median_rank"] == 1.0


def test_recall_worst_case():
    # Correct answer always ranked last.
    scores = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [1.0, 2.0, 0.0]])
    correct = np.array([0, 0, 2])
    out = recall_at_k(scores, correct, ks=(1, 5))
    assert out["recall_at_1"] == 0.0
    assert out["median_rank"] == 3.0


def test_tpr_at_fpr_separable():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert tpr_at_fpr(labels, scores, 0.01) == 1.0


def test_tpr_at_fpr_is_not_accuracy():
    # A degenerate scorer that ranks everything identically has no useful
    # operating point, even though thresholding it would look accurate on
    # a heavily imbalanced pair set.
    labels = np.array([0] * 99 + [1])
    scores = np.zeros(100)
    assert tpr_at_fpr(labels, scores, 0.01) == 0.0


def test_verification_single_class_is_nan():
    out = verification(np.ones(10, dtype=int), np.random.rand(10))
    assert np.isnan(out["auc"])


def _df(rows):
    cols = ["garment_id", "design_id", "lookalike_group"]
    return pd.DataFrame(rows, columns=cols)


def test_split_groups_joins_identical_units():
    df = _df([("g1", "d1", ""), ("g2", "d1", ""), ("g3", "d2", "")])
    groups = split_groups(df)
    assert groups["g1"] == groups["g2"]
    assert groups["g3"] != groups["g1"]


def test_split_groups_joins_lookalikes_across_designs():
    df = _df([("g1", "d1", "lg1"), ("g2", "d2", "lg1"), ("g3", "d3", "")])
    groups = split_groups(df)
    assert groups["g1"] == groups["g2"]
    assert groups["g3"] != groups["g1"]


def test_split_groups_chains_transitively():
    # g1 and g2 share a design, g2 and g3 share a lookalike group, so all three
    # must land in one split or the design leaks.
    df = _df([("g1", "d1", ""), ("g2", "d1", "lg1"), ("g3", "d9", "lg1")])
    groups = split_groups(df)
    assert groups["g1"] == groups["g2"] == groups["g3"]


def test_split_groups_ignores_blank_lookalike():
    df = _df([("g1", "d1", ""), ("g2", "d2", "")])
    groups = split_groups(df)
    assert groups["g1"] != groups["g2"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
