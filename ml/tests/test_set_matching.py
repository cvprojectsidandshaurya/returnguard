"""Tests for model-agnostic multi-view evidence scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.set_matching import score_multi_view  # noqa: E402


def test_score_multi_view_returns_strongest_pair_and_rider_coverage():
    packing = np.array([[1.0, 0.0], [0.0, 1.0]])
    rider = np.array([[0.9, 0.1], [0.0, 1.0]])
    result = score_multi_view(packing, rider)
    assert result.max_pair_score == pytest.approx(1.0)
    assert result.best_rider_index == 1
    assert result.best_packing_index == 1
    assert result.mean_rider_score < 1.0
    assert len(result.rider_best_scores) == 2


def test_score_multi_view_normalises_vectors_before_scoring():
    result = score_multi_view(np.array([[3.0, 0.0]]), np.array([[7.0, 0.0]]))
    assert result.max_pair_score == pytest.approx(1.0)


def test_score_multi_view_rejects_mismatched_or_empty_embeddings():
    with pytest.raises(ValueError, match="same dimension"):
        score_multi_view(np.ones((1, 2)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="non-empty"):
        score_multi_view(np.empty((0, 2)), np.ones((1, 2)))
