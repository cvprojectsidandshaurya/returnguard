"""Tests for local feature evidence matching."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.local_features import score_local_features  # noqa: E402


def test_blank_images_have_no_local_evidence():
    blank = np.zeros((256, 256), dtype=np.uint8)
    result = score_local_features(blank, blank)
    assert result.ratio_test_matches == 0
    assert result.geometric_inliers == 0


def test_translated_detail_has_geometric_inliers():
    rng = np.random.default_rng(13)
    reference = rng.integers(0, 256, size=(400, 400), dtype=np.uint8)
    transform = np.float32([[1, 0, 12], [0, 1, -8]])
    query = cv2.warpAffine(reference, transform, (400, 400))
    result = score_local_features(reference, query)
    assert result.ratio_test_matches >= 4
    assert result.geometric_inliers >= 4
    assert result.inlier_ratio > 0.5
