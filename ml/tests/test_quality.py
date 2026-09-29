"""Fast tests for model-free capture-quality checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quality import QualityThresholds, assess_image, laplacian_variance  # noqa: E402


def test_constant_image_is_blurry_and_dark():
    image = Image.fromarray(np.full((400, 400, 3), 10, dtype=np.uint8))
    result = assess_image(image)
    assert not result.passed
    assert "too_dark" in result.retake_reasons
    assert "too_blurry" in result.retake_reasons


def test_high_frequency_image_is_sharp():
    grid = (np.indices((400, 400)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = Image.fromarray(np.stack([grid, grid, grid], axis=-1))
    result = assess_image(image, QualityThresholds(min_laplacian_variance=40.0))
    assert result.laplacian_variance > 40.0
    assert "too_blurry" not in result.retake_reasons


def test_laplacian_requires_grayscale():
    with np.testing.assert_raises(ValueError):
        laplacian_variance(np.zeros((4, 4, 3), dtype=np.float32))
