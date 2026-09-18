"""Cheap, deterministic image-quality checks for packing and rider captures.

These are deliberately model-free. They run before embedding so a rider can be
asked for a retake instead of turning a bad photo into a misleading match score.
They do not decide whether a garment is present; that belongs to the later
segmentation/detector stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityThresholds:
    """Thresholds should be fitted on captured validation photos, not guessed."""

    min_width: int = 320
    min_height: int = 320
    min_brightness: float = 35.0
    max_brightness: float = 235.0
    min_laplacian_variance: float = 40.0


@dataclass(frozen=True)
class QualityResult:
    width: int
    height: int
    brightness: float
    laplacian_variance: float
    passed: bool
    retake_reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        result = asdict(self)
        result["retake_reasons"] = list(self.retake_reasons)
        return result


def _grayscale(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian, a standard inexpensive blur cue."""
    if gray.ndim != 2:
        raise ValueError("laplacian_variance expects a two-dimensional grayscale array")
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    )
    return float(laplacian.var())


def assess_image(image: Image.Image, thresholds: QualityThresholds = QualityThresholds()) -> QualityResult:
    """Return measurements and retake reasons for one image."""
    width, height = image.size
    gray = _grayscale(image)
    brightness = float(gray.mean())
    sharpness = laplacian_variance(gray)
    reasons: list[str] = []
    if width < thresholds.min_width or height < thresholds.min_height:
        reasons.append("resolution_too_low")
    if brightness < thresholds.min_brightness:
        reasons.append("too_dark")
    if brightness > thresholds.max_brightness:
        reasons.append("too_bright")
    if sharpness < thresholds.min_laplacian_variance:
        reasons.append("too_blurry")
    return QualityResult(width, height, brightness, sharpness, not reasons, tuple(reasons))


def assess_path(path: Path, thresholds: QualityThresholds = QualityThresholds()) -> QualityResult:
    """Assess a path without leaving an image file open."""
    with Image.open(path) as image:
        return assess_image(image, thresholds)
