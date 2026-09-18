"""Multi-view matching primitives for a packing set and a rider-photo set.

This module returns evidence scores only. It deliberately does not turn a score
into MATCH or DIFFERENT_PRODUCT because those thresholds must be calibrated by
the evaluation owner on validation data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MultiViewScore:
    """Scores and the strongest image pair in a packing-versus-rider comparison."""

    max_pair_score: float
    mean_rider_score: float
    lower_quartile_rider_score: float
    best_rider_index: int
    best_packing_index: int
    rider_best_scores: tuple[float, ...]

    def as_dict(self) -> dict:
        result = asdict(self)
        result["rider_best_scores"] = list(self.rider_best_scores)
        return result


def _normalise(vectors: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{label} embeddings must be a non-empty two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{label} embeddings include a zero vector")
    return array / norms


def score_multi_view(packing_embeddings: np.ndarray, rider_embeddings: np.ndarray) -> MultiViewScore:
    """Score every rider image against every packing image with cosine similarity.

    ``max_pair_score`` is compatible with the project's established max-pooling
    baseline. The mean and lower-quartile rider scores are retained as evidence
    for later calibration: a single clear photo should not hide a set where all
    other rider views disagree.
    """
    packing = _normalise(packing_embeddings, "packing")
    rider = _normalise(rider_embeddings, "rider")
    if packing.shape[1] != rider.shape[1]:
        raise ValueError("packing and rider embeddings must have the same dimension")

    similarities = rider @ packing.T
    best_packing_for_rider = similarities.argmax(axis=1)
    rider_best = similarities[np.arange(len(rider)), best_packing_for_rider]
    best_rider_index = int(rider_best.argmax())
    best_packing_index = int(best_packing_for_rider[best_rider_index])
    return MultiViewScore(
        max_pair_score=float(rider_best[best_rider_index]),
        mean_rider_score=float(rider_best.mean()),
        lower_quartile_rider_score=float(np.quantile(rider_best, 0.25)),
        best_rider_index=best_rider_index,
        best_packing_index=best_packing_index,
        rider_best_scores=tuple(float(value) for value in rider_best),
    )
