"""Validation-only policy fitting for return decisions.

Training produces similarity evidence, not a fraud label.  This module is the
only place that maps a validated score to MATCH, SUSPICIOUS, or
DIFFERENT_PRODUCT.  RETAKE is reserved for the capture-quality gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCORE_FIELDS = {"max_pair_score", "mean_rider_score", "lower_quartile_rider_score"}
DECISIONS = {"MATCH", "SUSPICIOUS", "DIFFERENT_PRODUCT", "RETAKE"}


@dataclass(frozen=True)
class DecisionPolicy:
    """A serialisable policy fitted only from validation-set comparison scores."""

    score_field: str
    different_product_threshold: float
    match_threshold: float
    target_match_false_positive_rate: float
    target_different_false_reject_rate: float
    positive_examples: int
    negative_examples: int
    observed_match_true_positive_rate: float
    observed_different_true_negative_rate: float
    version: int = 1

    def __post_init__(self) -> None:
        if self.score_field not in SCORE_FIELDS:
            raise ValueError(f"unknown score field {self.score_field!r}")
        if self.different_product_threshold >= self.match_threshold:
            raise ValueError("policy needs a non-empty suspicious band")
        if not 0.0 < self.target_match_false_positive_rate < 1.0:
            raise ValueError("target_match_false_positive_rate must be between zero and one")
        if not 0.0 < self.target_different_false_reject_rate < 1.0:
            raise ValueError("target_different_false_reject_rate must be between zero and one")
        if self.observed_match_true_positive_rate < 0.5 or self.observed_different_true_negative_rate < 0.5:
            raise ValueError("validation discrimination is too weak to deploy this policy")

    def decide(self, score: float) -> str:
        if not np.isfinite(score):
            raise ValueError("score must be finite")
        if score >= self.match_threshold:
            return "MATCH"
        if score <= self.different_product_threshold:
            return "DIFFERENT_PRODUCT"
        return "SUSPICIOUS"

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "DecisionPolicy":
        return cls(**json.loads(Path(path).read_text()))


def fit_decision_policy(
    scores: np.ndarray,
    same_product: np.ndarray,
    score_field: str = "mean_rider_score",
    target_match_false_positive_rate: float = 0.01,
    target_different_false_reject_rate: float = 0.01,
) -> DecisionPolicy:
    """Fit strict MATCH and DIFFERENT_PRODUCT thresholds from validation scores.

    A non-matching return must score above the match threshold no more than the
    configured false-positive budget.  A genuine return must score below the
    different-product threshold no more than the false-reject budget.  Anything
    between the two is deliberately routed to review instead of being guessed.
    """
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(same_product, dtype=bool)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("scores and same_product must be equally sized one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    positives = values[labels]
    negatives = values[~labels]
    if len(positives) < 2 or len(negatives) < 2:
        raise ValueError("need at least two positive and two negative validation comparisons")

    # Each target is an upper bound, not a request to spend the whole error
    # budget.  Use the two class boundaries to make a conservative three-way
    # policy: negatives establish the top of the DIFFERENT_PRODUCT region and
    # positives establish the bottom of MATCH.  That leaves all ambiguous
    # scores in an explicit review band.
    negative_ceiling = float(np.quantile(negatives, 1.0 - target_match_false_positive_rate, method="higher"))
    positive_floor = float(np.quantile(positives, target_different_false_reject_rate, method="lower"))
    different_threshold = min(negative_ceiling, positive_floor)
    match_threshold = max(negative_ceiling, positive_floor)
    match_tpr = float((positives >= match_threshold).mean())
    different_tnr = float((negatives <= different_threshold).mean())
    return DecisionPolicy(
        score_field=score_field,
        different_product_threshold=different_threshold,
        match_threshold=match_threshold,
        target_match_false_positive_rate=target_match_false_positive_rate,
        target_different_false_reject_rate=target_different_false_reject_rate,
        positive_examples=int(len(positives)),
        negative_examples=int(len(negatives)),
        observed_match_true_positive_rate=match_tpr,
        observed_different_true_negative_rate=different_tnr,
    )
