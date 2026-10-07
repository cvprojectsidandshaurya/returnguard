"""Validation-only policy fitting for product or physical-unit decisions.

Training produces similarity evidence, not a fraud label.  This module is the
only place that maps a validated score to MATCH, SUSPICIOUS, or an explicit
identity-policy-specific non-match. RETAKE is reserved for a rider quality failure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCORE_FIELDS = {"max_pair_score", "mean_rider_score", "lower_quartile_rider_score"}
IDENTITY_POLICIES = {"product", "unit"}
DECISIONS = {"MATCH", "SUSPICIOUS", "DIFFERENT_PRODUCT", "DIFFERENT_UNIT", "RETAKE", "REFERENCE_INVALID"}


@dataclass(frozen=True)
class DecisionPolicy:
    """A serialisable policy fitted only from validation-set comparison scores."""

    score_field: str
    identity_policy: str
    different_identity_threshold: float
    match_threshold: float
    target_match_false_positive_rate: float
    target_different_false_reject_rate: float
    positive_examples: int
    negative_examples: int
    observed_match_true_positive_rate: float
    observed_different_true_negative_rate: float
    is_deployable: bool
    version: int = 2

    def __post_init__(self) -> None:
        if self.score_field not in SCORE_FIELDS:
            raise ValueError(f"unknown score field {self.score_field!r}")
        if self.identity_policy not in IDENTITY_POLICIES:
            raise ValueError(f"unknown identity policy {self.identity_policy!r}")
        if not np.isfinite((self.different_identity_threshold, self.match_threshold)).all():
            raise ValueError("policy thresholds must be finite")
        if self.different_identity_threshold > self.match_threshold:
            raise ValueError("different-identity threshold may not exceed match threshold")
        if not 0.0 < self.target_match_false_positive_rate < 1.0:
            raise ValueError("target_match_false_positive_rate must be between zero and one")
        if not 0.0 < self.target_different_false_reject_rate < 1.0:
            raise ValueError("target_different_false_reject_rate must be between zero and one")
    def assert_deployable(self) -> None:
        """Refuse inference for an artifact that calibration marked unsafe."""
        if not self.is_deployable:
            raise ValueError(
                "calibration policy is not deployable; collect more validation data or improve separation before serving it"
            )
        if self.different_identity_threshold >= self.match_threshold:
            raise ValueError("deployable policy needs a non-empty suspicious band")

    def decide(self, score: float) -> str:
        if not np.isfinite(score):
            raise ValueError("score must be finite")
        if score >= self.match_threshold:
            return "MATCH"
        if score <= self.different_identity_threshold:
            return "DIFFERENT_PRODUCT" if self.identity_policy == "product" else "DIFFERENT_UNIT"
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
    same_identity: np.ndarray,
    identity_policy: str = "product",
    score_field: str = "mean_rider_score",
    target_match_false_positive_rate: float = 0.01,
    target_different_false_reject_rate: float = 0.01,
    min_positive_examples: int = 2,
    min_negative_examples: int = 2,
) -> DecisionPolicy:
    """Fit strict MATCH and policy-specific non-match thresholds from validation scores.

    A non-matching return must score above the match threshold no more than the
    configured false-positive budget. A genuine return must score below the
    different-identity threshold no more than the false-reject budget. Anything
    between the two is deliberately routed to review instead of being guessed.
    """
    values = np.asarray(scores, dtype=float)
    if identity_policy not in IDENTITY_POLICIES:
        raise ValueError(f"identity_policy must be one of {sorted(IDENTITY_POLICIES)}")
    labels = np.asarray(same_identity, dtype=bool)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("scores and same_identity must be equally sized one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    positives = values[labels]
    negatives = values[~labels]
    if min_positive_examples < 2 or min_negative_examples < 2:
        raise ValueError("minimum calibration counts must each be at least two")
    if len(positives) < 2 or len(negatives) < 2:
        raise ValueError("need at least two positive and two negative validation comparisons")

    # Each target is an upper bound, not a request to spend the whole error
    # budget.  Use the two class boundaries to make a conservative three-way
    # policy: negatives establish the top of the non-match region and
    # positives establish the bottom of MATCH.  That leaves all ambiguous
    # scores in an explicit review band.
    negative_ceiling = float(np.quantile(negatives, 1.0 - target_match_false_positive_rate, method="higher"))
    positive_floor = float(np.quantile(positives, target_different_false_reject_rate, method="lower"))
    different_threshold = min(negative_ceiling, positive_floor)
    match_threshold = max(negative_ceiling, positive_floor)
    match_tpr = float((positives >= match_threshold).mean())
    different_tnr = float((negatives <= different_threshold).mean())
    deployable = (
        len(positives) >= min_positive_examples
        and len(negatives) >= min_negative_examples
        and match_tpr >= 0.5
        and different_tnr >= 0.5
        and different_threshold < match_threshold
    )
    return DecisionPolicy(
        score_field=score_field,
        identity_policy=identity_policy,
        different_identity_threshold=different_threshold,
        match_threshold=match_threshold,
        target_match_false_positive_rate=target_match_false_positive_rate,
        target_different_false_reject_rate=target_different_false_reject_rate,
        positive_examples=int(len(positives)),
        negative_examples=int(len(negatives)),
        observed_match_true_positive_rate=match_tpr,
        observed_different_true_negative_rate=different_tnr,
        is_deployable=deployable,
    )
