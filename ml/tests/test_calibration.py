"""Tests for validation-only decision calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import DecisionPolicy, fit_decision_policy  # noqa: E402


def test_fit_policy_keeps_a_suspicious_band():
    scores = np.array([0.93, 0.91, 0.90, 0.12, 0.16, 0.19])
    labels = np.array([True, True, True, False, False, False])
    policy = fit_decision_policy(scores, labels, target_match_false_positive_rate=0.1, target_different_false_reject_rate=0.1)
    assert policy.identity_policy == "unit"
    assert policy.is_deployable
    assert policy.decide(0.95) == "MATCH"
    assert policy.decide(0.05) == "DIFFERENT_UNIT"
    assert policy.decide(0.5) == "SUSPICIOUS"


def test_fit_policy_records_overlapping_validation_scores_as_not_deployable():
    scores = np.array([0.20, 0.21, 0.22, 0.90, 0.91, 0.92])
    labels = np.array([True, True, True, False, False, False])
    policy = fit_decision_policy(scores, labels)
    assert not policy.is_deployable
    with pytest.raises(ValueError, match="not deployable"):
        policy.assert_deployable()


def test_policy_round_trip(tmp_path):
    policy = DecisionPolicy("mean_rider_score", "unit", 0.2, 0.8, 0.01, 0.01, 10, 90, 0.9, 0.8, True)
    path = tmp_path / "policy.json"
    policy.save(path)
    assert DecisionPolicy.load(path) == policy
