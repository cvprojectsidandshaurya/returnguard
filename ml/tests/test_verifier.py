"""Tests for the model-agnostic orchestration around the embedding model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import DecisionPolicy  # noqa: E402
from src.quality import QualityResult  # noqa: E402
from src.verifier import ReturnVerifier  # noqa: E402
import src.verifier as verifier_module  # noqa: E402


class DummyEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, paths: list[Path]) -> np.ndarray:
        self.calls += 1
        return np.array([[1.0, 0.0] for _ in paths], dtype=np.float32)


def _policy() -> DecisionPolicy:
    return DecisionPolicy("mean_rider_score", 0.25, 0.75, 0.01, 0.01, 10, 10, 1.0, 1.0)


def test_retake_short_circuits_embedding(monkeypatch):
    embedder = DummyEmbedder()
    monkeypatch.setattr(
        verifier_module,
        "assess_path",
        lambda *_: QualityResult(100, 100, 100.0, 0.0, False, ("resolution_too_low",)),
    )
    result = ReturnVerifier(embedder, _policy()).verify([Path("packing.jpg")], [Path("rider.jpg")])
    assert result.decision == "RETAKE"
    assert embedder.calls == 0
    assert result.retake_reasons == ("packing_0:resolution_too_low", "rider_0:resolution_too_low")


def test_verified_images_use_calibrated_multi_view_score(monkeypatch):
    monkeypatch.setattr(
        verifier_module,
        "assess_path",
        lambda *_: QualityResult(500, 500, 120.0, 100.0, True, ()),
    )
    result = ReturnVerifier(DummyEmbedder(), _policy()).verify([Path("packing.jpg")], [Path("rider.jpg")])
    assert result.decision == "MATCH"
    assert result.score == 1.0
    assert result.local_match is None
