"""End-to-end, checkpoint-backed return verification without web or UI code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .calibration import DecisionPolicy
from .fine_tuned_embed import FineTunedEmbedder
from .local_features import BestLocalMatch, best_local_match
from .quality import QualityResult, QualityThresholds, assess_path
from .set_matching import MultiViewScore, score_multi_view


@dataclass(frozen=True)
class VerificationResult:
    decision: str
    score: float | None
    score_field: str
    retake_reasons: tuple[str, ...]
    quality: dict[str, dict]
    multi_view: dict | None
    local_match: dict | None

    def as_dict(self) -> dict:
        result = asdict(self)
        result["retake_reasons"] = list(self.retake_reasons)
        return result


class ReturnVerifier:
    """Combine quality, global multi-view, and local-detail evidence for one return."""

    def __init__(
        self,
        embedder: FineTunedEmbedder,
        policy: DecisionPolicy,
        quality_thresholds: QualityThresholds = QualityThresholds(),
    ) -> None:
        self.embedder = embedder
        self.policy = policy
        self.quality_thresholds = quality_thresholds

    @classmethod
    def from_artifacts(
        cls,
        checkpoint: Path | str,
        policy_path: Path | str,
        device: str = "auto",
        quality_thresholds: QualityThresholds = QualityThresholds(),
    ) -> "ReturnVerifier":
        return cls(FineTunedEmbedder(checkpoint, device=device), DecisionPolicy.load(policy_path), quality_thresholds)

    def verify(self, packing_paths: list[Path], rider_paths: list[Path]) -> VerificationResult:
        if not packing_paths or not rider_paths:
            raise ValueError("packing_paths and rider_paths must both be non-empty")
        quality: dict[str, dict] = {}
        retake_reasons: list[str] = []
        for prefix, paths in (("packing", packing_paths), ("rider", rider_paths)):
            for index, path in enumerate(paths):
                key = f"{prefix}_{index}"
                try:
                    check: QualityResult = assess_path(path, self.quality_thresholds)
                    quality[key] = check.as_dict()
                    retake_reasons.extend(f"{key}:{reason}" for reason in check.retake_reasons)
                except Exception as exc:
                    quality[key] = {"passed": False, "retake_reasons": [f"unreadable_image: {exc}"]}
                    retake_reasons.append(f"{key}:unreadable_image")
        if retake_reasons:
            return VerificationResult("RETAKE", None, self.policy.score_field, tuple(retake_reasons), quality, None, None)

        packing_embeddings = self.embedder.encode(packing_paths)
        rider_embeddings = self.embedder.encode(rider_paths)
        multi_view: MultiViewScore = score_multi_view(packing_embeddings, rider_embeddings)
        score = float(getattr(multi_view, self.policy.score_field))
        local: BestLocalMatch | None
        try:
            local = best_local_match(packing_paths, rider_paths)
        except Exception:
            # Local features are optional evidence, never a reason to reject a usable capture.
            local = None
        return VerificationResult(
            decision=self.policy.decide(score),
            score=score,
            score_field=self.policy.score_field,
            retake_reasons=(),
            quality=quality,
            multi_view=multi_view.as_dict(),
            local_match=local.as_dict() if local else None,
        )
